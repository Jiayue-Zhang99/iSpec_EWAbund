import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
import logging
import multiprocessing
from multiprocessing import Pool
import matplotlib.pyplot as plt
from scipy.stats import norm
import re
import json

from config import (
    ATOMIC_Z,
    SOLAR_ABUND_INFO_REL,
    SOLAR_LINES_INFO_REL,
    TC_TABLE_REL,
    resolve_ispec_dir,
)

ISPEC_DIR = resolve_ispec_dir()
TC_PATH = os.path.join(ISPEC_DIR, TC_TABLE_REL)

# ========== For 5_ele_abundance_plot =========
def parse_element_token(s: str):
    """解析元素符号与电离态，返回 (base_element, ionization_stage)"""
    parts = str(s).split()
    base = parts[0]
    try:
        ion = int(parts[1]) if len(parts) > 1 else 1
    except ValueError:
        ion = 1
    return base, ion

# ---------- 合并电离态：把误差改为“标准误差（SEM）” ----------
def combine_ions_sem(g):
    """
    g: 同一 base_element 的各电离态行
    返回：
      mean_xh   : 电离态间 [X/H] 的平均值
      sem_xh    : 电离态间 [X/H] 的标准误差（std/√N），N 为电离态数
                  若 N==1，用该电离态自带的 std_[X/H] 兜底（也可改成 np.nan）
      n_lines   : 各电离态 n_lines 之和
      n_ions    : 该元素参与合并的电离态数
    """
    vals = g["[X/H]"].astype(float).to_numpy()
    k = np.isfinite(vals).sum()
    mean_xh = np.nanmean(vals)

    if k > 1:
        sem_xh = np.nanstd(vals, ddof=1) / np.sqrt(k)
    else:
        # 只有一个电离态时的兜底方式（保留该态的不确定度；也可以设为 np.nan）
        sem_xh = float(g["std_[X/H]"].iloc[0]) if len(g) > 0 else np.nan

    n_lines = g["n_lines"].sum()
    return pd.Series({"[X/H]": mean_xh, "std_[X/H]": sem_xh, "n_lines": n_lines, "n_ions": k})

def mkdir(p):
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def common_elements_on_Tc(td, rd):
    """只保留 target/ref 都有 Tc 的元素，避免错位"""
    common = sorted(set(td["Element"]) & set(rd["Element"]))
    if len(common) > 0:
        td = td[td["Element"].isin(common)]
        rd = rd[rd["Element"].isin(common)]
    return td, rd

# ========== 工具函数 ==========
def load_tc(path=TC_PATH):
    """读取/清洗 Tc 表，只保留 Element 与 Tc（优先 50%Tc_K；退而取 Tc_K），返回 tc_df: Element, Tc_K"""
    tc_raw = pd.read_csv(path)
    # 识别 Tc 列
    tc_col = None
    for c in ["50%Tc_K", "Tc50_K", "Tc_K"]:
        if c in tc_raw.columns:
            tc_col = c
            break
    if tc_col is None:
        raise RuntimeError("在 Tc 表中未找到 Tc 列（期望 50%Tc_K / Tc50_K / Tc_K）")

    tc_df = tc_raw[["Element", tc_col]].rename(columns={tc_col: "Tc_K"}).copy()
    # 清洗：丢掉非数值/缺失、去空格
    tc_df["Element"] = tc_df["Element"].astype(str).str.strip()
    tc_df["Tc_K"] = pd.to_numeric(tc_df["Tc_K"], errors="coerce")
    tc_df = tc_df.dropna(subset=["Tc_K"])
    return tc_df

def build_ref_for_star(star, a1_path, a2_path):
    """
    从 Meléndez2025 的 A46 表生成某颗星的 ref（两种：
      - ref_xh:  Element, [X/H], std_[X/H]
      - ref_xfe: Element, [X/Fe], e_[X/Fe]
    误差：
      - [X/H] = [Fe/H] + [X/Fe]，  σ^2 = σ_FeH^2 + σ_XFe^2（假设独立）
    """
    a1 = pd.read_csv(a1_path)
    a2 = pd.read_csv(a2_path)

    # 列名兜底选择
    def pick(df, names):
        for n in names:
            if n in df.columns:
                return n
        return None

    col_star_a1 = pick(a1, ["Star","Name","main_id","star","name"])
    col_star_a2 = pick(a2, ["Star","Name","main_id","star","name"])
    col_feh     = pick(a1, ["[Fe/H]","Fe_H","FeH"])
    col_efeh    = pick(a1, ["e_[Fe/H]","[Fe/H]_e","e_Fe_H","Fe_H_e","e_FeH","FeH_e"])
    if not all([col_star_a1, col_star_a2, col_feh]):
        raise RuntimeError("A46 表缺少 Star 或 [Fe/H] 列")

    # 取该 star Fe/H
    row_a1 = a1.loc[a1[col_star_a1] == star]
    if row_a1.empty:
        raise RuntimeError(f"在 A46_tablea1 中找不到 {star}")
    feh  = float(row_a1.iloc[0][col_feh])
    efeh = float(row_a1.iloc[0][col_efeh]) if (col_efeh and pd.notna(row_a1.iloc[0][col_efeh])) else np.nan

    # 提取该 star 的所有 [X/Fe] 与误差
    row_a2 = a2.loc[a2[col_star_a2] == star]
    if row_a2.empty:
        raise RuntimeError(f"在 A46_table2 中找不到 {star}")

    # 拉成长表：Element, [X/Fe], e_[X/Fe]
    ratio_cols, err_map = [], {}
    for c in row_a2.columns:
        if isinstance(c, str) and re.fullmatch(r"\[[A-Z][a-z]?/Fe\]", c):
            ratio_cols.append(c)
            # 常见误差列名
            candidates = [f"e_{c}", f"{c}_e"]
            err_map[c] = next((x for x in candidates if x in a2.columns), None)

    recs = []
    for c in ratio_cols:
        val = row_a2.iloc[0][c]
        if pd.isna(val):
            continue
        elem = re.match(r"\[([A-Z][a-z]?)/Fe\]", c).group(1)
        s = row_a2.iloc[0][err_map[c]] if err_map[c] and pd.notna(row_a2.iloc[0][err_map[c]]) else np.nan
        recs.append([elem, float(val), float(s) if pd.notna(s) else np.nan])

    ref_xfe = pd.DataFrame(recs, columns=["Element","[X/Fe]","e_[X/Fe]"])
    ref_xfe["Element"] = ref_xfe["Element"].astype(str).str.strip()

    # 转 [X/H] 与误差
    ref_xh = ref_xfe.copy()
    ref_xh["[X/H]"]    = ref_xh["[X/Fe]"] + feh
    ref_xh["std_[X/H]"] = np.sqrt(
        np.where(np.isfinite(ref_xh["e_[X/Fe]"]), ref_xh["e_[X/Fe]"]**2, 0.0) +
        (efeh**2 if np.isfinite(efeh) else 0.0)
    )

    # 加 Fe 自身
    ref_xh = pd.concat([ref_xh, pd.DataFrame([["Fe", feh, efeh, 1]], columns=["Element","[X/H]","std_[X/H]","_dummy"])], ignore_index=True)
    ref_xfe = pd.concat([ref_xfe, pd.DataFrame([["Fe", 0.0, efeh, ]], columns=["Element","[X/Fe]","e_[X/Fe]"])], ignore_index=True)

    return ref_xh[["Element","[X/H]","std_[X/H]"]], ref_xfe[["Element","[X/Fe]","e_[X/Fe]"]]

def _parse_bedell2018_token(colname: str):
    """
    输入: 形如 '[ScII/H]'、'[CI/H]'、'[CH/H]'
    输出: (base_element, tag) 其中 tag 只是用于区分来源，不影响最后 base_element 合并
    """
    s = str(colname).strip()
    if not (s.startswith("[") and s.endswith("]") and "/H" in s):
        return None, None

    # 取出括号内、'/H' 之前的 token：如 'ScII', 'CI', 'CH'
    token = s[1:s.index("/H")]

    # 特殊：CH -> 归到 C（Bedell 里常用 CH 给 C）
    if token.upper() == "CH":
        return "C", "CH"

    # 解析元素符号（1~2位：首字母大写，可能第二位小写）
    base = token[0].upper()
    rest = token[1:]
    if len(rest) >= 1 and rest[0].islower():
        base += rest[0]
        rest = rest[1:]

    # rest 可能是 I/II/III... 或空
    tag = rest if rest else "I"
    return base, tag


def build_ref_for_star_bedell2018_table2(
    star: str,
    table2_path: str,
    feh_override: float | None = None,
    e_feh_override: float | None = None,
    feh_table_path: str | None = None,          # <- 新增
    feh_source_hint: str = "Bedell_2018",       # <- 新增
):
    """
    用 Bedell2018_table2.csv 生成 ref。
    - ref_xh: Element, [X/H], std_[X/H]
    - ref_xfe: Element, [X/Fe], e_[X/Fe] （需要 feh_override；否则全 NaN）

    table2 的典型列:
      Star, [CI/H], e_[CI/H], [CH/H], e_[CH/H], ...（不含 Fe）
    """
    t2 = pd.read_csv(table2_path)

    star_col = "Star" if "Star" in t2.columns else None
    if star_col is None:
        raise RuntimeError(f"[Bedell2018 ref] 在 {table2_path} 找不到 'Star' 列")

    # 找到该 star 的行
    star_key = str(star).strip()
    sel = t2[star_col].astype(str).str.strip() == star_key
    if not sel.any():
        raise RuntimeError(f"[Bedell2018 ref] 在 table2 找不到 star='{star_key}'")

    row = t2.loc[sel].iloc[0]

    # 组装成“长表”，复用你现有的 combine_ions_for_one_target()
    rows = []
    for c in t2.columns:
        if not (isinstance(c, str) and c.startswith("[") and c.endswith("]") and "/H" in c):
            continue

        base, tag = _parse_bedell2018_token(c)
        if base is None:
            continue

        xh = pd.to_numeric(row.get(c), errors="coerce")
        err = pd.to_numeric(row.get("e_" + c), errors="coerce")  # Bedell 的误差列是 'e_[X/H]'

        if not np.isfinite(xh):
            continue

        rows.append({
            "element": f"{base} {tag}",     # 只为能被 base_element 合并；tag 无所谓
            "[X/H]": float(xh),
            "std_[X/H]": float(err) if np.isfinite(err) else np.nan,
            "n_lines": 1,                   # Bedell 表里没有逐线条数，就先给 1
        })

    df_long = pd.DataFrame(rows)
    if df_long.empty:
        raise RuntimeError(f"[Bedell2018 ref] star='{star_key}' 没解析到任何 [X/H] 列")

    g = combine_ions_for_one_target(df_long)  # -> base_element, [X/H], std_[X/H], n_lines, n_ions

    # 如果你希望 ref_xfe 也可用：必须补一个 Fe 行
    # ----------------- 自动获取 Fe/H（若没手动 override） -----------------
    if feh_override is None and feh_table_path is not None:
        feh_override, e_feh_override = read_feh_from_abundance_with_gaia(
            star=star,
            csv_path=feh_table_path,
            source_hint=feh_source_hint,
        )

    # ----------------- 把 Fe 行拼进 ref，再算 [X/Fe] -----------------
    if feh_override is not None:
        g = pd.concat([g, pd.DataFrame([{
            "base_element": "Fe",
            "[X/H]": float(feh_override),
            "std_[X/H]": float(e_feh_override) if e_feh_override is not None else np.nan,
            "n_lines": 0,
            "n_ions": 1
        }])], ignore_index=True)

    g2 = add_xfe_and_err(g)

    ref_xh = g2.rename(columns={"base_element": "Element"})[["Element", "[X/H]", "std_[X/H]"]]
    ref_xfe = g2.rename(columns={"base_element": "Element"})[["Element", "[X/Fe]", "e_[X/Fe]"]]
    return ref_xh, ref_xfe


def build_ref_for_star_choice(
    star: str,
    ref_name: str,
    bedell_table2_path: str | None = None,
    a46_a1_path: str | None = None,
    a46_a2_path: str | None = None,
    feh_override: float | None = None,
    e_feh_override: float | None = None,
    feh_table_path: str | None = None,
):
    """
    统一入口：根据 ref_name 选择不同 ref 构建方式
    - ref_name='A46'：走你原来的 build_ref_for_star（需要 a1/a2）
    - ref_name='Bedell2018'：走 Bedell table2（需要 bedell_table2_path；若要 X/Fe 还要 feh_override）
    """
    ref_name = str(ref_name).strip()

    if ref_name.lower() in ["a46", "melendez2025"]:
        # 复用你已有函数：build_ref_for_star(star, a1_path, a2_path)
        if a46_a1_path is None or a46_a2_path is None:
            raise RuntimeError("ref_name='A46' 需要同时提供 a46_a1_path 和 a46_a2_path")
        return build_ref_for_star(star, a1_path=a46_a1_path, a2_path=a46_a2_path)

    if ref_name.lower() in ["bedell2018", "bedell"]:
        if bedell_table2_path is None:
            raise RuntimeError("ref_name='Bedell2018' 需要提供 bedell_table2_path（即 Bedell2018_table2.csv）")
        return build_ref_for_star_bedell2018_table2(
            star=star,
            table2_path=bedell_table2_path,
            feh_override=feh_override,
            e_feh_override=e_feh_override,
            feh_table_path=feh_table_path,     # <- 透传
            # feh_source_hint=... (可选，不写就默认 Bedell_2018)
        )

    raise RuntimeError(f"未知 ref_name: {ref_name}")

def add_tc_to(df_like, tc_df):
    """把 Tc 合并进来，要求 df_like 里有列 'Element' 或 'base_element'"""
    d = df_like.copy()
    if "Element" not in d.columns and "base_element" in d.columns:
        d = d.rename(columns={"base_element":"Element"})
    d = d.merge(tc_df, on="Element", how="left")
    return d

def ensure_element_col(d):
    """把 df_combined（base_element）或 df（含 element/base_element）规范成 Element 列"""
    out = d.copy()
    if "Element" in out.columns:
        return out
    if "base_element" in out.columns:
        out = out.rename(columns={"base_element":"Element"})
        return out
    if "element" in out.columns:
        out["Element"] = out["element"].str.split().str[0]
        return out
    raise RuntimeError("DataFrame 里没有 Element / base_element / element 这三种可用列")



# ---------- 画图函数（migrated from 5_ele_abundance_plot.ipynb） ----------
# ---------- 画图函数 ----------
# ========== fig01 ==========
def plot_xh_by_Z_with_ion(df_in, target, outdir):
    """
    fig01：按 (Z, ion) 排序的 [X/H]，区分 Fe 与其他元素；横轴保留电离态标签；标注每个电离态的 n_lines
    输入 df_in 列要求：element, base_element, ion, Z, [X/H], std_[X/H], n_lines
    """
    mkdir(outdir)
    d = df_in.copy()
    order = d.sort_values(["Z","ion","base_element","element"])["element"].tolist()
    d["element"] = pd.Categorical(d["element"], categories=order, ordered=True)
    d = d.sort_values("element")

    plt.figure(figsize=(12,6), dpi=192)
    mask_fe = d["element"].str.startswith("Fe")
    plt.errorbar(d.loc[~mask_fe,"element"], d.loc[~mask_fe,"[X/H]"],
                 yerr=d.loc[~mask_fe,"std_[X/H]"], fmt="o",
                 color="blue", ecolor="gray", elinewidth=1.5, capsize=4, label="Other elements")
    plt.errorbar(d.loc[ mask_fe,"element"], d.loc[ mask_fe,"[X/H]"],
                 yerr=d.loc[ mask_fe,"std_[X/H]"], fmt="o",
                 color="red",  ecolor="gray", elinewidth=1.5, capsize=4, label="Fe")

    for _, r in d.iterrows():
        plt.text(r["element"], r["[X/H]"]+0.03, f"{int(r['n_lines'])}",
                 ha="center", va="bottom", fontsize=9)

    plt.axhline(0, color="k", ls="--", lw=1)
    plt.xlabel("Element (with ionization stage)", fontsize=14)
    plt.ylabel("[X/H] (dex)", fontsize=14)
    plt.title(f"{target}, [X/H] sorted by Z, then ion", fontsize=16)
    plt.xticks(rotation=45, ha="right")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "01_XH_ordered_by_Z_with_ion.png"), dpi=192)

# ========== fig02 ==========
def plot_xh_by_Z_combined(df_in, target, outdir):
    """
    fig02：合并电离态后的 [X/H]（误差=电离态间 SEM，或单一电离态的自身不确定度），按 Z 排序
    输入 df_in 列要求：base_element, [X/H], std_[X/H], n_lines, Z
    """
    mkdir(outdir)
    d = df_in.sort_values("Z").copy()
    mask_fe = d["base_element"]=="Fe"

    plt.figure(figsize=(12,6), dpi=192)
    plt.errorbar(d.loc[~mask_fe,"base_element"], d.loc[~mask_fe,"[X/H]"],
                 yerr=d.loc[~mask_fe,"std_[X/H]"], fmt="o", color="blue",
                 ecolor="gray", elinewidth=1.5, capsize=4, markersize=10, label="Other elements")
    plt.errorbar(d.loc[ mask_fe,"base_element"], d.loc[ mask_fe,"[X/H]"],
                 yerr=d.loc[ mask_fe,"std_[X/H]"], fmt="o", color="red",
                 ecolor="gray", elinewidth=1.5, capsize=4, markersize=10, label="Fe")

    for _, r in d.iterrows():
        plt.text(r["base_element"], r["[X/H]"]+0.03, f"{int(r['n_lines'])}",
                 ha="center", va="bottom", fontsize=12)

    plt.axhline(0, color="k", ls="--", lw=1)
    plt.ylabel("[X/H] (dex)", fontsize=14)
    plt.title(f"{target}, [X/H] sorted by Z (ions combined)", fontsize=16)
    plt.xticks(d["base_element"].unique(), rotation=45, ha="right", fontsize=13)
    plt.yticks(fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "02_XH_ordered_by_Z.png"), dpi=192)

# ========== fig03 ==========
def plot_xh_by_Tc_ordered(target_df, target, outdir):
    """
    fig03：按 Tc 排序（横轴显示元素名），y=[X/H]，无 ref
    输入 target_df 列要求：Element, [X/H], std_[X/H], n_lines, Tc_K
    """
    mkdir(outdir)
    d = target_df.dropna(subset=["Tc_K"]).copy().sort_values(["Tc_K","Element"])
    xticks = d["Element"].tolist()

    plt.figure(figsize=(12,6), dpi=192)
    plt.errorbar(xticks, d["[X/H]"], yerr=d["std_[X/H]"],
                 fmt="o", color="blue", ecolor="gray", elinewidth=1.5, capsize=4)
    for _, r in d.iterrows():
        n_val = pd.to_numeric(r.get("n_lines", 0), errors="coerce")
        n_show = int(n_val) if np.isfinite(n_val) else 0
        plt.text(r["Element"], r["[X/H]"]+0.03, f"{n_show}",
                 ha="center", va="bottom", fontsize=9)
    plt.axhline(0, color="k", ls="--", lw=1)
    plt.ylabel("[X/H] (dex)", fontsize=14)
    plt.title(f"{target}, [X/H] ordered by T$_c$", fontsize=16)
    plt.xticks(xticks, rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "03_XH_ordered_by_Tc.png"), dpi=192)

# ========== fig04 ==========
def plot_xh_vs_Tc_labels(target_df, target, outdir):
    """
    fig04：y=[X/H]，x=Tc 数值，元素名称标在点上方，无 ref
    输入 target_df 列要求：Element, [X/H], std_[X/H], Tc_K
    """
    mkdir(outdir)
    d = target_df.dropna(subset=["Tc_K"]).copy().sort_values(["Tc_K","Element"])
    plt.figure(figsize=(12,6), dpi=192)
    plt.errorbar(d["Tc_K"], d["[X/H]"], yerr=d["std_[X/H]"],
                 fmt="o", color="blue", ecolor="gray", elinewidth=1.5, capsize=4)
    for _, r in d.iterrows():
        plt.text(r["Tc_K"], r["[X/H]"]+0.03, r["Element"], ha="center", va="bottom", fontsize=10)
    plt.axhline(0, color="k", ls="--", lw=1)
    plt.xlabel("Condensation temperature T$_c$ (K)", fontsize=14)
    plt.ylabel("[X/H] (dex)", fontsize=14)
    plt.title(f"{target}, [X/H] vs T$_c$", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "04_XH_vs_Tc_labels_Element.png"), dpi=192)

# ========== fig05 ==========
def plot_xh_by_Tc_with_ref(target_df, ref_df, target, ref_name, outdir):
    """
    fig05：按 Tc 排序（横轴为元素名），y=[X/H]，有 ref 对比
    输入两表列要求：Element, [X/H], std_[X/H], n_lines, Tc_K
    """
    mkdir(outdir)
    td = target_df.dropna(subset=["Tc_K"]).copy().sort_values(["Tc_K","Element"])
    rd = ref_df.dropna(subset=["Tc_K"]).copy().sort_values(["Tc_K","Element"])
    rd = rd[rd["Element"].isin(td["Element"])]
    td = td[td["Element"].isin(rd["Element"])]
    xticks = td["Element"].tolist()

    plt.figure(figsize=(12,6), dpi=192)
    plt.errorbar(xticks, td["[X/H]"], yerr=td["std_[X/H]"],
                 fmt="o", color="blue", ecolor="gray", elinewidth=1.5, capsize=4, label="Target")
    plt.errorbar(xticks, rd["[X/H]"], yerr=rd["std_[X/H]"],
                 fmt="o", mfc="white", mec="green", ecolor="green",
                 elinewidth=1.2, capsize=3, label=f"Ref ({ref_name})")
    for _, r in td.iterrows():
        plt.text(r["Element"], r["[X/H]"]+0.03, f"{int(r.get('n_lines',0))}",
                 ha="center", va="bottom", fontsize=9)
    plt.axhline(0, color="k", ls="--", lw=1)
    plt.ylabel("[X/H] (dex)", fontsize=14)
    plt.title(f"{target}, [X/H] ordered by T$_c$ with Ref", fontsize=16)
    plt.xticks(xticks, rotation=45, ha="right")
    plt.ylim(-0.2,0.2)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "05_XH_ordered_by_Tc_withRef.png"), dpi=192)

# ========== fig06 ==========
def plot_xfe_by_Tc_with_ref(target_xfe_df, ref_xfe_df, target, ref_name, outdir):
    """
    fig06：按 Tc 排序（横轴为元素名），y=[X/Fe]，有 ref 对比
    输入两表列要求：Element, [X/Fe], e_[X/Fe], Tc_K
    """
    mkdir(outdir)
    td = target_xfe_df.dropna(subset=["Tc_K"]).copy().sort_values(["Tc_K","Element"])
    rd = ref_xfe_df.dropna(subset=["Tc_K"]).copy().sort_values(["Tc_K","Element"])
    td = td[td["Element"].isin(rd["Element"])]
    rd = rd[rd["Element"].isin(td["Element"])]
    xticks = td["Element"].tolist()

    plt.figure(figsize=(12,6), dpi=192)
    plt.errorbar(xticks, td["[X/Fe]"], yerr=td["e_[X/Fe]"],
                 fmt="o", color="blue", ecolor="gray", elinewidth=1.5, capsize=4, label="Target")
    plt.errorbar(xticks, rd["[X/Fe]"], yerr=rd["e_[X/Fe]"],
                 fmt="o", mfc="white", mec="green", ecolor="green",
                 elinewidth=1.2, capsize=3, label=f"Ref ({ref_name})")
    plt.axhline(0, color="k", ls="--", lw=1)
    plt.ylabel("[X/Fe] (dex)", fontsize=14)
    plt.title(f"{target}, [X/Fe] ordered by T$_c$ with Ref", fontsize=16)
    plt.xticks(xticks, rotation=45, ha="right")
    plt.ylim(-0.2,0.2)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "06_XFe_ordered_by_Tc_withRef.png"), dpi=192)

# ========== fig06b ==========
def plot_xh_by_Tc_with_ref_and_nohfs(target_hfs_df, target_nohfs_df, ref_df, target, ref_name, outdir):
    """
    fig06b：按 Tc 排序（横轴为元素名），y=[X/H]，同时显示 Target(HFS) / Target(noHFS) / Ref
    其中 noHFS 仅绘制 hfs=y 的元素（红色）。
    """
    mkdir(outdir)
    td_hfs = target_hfs_df.dropna(subset=["Tc_K"]).copy().sort_values(["Tc_K", "Element"])
    rd = ref_df.dropna(subset=["Tc_K"]).copy().sort_values(["Tc_K", "Element"])

    # 蓝色/绿色主序列：按 HFS 与 ref 共同元素
    common_main = sorted(set(td_hfs["Element"]) & set(rd["Element"]))
    td_hfs = td_hfs[td_hfs["Element"].isin(common_main)].sort_values(["Tc_K", "Element"])
    rd = rd[rd["Element"].isin(common_main)].sort_values(["Tc_K", "Element"])
    xticks = td_hfs["Element"].tolist()
    x = np.arange(len(xticks))
    pos = {e: i for i, e in enumerate(xticks)}

    # 红色 no-HFS：仅 hfs=y 且有数值
    td_no = target_nohfs_df.dropna(subset=["Tc_K"]).copy()
    if "hfs" in td_no.columns:
        td_no = td_no[td_no["hfs"].astype(str).str.lower() == "y"]
    td_no = td_no[td_no["Element"].isin(xticks)].copy()
    td_no = td_no[np.isfinite(pd.to_numeric(td_no.get("[X/H]"), errors="coerce"))]
    td_no = td_no.sort_values(["Tc_K", "Element"])

    plt.figure(figsize=(13, 6), dpi=192)
    plt.errorbar(x, td_hfs["[X/H]"], yerr=td_hfs["std_[X/H]"],
                 fmt="o", color="blue", ecolor="gray", elinewidth=1.5, capsize=4, label="Target (HFS)")

    if len(td_no):
        x_no = [pos[e] for e in td_no["Element"].tolist() if e in pos]
        y_no = pd.to_numeric(td_no["[X/H]"], errors="coerce")
        e_no = pd.to_numeric(td_no.get("std_[X/H]", np.nan), errors="coerce")
        plt.errorbar(x_no, y_no, yerr=e_no,
                     fmt="o", mfc="white", mec="red", color="red", ecolor="red",
                     elinewidth=1.2, capsize=3, label="Target (no HFS)")

    rd_aligned = rd.set_index("Element").loc[xticks].reset_index()
    plt.errorbar(x, rd_aligned["[X/H]"], yerr=rd_aligned["std_[X/H]"],
                 fmt="o", mfc="white", mec="green", ecolor="green", elinewidth=1.2, capsize=3,
                 label=f"Ref ({ref_name})")

    plt.axhline(0, color="k", ls="--", lw=1)
    plt.ylabel("[X/H] (dex)", fontsize=14)
    plt.title(f"{target}, [X/H] ordered by T$_c$ with Ref and no-HFS comparison", fontsize=16)
    plt.xticks(x, xticks, rotation=45, ha="right")
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "06b_XH_ordered_by_Tc_withRef_and_noHFS.png"), dpi=192)

# ========== fig06c ==========
def plot_xfe_by_Tc_with_ref_and_nohfs(target_hfs_xfe_df, target_nohfs_xfe_df, ref_xfe_df, target, ref_name, outdir):
    """
    fig06c：按 Tc 排序（横轴为元素名），y=[X/Fe]，同时显示 Target(HFS) / Target(noHFS) / Ref
    其中 noHFS 仅绘制 hfs=y 的元素（红色）。
    """
    mkdir(outdir)
    td_hfs = target_hfs_xfe_df.dropna(subset=["Tc_K"]).copy().sort_values(["Tc_K", "Element"])
    rd = ref_xfe_df.dropna(subset=["Tc_K"]).copy().sort_values(["Tc_K", "Element"])

    common_main = sorted(set(td_hfs["Element"]) & set(rd["Element"]))
    td_hfs = td_hfs[td_hfs["Element"].isin(common_main)].sort_values(["Tc_K", "Element"])
    rd = rd[rd["Element"].isin(common_main)].sort_values(["Tc_K", "Element"])
    xticks = td_hfs["Element"].tolist()
    x = np.arange(len(xticks))
    pos = {e: i for i, e in enumerate(xticks)}

    td_no = target_nohfs_xfe_df.dropna(subset=["Tc_K"]).copy()
    if "hfs" in td_no.columns:
        td_no = td_no[td_no["hfs"].astype(str).str.lower() == "y"]
    td_no = td_no[td_no["Element"].isin(xticks)].copy()
    td_no = td_no[np.isfinite(pd.to_numeric(td_no.get("[X/Fe]"), errors="coerce"))]
    td_no = td_no.sort_values(["Tc_K", "Element"])

    plt.figure(figsize=(13, 6), dpi=192)
    plt.errorbar(x, td_hfs["[X/Fe]"], yerr=td_hfs["e_[X/Fe]"],
                 fmt="o", color="blue", ecolor="gray", elinewidth=1.5, capsize=4, label="Target (HFS)")

    if len(td_no):
        x_no = [pos[e] for e in td_no["Element"].tolist() if e in pos]
        y_no = pd.to_numeric(td_no["[X/Fe]"], errors="coerce")
        e_no = pd.to_numeric(td_no.get("e_[X/Fe]", np.nan), errors="coerce")
        plt.errorbar(x_no, y_no, yerr=e_no,
                     fmt="o", mfc="white", mec="red", color="red", ecolor="red",
                     elinewidth=1.2, capsize=3, label="Target (no HFS)")

    rd_aligned = rd.set_index("Element").loc[xticks].reset_index()
    plt.errorbar(x, rd_aligned["[X/Fe]"], yerr=rd_aligned["e_[X/Fe]"],
                 fmt="o", mfc="white", mec="green", ecolor="green", elinewidth=1.2, capsize=3,
                 label=f"Ref ({ref_name})")

    plt.axhline(0, color="k", ls="--", lw=1)
    plt.ylabel("[X/Fe] (dex)", fontsize=14)
    plt.title(f"{target}, [X/Fe] ordered by T$_c$ with Ref and no-HFS comparison", fontsize=16)
    plt.xticks(x, xticks, rotation=45, ha="right")
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "06c_XFe_ordered_by_Tc_withRef_and_noHFS.png"), dpi=192)

# ========== fig07 ==========
def plot_xh_vs_Tc_with_ref_labels(target_df, ref_df, target, ref_name, outdir):
    """
    fig07：y=[X/H]，x=Tc，元素名标在点上方，有 ref 对比
    输入两表列要求：Element, [X/H], std_[X/H], Tc_K
    """
    mkdir(outdir)
    td = target_df.dropna(subset=["Tc_K"]).copy().sort_values(["Tc_K","Element"])
    rd = ref_df.dropna(subset=["Tc_K"]).copy().sort_values(["Tc_K","Element"])
    td, rd = common_elements_on_Tc(td, rd)

    plt.figure(figsize=(12,6), dpi=192)
    plt.errorbar(td["Tc_K"], td["[X/H]"], yerr=td["std_[X/H]"],
                 fmt="o", color="blue", ecolor="gray", elinewidth=1.5, capsize=4, label="Target")
    plt.errorbar(rd["Tc_K"], rd["[X/H]"], yerr=rd["std_[X/H]"],
                 fmt="o", mfc="white", mec="green", ecolor="green",
                 elinewidth=1.2, capsize=3, label=f"Ref ({ref_name})")

    for _, r in td.iterrows():
        plt.text(r["Tc_K"], r["[X/H]"]+0.03, r["Element"], ha="center", va="bottom", fontsize=10)
    for _, r in rd.iterrows():
        plt.text(r["Tc_K"], r["[X/H]"]-0.03, r["Element"], ha="center", va="top", fontsize=10, color="green")

    plt.axhline(0, color="k", ls="--", lw=1)
    plt.xlabel("Condensation temperature T$_c$ (K)", fontsize=14)
    plt.ylabel("[X/H] (dex)", fontsize=14)
    plt.title(f"{target}, [X/H] vs T$_c$ with Ref", fontsize=16)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "07_XH_vs_Tc_labels_withRef.png"), dpi=192)

# ========== fig08 ==========
def plot_xfe_vs_Tc_with_ref_labels(target_xfe_df, ref_xfe_df, target, ref_name, outdir):
    """
    fig08：y=[X/Fe]，x=Tc，元素名标在点上方，有 ref 对比
    输入两表列要求：Element, [X/Fe], e_[X/Fe], Tc_K
    """
    mkdir(outdir)
    td = target_xfe_df.dropna(subset=["Tc_K"]).copy().sort_values(["Tc_K","Element"])
    rd = ref_xfe_df.dropna(subset=["Tc_K"]).copy().sort_values(["Tc_K","Element"])
    td, rd = common_elements_on_Tc(td, rd)

    plt.figure(figsize=(12,6), dpi=192)
    plt.errorbar(td["Tc_K"], td["[X/Fe]"], yerr=td["e_[X/Fe]"],
                 fmt="o", color="blue", ecolor="gray", elinewidth=1.5, capsize=4, label="Target")
    plt.errorbar(rd["Tc_K"], rd["[X/Fe]"], yerr=rd["e_[X/Fe]"],
                 fmt="o", mfc="white", mec="green", ecolor="green",
                 elinewidth=1.2, capsize=3, label=f"Ref ({ref_name})")

    for _, r in td.iterrows():
        plt.text(r["Tc_K"], r["[X/Fe]"]+0.03, r["Element"], ha="center", va="bottom", fontsize=10)
    for _, r in rd.iterrows():
        plt.text(r["Tc_K"], r["[X/Fe]"]-0.03, r["Element"], ha="center", va="top", fontsize=10, color="green")

    plt.axhline(0, color="k", ls="--", lw=1)
    plt.xlabel("Condensation temperature T$_c$ (K)", fontsize=14)
    plt.ylabel("[X/Fe] (dex)", fontsize=14)
    plt.title(f"{target}, [X/Fe] vs T$_c$ with Ref", fontsize=16)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "08_XFe_vs_Tc_labels_withRef.png"), dpi=192)

# ========== Step 5 pipeline runner ==========
def _step5_agg_hfs_flag(vals):
    vv = {str(v).strip().lower() for v in vals if str(v).strip() != ""}
    if "y" in vv:
        return "y"
    if "f" in vv:
        return "f"
    return "n"


def _step5_build_xfe_from_xh(xh_df):
    fe_row = xh_df.loc[xh_df["Element"] == "Fe"].iloc[0] if (xh_df["Element"] == "Fe").any() else None
    if fe_row is not None:
        xfe = xh_df.copy()
        xfe["[X/Fe]"] = xfe["[X/H]"] - fe_row["[X/H]"]
        xfe["e_[X/Fe]"] = np.sqrt(
            np.where(np.isfinite(xfe["std_[X/H]"]), xfe["std_[X/H]"] ** 2, 0.0)
            + (fe_row["std_[X/H]"] ** 2 if np.isfinite(fe_row["std_[X/H]"]) else 0.0)
        )
    else:
        xfe = xh_df.assign(**{"[X/Fe]": np.nan, "e_[X/Fe]": np.nan})
    return xfe


def run_step5_ele_abundance_plots(
    *,
    target,
    output_abundances_result_path_linebyline_q2,
    abundance_plot_folder,
    output_folder,
    ref_name,
    ref_star,
    pipeline_options=None,
    ispec_dir=None,
):
    print("[STEP 1] Loading abundance summary and preparing figure output folder.")
    df = pd.read_csv(output_abundances_result_path_linebyline_q2)
    os.makedirs(abundance_plot_folder, exist_ok=True)

    print("[STEP 2] Building element and ion level tables (fig01/fig02).")
    tmp = df["element"].apply(parse_element_token)
    df["base_element"] = tmp.apply(lambda x: x[0])
    df["ion"] = tmp.apply(lambda x: x[1])
    df["Z"] = df["base_element"].map(ATOMIC_Z).fillna(9999)
    df_combined = df.groupby("base_element", as_index=False).apply(combine_ions_sem)
    df_combined["Z"] = df_combined["base_element"].map(ATOMIC_Z)
    plot_xh_by_Z_with_ion(df, target=target, outdir=abundance_plot_folder)
    plot_xh_by_Z_combined(df_combined, target=target, outdir=abundance_plot_folder)

    print("[STEP 3] Preparing Tc-based target tables and HFS flags.")
    tc_df = load_tc(TC_PATH)
    summary_path = output_abundances_result_path_linebyline_q2
    raw_path = summary_path.replace('.csv', '_raw.csv')

    df_summary_src = pd.read_csv(summary_path)
    df_summary_src["element"] = df_summary_src["element"].astype(str)
    if "hfs" not in df_summary_src.columns:
        df_summary_src["hfs"] = "n"

    tmp_sum = df_summary_src["element"].apply(parse_element_token)
    df_summary_src["base_element"] = tmp_sum.apply(lambda x: x[0])
    df_summary_src["ion"] = tmp_sum.apply(lambda x: x[1])
    df_summary_src["[X/H]"] = pd.to_numeric(df_summary_src.get("[X/H]"), errors="coerce")
    df_summary_src["std_[X/H]"] = pd.to_numeric(df_summary_src.get("std_[X/H]"), errors="coerce")
    df_summary_src["n_lines"] = pd.to_numeric(df_summary_src.get("n_lines"), errors="coerce")

    df_combined_hfs = df_summary_src.groupby("base_element", as_index=False).apply(combine_ions_sem)
    df_combined_hfs["Z"] = df_combined_hfs["base_element"].map(ATOMIC_Z)
    target_xh = ensure_element_col(df_combined_hfs)[["Element", "[X/H]", "std_[X/H]", "n_lines", "Z"]].copy()

    flag_sum = df_summary_src[["base_element", "hfs"]].copy()
    flag_sum["hfs"] = flag_sum["hfs"].astype(str)
    flag_sum = flag_sum.groupby("base_element", as_index=False)["hfs"].agg(_step5_agg_hfs_flag)

    if os.path.exists(raw_path):
        raw_df = pd.read_csv(raw_path)
        raw_df["element"] = raw_df["element"].astype(str)
        if "hfs" in raw_df.columns:
            tmp_raw = raw_df["element"].apply(parse_element_token)
            raw_df["base_element"] = tmp_raw.apply(lambda x: x[0])
            h_raw_num = pd.to_numeric(raw_df["hfs"], errors="coerce")
            raw_df["_hraw"] = np.where(np.isfinite(h_raw_num), "y", raw_df["hfs"].astype(str).str.strip().str.lower())
            flag_raw = raw_df.groupby("base_element", as_index=False)["_hraw"].agg(_step5_agg_hfs_flag).rename(columns={"_hraw": "hfs_raw"})
        else:
            flag_raw = pd.DataFrame(columns=["base_element", "hfs_raw"])
    else:
        flag_raw = pd.DataFrame(columns=["base_element", "hfs_raw"])

    hfs_flag_elem = flag_sum.merge(flag_raw, on="base_element", how="outer")
    hfs_flag_elem["hfs"] = hfs_flag_elem.apply(
        lambda r: _step5_agg_hfs_flag([str(r.get("hfs", "n")), str(r.get("hfs_raw", "n"))]), axis=1
    )
    hfs_flag_elem = hfs_flag_elem.rename(columns={"base_element": "Element"})[["Element", "hfs"]]

    target_xh = target_xh.merge(hfs_flag_elem, on="Element", how="left")
    target_xh["hfs"] = target_xh["hfs"].fillna("n")
    target_xh = add_tc_to(target_xh, tc_df)
    target_xfe = _step5_build_xfe_from_xh(target_xh)

    pipeline_options = pipeline_options or {}
    plot_hfs_with_external_ew = bool(pipeline_options.get("plot_hfs_with_external_ews", False))
    use_hfs_plots = (not pipeline_options.get("input_has_external_ews")) or plot_hfs_with_external_ew
    if pipeline_options.get("input_has_external_ews") and plot_hfs_with_external_ew:
        print("[STEP 3] plot_hfs_with_external_ews=True, still drawing HFS/no-HFS compare figures.")

    nohfs_summary_path = os.path.join(output_folder, "HFS", "tables", f"abundances_nohfs_blends_{target}_summary.csv")
    target_xh_nohfs = target_xh[["Element", "Z", "hfs"]].copy()
    target_xh_nohfs["[X/H]"] = np.nan
    target_xh_nohfs["std_[X/H]"] = np.nan
    target_xh_nohfs["n_lines"] = np.nan

    target_xfe_nohfs = target_xfe[["Element", "Z", "hfs"]].copy()
    target_xfe_nohfs["[X/Fe]"] = np.nan
    target_xfe_nohfs["e_[X/Fe]"] = np.nan

    if use_hfs_plots:
        if os.path.exists(nohfs_summary_path):
            nohfs_sum = pd.read_csv(nohfs_summary_path)
            nohfs_sum["element"] = nohfs_sum["element"].astype(str)
            nohfs_sum = nohfs_sum.rename(columns={
                "element": "Element",
                "xh_mean": "nohfs_xh",
                "xh_std": "nohfs_xh_std",
                "xfe_mean": "nohfs_xfe",
                "xfe_std": "nohfs_xfe_std",
                "n_lines": "nohfs_n_lines",
            })
            target_xh_nohfs = target_xh_nohfs.merge(
                nohfs_sum[["Element", "nohfs_xh", "nohfs_xh_std", "nohfs_n_lines"]],
                on="Element", how="left"
            )
            mask_y = target_xh_nohfs["hfs"].astype(str).str.lower() == "y"
            target_xh_nohfs.loc[mask_y, "[X/H]"] = pd.to_numeric(target_xh_nohfs.loc[mask_y, "nohfs_xh"], errors="coerce")
            target_xh_nohfs.loc[mask_y, "std_[X/H]"] = pd.to_numeric(target_xh_nohfs.loc[mask_y, "nohfs_xh_std"], errors="coerce")
            target_xh_nohfs.loc[mask_y, "n_lines"] = pd.to_numeric(target_xh_nohfs.loc[mask_y, "nohfs_n_lines"], errors="coerce")
            target_xh_nohfs = target_xh_nohfs.drop(columns=["nohfs_xh", "nohfs_xh_std", "nohfs_n_lines"])

            target_xfe_nohfs = target_xfe_nohfs.merge(
                nohfs_sum[["Element", "nohfs_xfe", "nohfs_xfe_std"]],
                on="Element", how="left"
            )
            mask_y2 = target_xfe_nohfs["hfs"].astype(str).str.lower() == "y"
            target_xfe_nohfs.loc[mask_y2, "[X/Fe]"] = pd.to_numeric(target_xfe_nohfs.loc[mask_y2, "nohfs_xfe"], errors="coerce")
            target_xfe_nohfs.loc[mask_y2, "e_[X/Fe]"] = pd.to_numeric(target_xfe_nohfs.loc[mask_y2, "nohfs_xfe_std"], errors="coerce")
            target_xfe_nohfs = target_xfe_nohfs.drop(columns=["nohfs_xfe", "nohfs_xfe_std"])
        else:
            print(f"[warn] no-HFS summary not found for fig06b/06c: {nohfs_summary_path}")
    else:
        print("[STEP 3] Skip no-HFS summary when external EWs are used.")

    target_xh_nohfs = add_tc_to(target_xh_nohfs, tc_df)
    target_xfe_nohfs = add_tc_to(target_xfe_nohfs, tc_df)

    print("[STEP 4] Building reference abundance table.")
    if ispec_dir is None:
        ispec_dir = resolve_ispec_dir()
    if ref_name == "Bedell2018":
        bedell_table2 = os.path.join(ispec_dir, "input/abundances/D_PASTA/BedellHARPS/Bedell2018_table2.csv")
        bedell_feh_table = os.path.join(ispec_dir, "input/abundances/D_PASTA/BedellHARPS/abundance_with_gaia.csv")
        ref_xh, ref_xfe = build_ref_for_star_choice(
            star=ref_star,
            ref_name="Bedell2018",
            bedell_table2_path=bedell_table2,
            feh_table_path=bedell_feh_table,
        )
    elif ref_name == "Melendez2025":
        a1_path = os.path.join(ispec_dir, "input/abundances/D_PASTA/Melendez2025/A46_tablea1_raw.csv")
        a2_path = os.path.join(ispec_dir, "input/abundances/D_PASTA/Melendez2025/A46_table2_raw.csv")
        ref_xh, ref_xfe = build_ref_for_star_choice(
            star=ref_star,
            ref_name="Melendez2025",
            a46_a1_path=a1_path,
            a46_a2_path=a2_path,
        )
    else:
        raise RuntimeError(f"Unsupported REF_NAME for step 5: {ref_name}")

    ref_xh = add_tc_to(ref_xh, tc_df)
    ref_xfe = add_tc_to(ref_xfe, tc_df)

    print("[STEP 5] Generating all abundance comparison plots.")
    plot_xh_by_Tc_ordered(target_xh, target=target, outdir=abundance_plot_folder)
    plot_xh_vs_Tc_labels(target_xh, target=target, outdir=abundance_plot_folder)
    plot_xh_by_Tc_with_ref(target_xh, ref_xh, target=target, ref_name=ref_star, outdir=abundance_plot_folder)
    plot_xfe_by_Tc_with_ref(target_xfe, ref_xfe, target=target, ref_name=ref_star, outdir=abundance_plot_folder)

    if use_hfs_plots:
        plot_xh_by_Tc_with_ref_and_nohfs(target_xh, target_xh_nohfs, ref_xh, target=target, ref_name=ref_star, outdir=abundance_plot_folder)
        plot_xfe_by_Tc_with_ref_and_nohfs(target_xfe, target_xfe_nohfs, ref_xfe, target=target, ref_name=ref_star, outdir=abundance_plot_folder)
    else:
        print("[STEP 5] Skip fig06b / fig06c (external EWs and no HFS compare).")

    plot_xh_vs_Tc_with_ref_labels(target_xh, ref_xh, target=target, ref_name=ref_star, outdir=abundance_plot_folder)
    plot_xfe_vs_Tc_with_ref_labels(target_xfe, ref_xfe, target=target, ref_name=ref_star, outdir=abundance_plot_folder)

    print(f"[STEP 6] Step_element_abundance_plot completed. Output folder: {abundance_plot_folder}")
    return {
        "summary_path": summary_path,
        "plot_dir": abundance_plot_folder,
        "ref_star": ref_star,
        "ref_name": ref_name,
        "used_hfs_compare": bool(use_hfs_plots),
        "nohfs_summary_path": nohfs_summary_path,
    }


######## ===== For 6_combine_abundances ===== ########
def safe_float(s):
    try:
        return float(s)
    except Exception:
        return np.nan

def combine_ions_for_one_target(df):
    """
    输入：某 target 的 abundances_<target>.csv（长表，含电离态）
    返回：合并后的 DataFrame，列含：
      base_element, [X/H], std_[X/H], n_lines
    """
    d = df.copy()
    # 元素基名 & 电离态
    toks = d["element"].astype(str).str.strip()
    d["base_element"] = toks.str.split().str[0]
    # 规范数值列名
    if "[X/H]" not in d.columns:
        raise RuntimeError("input csv 缺少 [X/H] 列")
    # 容错：把 std 列与 n_lines 转为数值
    d["std_[X/H]"] = pd.to_numeric(d.get("std_[X/H]"), errors="coerce")
    d["n_lines"]    = pd.to_numeric(d.get("n_lines"), errors="coerce").fillna(0).astype(int)

    # 分组合并电离态（用你定义的 SEM 规则）
    g = d.groupby("base_element", as_index=False).apply(combine_ions_sem)
    g = g.reset_index().rename(columns={"level_0":"base_element"})
    # combine_ions_sem 给出的列名即 "[X/H]","std_[X/H]","n_lines","n_ions"
    g = g[["base_element","[X/H]","std_[X/H]","n_lines","n_ions"]]
    return g

def add_xfe_and_err(g_combined):
    """
    在合并后的表上，根据 Fe 的 [X/H] 与 std_[X/H] 推出 [X/Fe] 与 e_[X/Fe]
    """
    out = g_combined.copy()
    # 找 Fe 行
    if not (out["base_element"] == "Fe").any():
        out["[X/Fe]"] = np.nan
        out["e_[X/Fe]"] = np.nan
        return out

    fe_row = out.loc[out["base_element"] == "Fe"].iloc[0]
    feh = safe_float(fe_row["[X/H]"])
    e_feh = safe_float(fe_row["std_[X/H]"])
    out["[X/Fe]"]  = out["[X/H]"] - feh
    # 误差传播（若某个分量缺失则按已有项计算）
    out["e_[X/Fe]"] = np.sqrt(
        np.where(np.isfinite(out["std_[X/H]"]), out["std_[X/H]"]**2, 0.0) +
        (e_feh**2 if np.isfinite(e_feh) else 0.0)
    )
    return out

def build_master_abundance_table(ispec_dir, targets=None, out_path=None, tc_path=TC_PATH):
    """
    生成总表：行=元素；列=每个 target 的 [X/H],[X/H]_err,[X/Fe],[X/Fe]_err
    若 tc_path 提供，则把 Tc_K 并入并放到第 2 列。
    """
    root = os.path.join(ispec_dir, "output")
    if targets is None:
        candidates = []
        for sub in os.listdir(root):
            subdir = os.path.join(root, sub)
            if os.path.isdir(subdir) and os.path.isfile(os.path.join(subdir, f"abundances_{sub}.csv")):
                candidates.append(sub)
        targets = sorted(candidates)

    master = pd.DataFrame({"Element": []})

    for t in targets:
        csv_path = os.path.join(root, t, f"abundances_{t}.csv")
        if not os.path.isfile(csv_path):
            continue
        df = pd.read_csv(csv_path)
        for c in ["element", "[X/H]"]:
            if c not in df.columns:
                raise RuntimeError(f"{csv_path} 缺少列: {c}")

        comb = combine_ions_for_one_target(df)
        comb = add_xfe_and_err(comb)

        comb = comb.rename(columns={"base_element":"Element"})
        comb = comb[["Element", "[X/H]", "std_[X/H]", "[X/Fe]", "e_[X/Fe]"]]
        comb = comb.rename(columns={
            "[X/H]":      f"{t}_[X/H]",
            "std_[X/H]":  f"{t}_[X/H]_err",
            "[X/Fe]":     f"{t}_[X/Fe]",
            "e_[X/Fe]":   f"{t}_[X/Fe]_err",
        })
        master = master.merge(comb, on="Element", how="outer")

    # （可选）按原子序排序
    try:
        z_map = {k:int(v) for k,v in ATOMIC_Z.items()}
        master["_Z"] = master["Element"].map(z_map)
        master = master.sort_values(by=["_Z","Element"], kind="stable").drop(columns="_Z")
    except Exception:
        pass

    # ==== 新增：并入 Tc 并放到第 2 列 ====
    if tc_path:
        tc_df = load_tc(tc_path)
        master = master.merge(tc_df, on="Element", how="left")
        # 把 Tc_K 调到第二列
        cols = list(master.columns)
        cols.remove("Tc_K")
        cols = [cols[0], "Tc_K"] + cols[1:]
        master = master[cols]

    if out_path is None:
        out_path = os.path.join(root, "all_targets_abundances.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    master.to_csv(out_path, index=False)
    print(f"汇总完成：{out_path}")
    return master

# ========== For 3_atmos_params_iteration =========
# using q2
# 把 iSpec 的 linemasks 转成 q2 所需格式的 lines.csv
# 核心点：q2 的 lines.csv 需要：id, wave, species, EP, loggf, EW

def load_ispec_params_from_dump(ispec, dump_file):
    """
    从 iSpec 的 atmos_params_*.dump 里读出迭代后的参数（你用 ispec.save_results 保存的那个）。
    返回：teff, logg, feh(MH), vt(vmic)
    """
    restored = ispec.restore_results(dump_file)
    params = restored[0]
    return {
        "teff": float(params["teff"]),
        "logg": float(params["logg"]),
        "feh":  float(params["MH"]),
        "vt":   float(params["vmic"]),
    }

def _lm_to_q2_line_df(linemasks, star_id):
    """
    把 iSpec linemasks（包含 ew/wave_A/EP/loggf 等）转成“长表”：
    wavelength,species,ep,gf,ew
    注意：iSpec 的 element 是 'Fe 1'/'Fe 2'（不是 'Fe I'/'Fe II'）。
    """
    rows = []
    for lm in linemasks:
        elem = str(lm["element"]).strip()
        if elem not in ["Fe 1", "Fe 2", "Fe I", "Fe II"]:
            continue

        species = 26.0 if elem in ["Fe 1", "Fe I"] else 26.1
        rows.append({
            "wavelength": float(lm["wave_A"]),           # Å
            "species":    float(species),
            "ep":         float(lm["lower_state_eV"]),   # eV
            "gf":         float(lm["loggf"]),            # log(gf)
            "ew":         float(lm["ew"]),               # mÅ
            "id":         str(star_id),
        })

    df = pd.DataFrame(rows)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


def _merge_fe_ew_overrides(
    df_base: pd.DataFrame,
    ov: Optional[pd.DataFrame],
    wave_tol: float,
    label: str,
) -> pd.DataFrame:
    """
    用 Bedell Table1 导出的长表（wavelength, species, ep, gf, ew）按 (species, wave_key, ep, gf)
    对齐后覆盖 df_base 中的 ew（用于 q2 的 Sun/target 一致化）。
    """
    if ov is None or len(ov) == 0:
        return df_base
    df = df_base.copy()
    ov = ov.copy()
    df["_wk"] = (df["wavelength"].astype(float) / wave_tol).round().astype(np.int64)
    ov["_wk"] = (ov["wavelength"].astype(float) / wave_tol).round().astype(np.int64)
    df["_epk"] = np.round(df["ep"].astype(float), 4)
    df["_gfk"] = np.round(df["gf"].astype(float), 4)
    ov["_epk"] = np.round(ov["ep"].astype(float), 4)
    ov["_gfk"] = np.round(ov["gf"].astype(float), 4)
    keys = ["species", "_wk", "_epk", "_gfk"]
    missing = [k for k in keys if k not in ov.columns]
    if missing:
        raise ValueError(f"target/sun FE override DataFrame 缺少列: {missing}")
    sub = ov.drop_duplicates(subset=keys, keep="last")[keys + ["ew"]].rename(columns={"ew": "_ew_ov"})
    out = df.merge(sub, on=keys, how="left")
    n = int(out["_ew_ov"].notna().sum())
    if n:
        out["ew"] = out["_ew_ov"].where(out["_ew_ov"].notna(), out["ew"])
    out = out.drop(columns=["_ew_ov", "_wk", "_epk", "_gfk"], errors="ignore")
    print(f"[export_q2_inputs] FE EW overrides applied ({label}): {n} / {len(df_base)}")
    return out


def export_q2_inputs_wide(
    ispec,
    target_id,
    used_linemasks_fe,              # 你在 3_ 里最终用于 Fe 的 linemasks（包含 ew）
    solar_lines_csv,                # 你的 master solar EW 表（含 instrument/resolution/element/wave_A/EP/loggf/ew_mA）
    instrument, resolution,
    out_dir,
    sun_dump_file,                  # ★新增：对应 instrument 的 Sun 的 iSpec dump（用它读 Teff/logg/MH/vmic）
    sun_id="Sun",
    wave_tol=0.001,                 # Å：对齐同一条线用
    target_fe_ew_overrides=None,    # 可选：Bedell Table1 等 Fe 长表 columns wavelength,species,ep,gf,ew
    sun_fe_ew_overrides=None,
):
    os.makedirs(out_dir, exist_ok=True)

    # ---------- 1) target Fe lines（长表） ----------
    df_t = _lm_to_q2_line_df(used_linemasks_fe, target_id)
    print(f"[export_q2_inputs] target Fe lines: {len(df_t)}")
    df_t = _merge_fe_ew_overrides(df_t, target_fe_ew_overrides, wave_tol, "target")

    # ---------- 2) Sun Fe lines（从你的 master solar EW 表来） ----------
    sun_master = pd.read_csv(solar_lines_csv)

    # ---------- NEW: instrument 先筛出来 ----------
    sun_inst = sun_master[sun_master["instrument"].astype(str).str.strip() == str(instrument).strip()].copy()
    if len(sun_inst) == 0:
        raise RuntimeError(f"[export_q2_inputs] solar master has no rows for instrument={instrument}")

    # ---------- NEW: 自动选最接近的 resolution ----------
    # 注意：这里转 float 是为了兼容 int/float/字符串等情况
    avail = pd.to_numeric(sun_inst["resolution"], errors="coerce").values
    if not np.isfinite(avail).any():
        raise RuntimeError(f"[export_q2_inputs] solar master resolution column cannot be parsed for instrument={instrument}")

    req = float(resolution)
    res_use = avail[np.nanargmin(np.abs(avail - req))]

    # ---------- NEW: 用 res_use 过滤出太阳 Fe 线 ----------
    sun_sel = sun_inst[
        (pd.to_numeric(sun_inst["resolution"], errors="coerce") == float(res_use)) &
        (sun_inst["element"].isin(["Fe I", "Fe II", "Fe 1", "Fe 2"]))
    ].copy()

    print(f"[export_q2_inputs] solar resolution requested={resolution}, used={res_use}")

    # 兼容 Fe I/Fe 1 两种写法
    def _elem_to_species(x):
        x = str(x).strip()
        return 26.0 if x in ["Fe I", "Fe 1"] else 26.1

    df_s = pd.DataFrame({
        "wavelength": sun_sel["wave_A"].astype(float),
        "species":    sun_sel["element"].map(_elem_to_species).astype(float),
        "ep":         sun_sel["EP"].astype(float),
        "gf":         sun_sel["loggf"].astype(float),
        "ew":         sun_sel["ew_mA"].astype(float),
        "id":         sun_id,
    }).dropna()
    print(f"[export_q2_inputs] sun Fe lines   : {len(df_s)}")
    df_s = _merge_fe_ew_overrides(df_s, sun_fe_ew_overrides, wave_tol, "Sun")

    # ---------- 3) 只保留共同谱线（按 species + wavelength 对齐） ----------
    df_t["wave_key"] = (df_t["wavelength"]/wave_tol).round().astype(int)
    df_s["wave_key"] = (df_s["wavelength"]/wave_tol).round().astype(int)

    key_cols = ["species","wave_key","ep","gf"]
    common = pd.merge(
        df_t[key_cols].drop_duplicates(),
        df_s[key_cols].drop_duplicates(),
        on=key_cols, how="inner"
    )
    print(f"[export_q2_inputs] common Fe lines: {len(common)}")

    df_t2 = df_t.merge(common, on=key_cols, how="inner")
    df_s2 = df_s.merge(common, on=key_cols, how="inner")

    # ---------- 4) 生成 q2 需要的 lines 宽表 ----------
    base = common.copy()
    # 用 Sun 的 wavelength 作为输出 wavelength（更稳定）
    wave_map = df_s2.groupby(key_cols)["wavelength"].median().reset_index()
    base = base.merge(wave_map, on=key_cols, how="left")

    # 各自 EW
    sun_ew = df_s2.groupby(key_cols)["ew"].median().reset_index().rename(columns={"ew": sun_id})
    tar_ew = df_t2.groupby(key_cols)["ew"].median().reset_index().rename(columns={"ew": str(target_id)})

    lines_wide = base.merge(sun_ew, on=key_cols, how="inner").merge(tar_ew, on=key_cols, how="inner")

    # 输出列顺序必须含：wavelength,species,ep,gf，然后是各星的 EW 列
    lines_wide = lines_wide[["wavelength","species","ep","gf",sun_id,str(target_id)]].sort_values(
        by=["species","wavelength"]
    )

    lines_csv = os.path.join(out_dir, "lines_q2.csv")
    lines_wide.to_csv(lines_csv, index=False)
    print(f"[export_q2_inputs] lines.csv rows : {len(lines_wide)} -> {lines_csv}")

    # ---------- 5) stars.csv（Sun 参数从 dump 读，target 用初值） ----------
    sun_par = load_ispec_params_from_dump(ispec, sun_dump_file)

    # 这里 target 的初值：你可以继续用你现在的 initial_teff/logg/MH/vmic
    # 所以我把 target 初值作为参数传进来更灵活；你也可以在外面写好 DataFrame 再存
    # ——为了和你“精确到改哪段”一致，我建议 stars_df 在 3_ 里生成（下一节给你替换代码）

    stars_csv = os.path.join(out_dir, "stars_q2.csv")

    return lines_csv, stars_csv, sun_par

# ========= For 0_Sun_abundance ==========
def _build_sun_atmos_offset_row(
    measured,
    adopted,
    *,
    solar_label,
    instrument,
    resolution,
    atmos_method,
    feh_q2_column=np.nan,
    afe_sun_reference=np.nan,
    note="",
):
    m = {k: float(measured[k]) for k in ("teff", "logg", "MH", "vmic", "alpha")}
    a = {k: float(adopted[k]) for k in ("teff", "logg", "MH", "vmic", "alpha")}
    return {
        "solar_label": str(solar_label),
        "instrument": str(instrument),
        "resolution": int(resolution),
        "atmos_method": str(atmos_method),
        "teff_measured": m["teff"],
        "logg_measured": m["logg"],
        "MH_measured": m["MH"],
        "vmic_measured": m["vmic"],
        "alpha_measured": m["alpha"],
        "feh_q2_column": float(feh_q2_column) if np.isfinite(feh_q2_column) else np.nan,
        "AFe_q2_column": float(feh_q2_column) if np.isfinite(feh_q2_column) else np.nan,
        "AFe_sun_reference": float(afe_sun_reference) if np.isfinite(afe_sun_reference) else np.nan,
        "teff_adopted": a["teff"],
        "logg_adopted": a["logg"],
        "MH_adopted": a["MH"],
        "vmic_adopted": a["vmic"],
        "alpha_adopted": a["alpha"],
        "offset_teff": a["teff"] - m["teff"],
        "offset_logg": a["logg"] - m["logg"],
        "offset_MH": a["MH"] - m["MH"],
        "offset_vmic": a["vmic"] - m["vmic"],
        "offset_alpha": a["alpha"] - m["alpha"],
        "note": str(note),
    }


def _save_sun_atmos_offset_csv(row, per_target_csv, master_csv=None):
    pd.DataFrame([row]).to_csv(per_target_csv, index=False)
    print(f"[0_Sun][Part3] Saved offset table -> {per_target_csv}")
    if master_csv:
        os.makedirs(os.path.dirname(master_csv), exist_ok=True)
        if os.path.isfile(master_csv):
            master = pd.read_csv(master_csv)
            mask = (
                (master["solar_label"].astype(str) == row["solar_label"])
                & (master["instrument"].astype(str) == row["instrument"])
                & (pd.to_numeric(master["resolution"], errors="coerce") == row["resolution"])
            )
            master = master.loc[~mask]
            master = pd.concat([master, pd.DataFrame([row])], ignore_index=True)
        else:
            master = pd.DataFrame([row])
        master.to_csv(master_csv, index=False)
        print(f"[0_Sun][Part3] Updated master offset table -> {master_csv}")


def _resolve_solar_library_paths(*, ispec_dir: str, pipeline_options: dict[str, object]):
    if bool(pipeline_options.get("input_has_external_ews", False)):
        base = os.path.join(ispec_dir, "output", "withEW_Bedell")
        os.makedirs(base, exist_ok=True)
        solar_lines_csv = os.path.join(base, "solar_lines_instru_Bedell2018_Sun.csv")
    else:
        solar_lines_csv = os.path.join(ispec_dir, SOLAR_LINES_INFO_REL)
    solar_abund_csv = os.path.join(ispec_dir, SOLAR_ABUND_INFO_REL)
    return solar_lines_csv, solar_abund_csv


def _to_roman_ion(ele_label: str):
    s = str(ele_label).strip().replace("  ", " ")
    parts = s.split(" ")
    sym = parts[0] if len(parts) else s
    ion_raw = parts[1] if len(parts) > 1 else "1"
    ion_map = {"1": "I", "2": "II", "I": "I", "II": "II", "i": "I", "ii": "II"}
    ion_roman = ion_map.get(str(ion_raw), str(ion_raw))
    ion_num = 1 if ion_roman == "I" else 2 if ion_roman == "II" else ion_roman
    return f"{sym} {ion_roman}", f"{sym} {ion_num}"


def _run_sun_line_abundance_exports(
    *,
    ispec,
    linemasks,
    target: str,
    solar_label: str,
    instrument: str,
    from_resolution: float,
    output_folder: str,
    ispec_dir: str,
    teff: float,
    logg: float,
    mh: float,
    alpha: float,
    vmic: float,
    atmosphere_layers,
    solar_abundances,
    code: str,
    pipeline_options: dict[str, object],
):
    elem_norm = np.array([str(e).strip() for e in linemasks["element"]])
    elements = sorted(set(elem_norm.tolist()))
    solar_lines = []
    for ele in elements:
        sel = elem_norm == ele
        lm_ele = linemasks[sel]
        if len(lm_ele) == 0:
            continue
        spec_abund, _, x_over_h, _ = ispec.determine_abundances(
            atmosphere_layers,
            teff,
            logg,
            mh,
            alpha,
            lm_ele,
            solar_abundances,
            microturbulence_vel=vmic,
            verbose=0,
            code=code,
        )
        spec_abund = np.asarray(spec_abund, dtype=float)
        x_over_h = np.asarray(x_over_h, dtype=float)
        for lm, logeps_m12, xh in zip(lm_ele, spec_abund, x_over_h):
            if not (np.isfinite(logeps_m12) and np.isfinite(xh)):
                continue
            solar_lines.append(
                {
                    "instrument": instrument,
                    "resolution": int(from_resolution),
                    "solar_label": str(solar_label),
                    "element": ele,
                    "wave_A": float(lm["wave_A"]),
                    "EP": float(lm["lower_state_eV"]),
                    "loggf": float(lm["loggf"]),
                    "ew_mA": float(lm["ew"]),
                    "logeps_sun": float(logeps_m12 + 12.0),
                    "[X/H]_sun_Grevesse": float(xh),
                }
            )
    df_solar_lines = pd.DataFrame(solar_lines)
    print(f"[0_Sun][Part4] Solar per-line rows: {len(df_solar_lines)}")

    solar_lines_csv, solar_abund_csv = _resolve_solar_library_paths(
        ispec_dir=ispec_dir, pipeline_options=pipeline_options
    )
    if os.path.exists(solar_lines_csv):
        master = pd.read_csv(solar_lines_csv)
        master = master[
            ~(
                (master["instrument"].astype(str) == str(instrument))
                & (
                    pd.to_numeric(master["resolution"], errors="coerce")
                    == int(from_resolution)
                )
            )
        ]
        master = pd.concat([master, df_solar_lines], ignore_index=True)
    else:
        master = df_solar_lines
    master.to_csv(solar_lines_csv, index=False)
    print(f"[0_Sun][Part4] Saved solar_lines_instru -> {solar_lines_csv} rows={len(master)}")

    g = df_solar_lines.groupby("element")
    df_solar_abund = pd.DataFrame(
        {
            "instrument": instrument,
            "resolution": int(from_resolution),
            "solar_label": str(solar_label),
            "element": g.size().index,
            "n_lines": g.size().values,
            "logeps_sun_mean": g["logeps_sun"].mean().values,
            "[X/H]_sun_mean_Grevesse": g["[X/H]_sun_Grevesse"].mean().values,
            "std_[X/H]_sun": g["[X/H]_sun_Grevesse"].std(ddof=1).values,
        }
    )
    if os.path.exists(solar_abund_csv):
        master_a = pd.read_csv(solar_abund_csv)
        master_a = master_a[
            ~(
                (master_a["instrument"].astype(str) == str(instrument))
                & (
                    pd.to_numeric(master_a["resolution"], errors="coerce")
                    == int(from_resolution)
                )
            )
        ]
        master_a = pd.concat([master_a, df_solar_abund], ignore_index=True)
    else:
        master_a = df_solar_abund
    master_a.to_csv(solar_abund_csv, index=False)
    print(f"[0_Sun][Part4] Saved solar_abund_instru -> {solar_abund_csv} rows={len(master_a)}")

    q2_raw_csv = os.path.join(output_folder, f"abundances_linebyline_q2_{target}_raw.csv")
    q2_sum_csv = os.path.join(output_folder, f"abundances_linebyline_q2_{target}.csv")
    if len(df_solar_lines):
        q2_raw = df_solar_lines.copy()
        q2_raw["element_roman"] = q2_raw["element"].map(lambda x: _to_roman_ion(x)[0])
        q2_raw["element_num"] = q2_raw["element"].map(lambda x: _to_roman_ion(x)[1])
        q2_raw["id"] = target
        q2_raw["EP"] = pd.to_numeric(q2_raw.get("EP"), errors="coerce")
        q2_raw["loggf"] = pd.to_numeric(q2_raw.get("loggf"), errors="coerce")
        q2_raw["ew_mA"] = pd.to_numeric(q2_raw.get("ew_mA"), errors="coerce")
        q2_raw["wave_A"] = pd.to_numeric(q2_raw.get("wave_A"), errors="coerce")
        q2_raw["[X/H]_star_Grevesse"] = pd.to_numeric(
            q2_raw.get("[X/H]_sun_Grevesse"), errors="coerce"
        )
        q2_raw["[X/H]_sun_Grevesse"] = 0.0
        q2_raw["d[X/H]"] = q2_raw["[X/H]_star_Grevesse"] - q2_raw["[X/H]_sun_Grevesse"]
        q2_raw["logeps_star"] = pd.to_numeric(q2_raw.get("logeps_sun"), errors="coerce")
        q2_raw["logeps_sun"] = q2_raw["logeps_star"] - q2_raw["d[X/H]"]
        q2_raw["dlogeps"] = q2_raw["logeps_star"] - q2_raw["logeps_sun"]
        q2_raw_out = q2_raw[
            [
                "id",
                "element_roman",
                "wave_A",
                "EP",
                "loggf",
                "ew_mA",
                "logeps_star",
                "logeps_sun",
                "dlogeps",
                "[X/H]_star_Grevesse",
                "[X/H]_sun_Grevesse",
                "d[X/H]",
            ]
        ].rename(columns={"element_roman": "element"})
        q2_raw_out.to_csv(q2_raw_csv, index=False)

        q2_sum = (
            q2_raw.assign(_xh=pd.to_numeric(q2_raw["d[X/H]"], errors="coerce"))
            .groupby("element_num", as_index=False)
            .agg(
                n_lines=("wave_A", "size"),
                **{"[X/H]": ("_xh", "mean"), "std_[X/H]": ("_xh", "std")},
            )
            .rename(columns={"element_num": "element"})
        )
        q2_sum.to_csv(q2_sum_csv, index=False)
        print(f"[0_Sun][Part5] Saved q2 raw -> {q2_raw_csv} rows={len(q2_raw_out)}")
        print(f"[0_Sun][Part5] Saved q2 summary -> {q2_sum_csv} rows={len(q2_sum)}")
    else:
        pd.DataFrame(
            columns=[
                "id",
                "element",
                "wave_A",
                "EP",
                "loggf",
                "ew_mA",
                "logeps_star",
                "logeps_sun",
                "dlogeps",
                "[X/H]_star_Grevesse",
                "[X/H]_sun_Grevesse",
                "d[X/H]",
            ]
        ).to_csv(q2_raw_csv, index=False)
        pd.DataFrame(columns=["element", "n_lines", "[X/H]", "std_[X/H]"]).to_csv(
            q2_sum_csv, index=False
        )
        print("[0_Sun][Part5] No line abundance rows; wrote empty q2 outputs.")

    return {
        "solar_lines_csv": solar_lines_csv,
        "solar_abund_csv": solar_abund_csv,
        "q2_raw_csv": q2_raw_csv,
        "q2_summary_csv": q2_sum_csv,
    }


def run_sun_part3plus_pipeline(
    *,
    ispec,
    target: str,
    solar_label: str,
    instrument: str,
    from_resolution: float,
    atmos_iter_method: str,
    output_folder: str,
    output_atmos_params_dumpfile_path: str,
    sun_atmos_offset_csv: str,
    sun_atmos_offset_master_csv: str,
    solar_atmos_literature: dict[str, float],
    solar_afe_literature: float,
    pipeline_options: dict[str, object],
    ispec_dir: str,
    spectrum_norm_path: str,
    initial_teff: float,
    initial_logg: float,
    initial_mh: float,
    q2_solution: dict[str, float] | None = None,
    glb_solution: dict[str, object] | None = None,
    linemasks_fe=None,
):
    from atmos_lbl_pipeline import step_log
    from abund_lbl_pipeline import (
        apply_lineqa_for_abundance,
        load_star_and_atmosphere_context,
        plot_all_element_gaussian_fits,
        prepare_abundance_engine,
        run_hfs_blends_and_update_outputs,
        run_nonfe_line_review,
    )

    adopted = dict(solar_atmos_literature)
    atmos_iter_method = str(atmos_iter_method).strip().lower()
    step_log("3", f"Running Sun part3+ pipeline with atmos_iter_method={atmos_iter_method}.")

    if atmos_iter_method == "glb":
        if not glb_solution:
            raise ValueError("glb_solution is required when atmos_iter_method='glb'.")
        params = dict(glb_solution["params"])
        errors = dict(glb_solution["errors"])
        status = dict(glb_solution.get("status", {}))
        x_over_h = glb_solution.get("x_over_h")
        selected_x_over_h = glb_solution.get("selected_x_over_h")
        fitted_lines_params = glb_solution.get("fitted_lines_params")
        used_linemasks = glb_solution.get("used_linemasks")
        spec_abund_fe = glb_solution.get("spec_abund_fe")
        measured = {
            "teff": float(params["teff"]),
            "logg": float(params["logg"]),
            "MH": float(params["MH"]),
            "vmic": float(params["vmic"]),
            "alpha": float(params.get("alpha", 0.0)),
        }
        params_sun = dict(adopted)
        errors_sun = {
            "teff": float(errors.get("teff", 0.0) or 0.0),
            "logg": float(errors.get("logg", 0.0) or 0.0),
            "MH": float(errors.get("MH", 0.0) or 0.0),
            "alpha": float(errors.get("alpha", 0.0) or 0.0),
            "vmic": float(errors.get("vmic", 0.0) or 0.0),
        }
        status_sun = {
            "converged": True,
            "method": "literature_solar_anchor_method1",
            "measured_method": "ispec_glb_Fe_EW",
            **status,
        }
        ispec.save_results(
            output_atmos_params_dumpfile_path,
            (
                params_sun,
                errors_sun,
                status_sun,
                x_over_h,
                selected_x_over_h,
                fitted_lines_params,
                used_linemasks,
                spec_abund_fe,
            ),
        )
        row = _build_sun_atmos_offset_row(
            measured,
            adopted,
            solar_label=solar_label,
            instrument=instrument,
            resolution=from_resolution,
            atmos_method="glb",
            note="measured from ispec.model_spectrum_from_ew; adopted=literature solar",
        )
    elif atmos_iter_method == "lbl":
        if not q2_solution:
            raise ValueError("q2_solution is required when atmos_iter_method='lbl'.")
        afe_q2 = float(q2_solution["feh_q2"])
        mh_q2_dex = q2_single_star_feh_column_to_mh(afe_q2, solar_afe_literature)
        measured = {
            "teff": float(q2_solution["teff_q2"]),
            "logg": float(q2_solution["logg_q2"]),
            "MH": float(mh_q2_dex),
            "vmic": float(q2_solution["vt_q2"]),
            "alpha": 0.0,
        }
        params_sun = dict(adopted)
        errors_sun = {"teff": 0.0, "logg": 0.0, "MH": 0.0, "alpha": 0.0, "vmic": 0.0}
        status_sun = {
            "converged": True,
            "method": "literature_solar_anchor_method1",
            "measured_method": "q2_single_star_Fe",
            "q2_teff": float(q2_solution["teff_q2"]),
            "q2_logg": float(q2_solution["logg_q2"]),
            "q2_vmic": float(q2_solution["vt_q2"]),
            "q2_feh_column": float(q2_solution["feh_q2"]),
        }
        if linemasks_fe is None:
            lm_path = os.path.join(
                output_folder, "linemasks", f"{target}_melendez2014_star_fitted_linemasks.txt"
            )
            _lm = ispec.read_line_regions(lm_path)
            _elem = np.array([str(e).strip() for e in _lm["element"]])
            linemasks_fe = _lm[(_elem == "Fe 1") | (_elem == "Fe 2")]

        ispec.save_results(
            output_atmos_params_dumpfile_path,
            (params_sun, errors_sun, status_sun, None, None, None, linemasks_fe, None),
        )
        row = _build_sun_atmos_offset_row(
            measured,
            adopted,
            solar_label=solar_label,
            instrument=instrument,
            resolution=from_resolution,
            atmos_method="lbl",
            feh_q2_column=float(q2_solution["feh_q2"]),
            afe_sun_reference=float(solar_afe_literature),
            note="MH_measured converted from q2 A(Fe) with literature A(Fe)_sun",
        )
    else:
        raise ValueError(f"Unknown atmos_iter_method: {atmos_iter_method!r}")

    _save_sun_atmos_offset_csv(row, sun_atmos_offset_csv, sun_atmos_offset_master_csv)
    step_log("3", f"Saved Sun atmospheric dump and offset table: {output_atmos_params_dumpfile_path}")

    run_find_linemasks = bool(pipeline_options.get("run_find_linemasks", True))
    use_external_ew = bool(pipeline_options.get("input_has_external_ews", False))
    run_part1_spectrum = run_find_linemasks and (not use_external_ew)
    plot_all_elements = bool(pipeline_options.get("sun_plot_all_elements", False))

    ctx4 = load_star_and_atmosphere_context(
        ispec=ispec,
        target=target,
        output_folder=output_folder,
        spectrum_norm_path=spectrum_norm_path,
        output_atmos_params_dumpfile_path=output_atmos_params_dumpfile_path,
        initial_teff=initial_teff,
        initial_logg=initial_logg,
        initial_MH=initial_mh,
    )
    plot_all_element_gaussian_fits(
        star_spectrum=ctx4["star_spectrum"],
        linemasks=ctx4["linemasks"],
        output_folder=output_folder,
        run_part1_spectrum=run_part1_spectrum,
        plot_all_elements=plot_all_elements,
        progress_every=10,
    )
    run_nonfe_line_review(
        output_folder=output_folder,
        target=target,
        run_part1_spectrum=run_part1_spectrum,
        plot_all_elements=plot_all_elements,
        pipeline_options=pipeline_options,
    )
    linemasks_used = apply_lineqa_for_abundance(
        ispec=ispec,
        linemasks=ctx4["linemasks"],
        output_folder=output_folder,
        linemask_output_folder=ctx4["linemask_output_folder"],
        target=target,
        pipeline_options=pipeline_options,
    )
    linemasks_used = linemasks_used[linemasks_used["wave_nm"] > 0]
    linemasks_used = linemasks_used[linemasks_used["ew"] > 0]

    engine = prepare_abundance_engine(
        ispec=ispec,
        ispec_dir=ispec_dir,
        teff=ctx4["teff"],
        logg=ctx4["logg"],
        mh=ctx4["mh"],
        alpha=ctx4["alpha"],
        code="moog",
    )
    export_out = _run_sun_line_abundance_exports(
        ispec=ispec,
        linemasks=linemasks_used,
        target=target,
        solar_label=solar_label,
        instrument=instrument,
        from_resolution=from_resolution,
        output_folder=output_folder,
        ispec_dir=ispec_dir,
        teff=ctx4["teff"],
        logg=ctx4["logg"],
        mh=ctx4["mh"],
        alpha=ctx4["alpha"],
        vmic=ctx4["vmic"],
        atmosphere_layers=engine["atmosphere_layers"],
        solar_abundances=engine["solar_abundances"],
        code=engine["code"],
        pipeline_options=pipeline_options,
    )

    hfs_res = run_hfs_blends_and_update_outputs(
        ispec=ispec,
        target=target,
        output_folder=output_folder,
        ispec_dir=ispec_dir,
        atmosphere_layers=engine["atmosphere_layers"],
        teff=ctx4["teff"],
        logg=ctx4["logg"],
        mh=ctx4["mh"],
        output_abundances_result_path_linebyline_q2=export_out["q2_summary_csv"],
        pipeline_options=pipeline_options,
        run_hfs_blends=bool(pipeline_options.get("sun_run_hfs_blends", True)),
        dry_run=bool(pipeline_options.get("sun_hfs_dry_run", False)),
    )
    step_log("6", "Sun part3+ pipeline completed.")
    return {"export": export_out, "hfs": hfs_res, "dump_file": output_atmos_params_dumpfile_path}


def run_sun_atmos_iteration(
    *,
    ispec,
    linemasks,
    atmos_iter_method: str,
    output_folder: str,
    solar_label: str,
    initial_teff: float,
    initial_logg: float,
    initial_mh: float,
    code: str = "moog",
    model: str | None = None,
    solar_abundances_file: str | None = None,
):
    """
    Execute Sun atmospheric-parameter iteration (glb/lbl) and return structured
    outputs for downstream run_sun_part3plus_pipeline().
    """
    from atmos_lbl_pipeline import step_log

    atmos_iter_method = str(atmos_iter_method).strip().lower()
    if model is None:
        model = os.path.join(resolve_ispec_dir(), "input", "atmospheres", "MARCS.GES") + os.sep
    if solar_abundances_file is None:
        if "ATLAS" in model:
            solar_abundances_file = os.path.join(
                resolve_ispec_dir(), "input", "abundances", "Grevesse.1998", "stdatom.dat"
            )
        else:
            solar_abundances_file = os.path.join(
                resolve_ispec_dir(), "input", "abundances", "Grevesse.2007", "stdatom.dat"
            )

    if atmos_iter_method == "glb":
        step_log("2", "Running Sun global atmospheric fit (ispec.model_spectrum_from_ew).")
        modeled_layers_pack = ispec.load_modeled_layers_pack(model)
        solar_abundances = ispec.read_solar_abundances(solar_abundances_file)
        initial_alpha = 0.0
        initial_vmic = ispec.estimate_vmic(initial_teff, initial_logg, initial_mh)
        if not ispec.valid_atmosphere_target(
            modeled_layers_pack,
            {"teff": initial_teff, "logg": initial_logg, "MH": initial_mh, "alpha": initial_alpha},
        ):
            step_log("2", "Warning: initial parameters are outside model grid.")
        results = ispec.model_spectrum_from_ew(
            linemasks,
            modeled_layers_pack,
            solar_abundances,
            initial_teff,
            initial_logg,
            initial_mh,
            initial_alpha,
            initial_vmic,
            free_params=["teff", "logg", "vmic"],
            adjust_model_metalicity=True,
            max_iterations=15,
            enhance_abundances=True,
            outliers_detection="robust",
            outliers_weight_limit=0.90,
            tmp_dir=None,
            code=code,
        )
        params, errors, status, x_over_h, selected_x_over_h, fitted_lines_params, used_linemasks = results
        step_log(
            "2",
            f"GLB solution: Teff={float(params['teff']):.2f}, logg={float(params['logg']):.3f}, "
            f"[M/H]={float(params['MH']):.3f}, vmic={float(params['vmic']):.3f}",
        )
        return {
            "q2_solution": None,
            "linemasks_fe": None,
            "glb_solution": {
                "params": params,
                "errors": errors,
                "status": status,
                "x_over_h": x_over_h,
                "selected_x_over_h": selected_x_over_h,
                "fitted_lines_params": fitted_lines_params,
                "used_linemasks": used_linemasks,
                "spec_abund_fe": np.asarray(x_over_h, dtype=float),
            },
        }

    if atmos_iter_method == "lbl":
        step_log("2", "Running Sun LBL atmospheric fit (q2 specpars).")
        import q2

        elem_col = np.array([str(e).strip() for e in linemasks["element"]])
        linemasks_fe = linemasks[(elem_col == "Fe 1") | (elem_col == "Fe 2")]
        q2_dir = os.path.join(output_folder, "q2_work")
        os.makedirs(q2_dir, exist_ok=True)
        initial_vmic = ispec.estimate_vmic(initial_teff, initial_logg, initial_mh)
        stars_csv, lines_csv = export_q2_inputs_single_wide(
            linemasks_fe,
            star_id=solar_label,
            teff=initial_teff,
            logg=initial_logg,
            feh=initial_mh,
            vt=float(initial_vmic),
            out_dir=q2_dir,
            fe_only=True,
        )

        data = q2.Data(stars_csv, lines_csv)
        sp = q2.specpars.SolvePars(grid="marcs")
        sp.errors = True
        solution_csv = os.path.join(q2_dir, "solution.csv")
        q2.specpars.solve_all(data, sp, solution_csv)
        sol = pd.read_csv(solution_csv)
        row = sol.loc[sol["id"] == str(solar_label)].iloc[0]
        teff_q2, logg_q2, feh_q2, vt_q2 = map(
            float, [row["teff"], row["logg"], row["feh"], row["vt"]]
        )
        step_log(
            "2",
            f"LBL solution: Teff={teff_q2:.2f}, logg={logg_q2:.3f}, feh_col={feh_q2:.3f}, vt={vt_q2:.3f}",
        )
        return {
            "linemasks_fe": linemasks_fe,
            "glb_solution": None,
            "q2_solution": {
                "teff_q2": teff_q2,
                "logg_q2": logg_q2,
                "feh_q2": feh_q2,
                "vt_q2": vt_q2,
            },
        }

    raise ValueError(f"Unknown atmos_iter_method: {atmos_iter_method!r}")


def _to_bool(x):
    """Robust bool conversion for 'discarded' or similar columns."""
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if x is None:
        return False
    s = str(x).strip().lower()
    return s in ("1", "true", "t", "yes", "y")

def export_q2_inputs_single_wide(
    linemasks,
    star_id,
    teff, logg, feh, vt,
    out_dir,
    fe_only=True,
    wave_tol=0.001,         # Å; used to group same line
    discard_col="discarded", # if exists in linemasks
):
    """
    Export q2 inputs for a SINGLE star (Sun-only use case).
    Writes:
      - stars_q2.csv : columns [id, teff, logg, feh, vt]
      - lines_q2.csv : wide format: [wavelength, species, ep, gf, <star_id>]
                       where <star_id> column stores EW (mA).

    Parameters
    ----------
    linemasks : numpy structured array / table
        iSpec linemasks that contains at least:
        element, wave_A, lower_state_eV, loggf, ew
    star_id : str
        The star label used as column name in lines_q2.csv
    teff, logg, feh, vt : float
        Initial parameters for q2
    out_dir : str
        Directory to write q2 inputs
    fe_only : bool
        If True, keep only Fe I/Fe II lines (recommended for specpars)
    wave_tol : float
        Wavelength tolerance (Å) for grouping duplicates
    discard_col : str
        Column name for discarded flag, if present

    Returns
    -------
    (stars_csv_path, lines_csv_path)
    """
    os.makedirs(out_dir, exist_ok=True)

    # ---------- 1) build long table from linemasks ----------
    rows = []
    names = getattr(linemasks, "dtype", None).names if hasattr(linemasks, "dtype") else None
    if names is None:
        raise TypeError("linemasks must be a numpy structured array with dtype.names")

    required = ["element", "wave_A", "lower_state_eV", "loggf", "ew"]
    missing = [c for c in required if c not in names]
    if missing:
        raise ValueError(f"linemasks missing required columns: {missing}")

    has_discard = (discard_col in names)

    for lm in linemasks:
        if has_discard and _to_bool(lm[discard_col]):
            continue

        elem = str(lm["element"]).strip()

        # keep only Fe lines for atmospheric parameter solving
        if fe_only:
            if elem not in ("Fe 1", "Fe 2", "Fe I", "Fe II"):
                continue
            species = 26.0 if elem in ("Fe 1", "Fe I") else 26.1
        else:
            # For non-Fe-only usage, you must provide/compute species properly.
            # Here we keep it strict to avoid silent wrong species.
            raise ValueError("fe_only=False is not supported in this helper (species would be ambiguous).")

        waveA = pd.to_numeric(lm["wave_A"], errors="coerce")
        ep    = pd.to_numeric(lm["lower_state_eV"], errors="coerce")
        gf    = pd.to_numeric(lm["loggf"], errors="coerce")
        ew    = pd.to_numeric(lm["ew"], errors="coerce")  # iSpec ew is typically in mÅ

        if not (np.isfinite(waveA) and np.isfinite(ep) and np.isfinite(gf) and np.isfinite(ew)):
            continue
        if ew <= 0:
            continue

        rows.append({
            "wavelength": float(waveA),
            "species": float(species),
            "ep": float(ep),
            "gf": float(gf),
            "ew": float(ew),
        })

    df = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).dropna()
    if df.empty:
        raise RuntimeError("No valid Fe lines found in linemasks for q2 export.")

    # ---------- 2) group duplicates (same species + close wavelength + same ep/gf) ----------
    df["wave_key"] = (df["wavelength"] / wave_tol).round().astype(int)
    key_cols = ["species", "wave_key", "ep", "gf"]

    g = df.groupby(key_cols, as_index=False).agg(
        wavelength=("wavelength", "median"),
        ew=("ew", "median"),
    )

    # ---------- 3) make wide lines table ----------
    lines_wide = g[["wavelength", "species", "ep", "gf"]].copy()
    lines_wide[str(star_id)] = g["ew"].astype(float)
    lines_wide = lines_wide.sort_values(by=["species", "wavelength"], kind="stable")

    lines_csv = os.path.join(out_dir, "lines_q2.csv")
    lines_wide.to_csv(lines_csv, index=False)

    # ---------- 4) stars.csv ----------
    stars_df = pd.DataFrame([{
        "id": str(star_id),
        "teff": float(teff),
        "logg": float(logg),
        "feh":  float(feh),
        "vt":   float(vt),
    }])
    stars_csv = os.path.join(out_dir, "stars_q2.csv")
    stars_df.to_csv(stars_csv, index=False)

    print(f"[export_q2_inputs_single_wide] lines: {len(lines_wide)} -> {lines_csv}")
    print(f"[export_q2_inputs_single_wide] stars: {len(stars_df)} -> {stars_csv}")
    return stars_csv, lines_csv

def _pick_col(df, candidates):
    """从 df 里按候选列名挑一个存在的"""
    for c in candidates:
        if c in df.columns:
            return c
    return None

def save_solar_afe_reference_from_q2_solution(
    solution_csv,
    solar_id,
    instrument,
    resolution,
    out_json,
    extra_meta=None,
):
    """
    从 q2 的 solution.csv 里提取太阳的最终解，并保存为 JSON。
    兼容：q2 在单星（无 reference）时经常把 A(Fe) 写在 'feh' 这一列里。
    """
    sol = pd.read_csv(solution_csv)

    # id 列名兼容
    id_col = _pick_col(sol, ["id", "star", "name", "Star", "Name"])
    if id_col is None:
        raise ValueError(f"Cannot find star id column in {solution_csv}")

    sub = sol.loc[sol[id_col].astype(str) == str(solar_id)]
    if sub.empty:
        raise ValueError(f"Cannot find solar_id={solar_id} in {solution_csv}")

    # 如果一个星有多行（多次迭代输出），取最后一行
    row = sub.iloc[-1]

    teff_col = _pick_col(sol, ["teff", "Teff"])
    logg_col = _pick_col(sol, ["logg", "Logg"])
    vt_col   = _pick_col(sol, ["vt", "vmic", "Vt", "Vmic"])
    feh_col  = _pick_col(sol, ["feh", "[Fe/H]", "FeH", "AFe", "afe", "A(Fe)"])

    if feh_col is None:
        raise ValueError("Cannot find feh/AFe column in q2 solution.csv")

    rec = {
        "solar_id": str(solar_id),
        "instrument": str(instrument),
        "resolution": int(resolution),
        # 注意：这里保存的是“太阳跑出来的 A(Fe)”（单星模式下通常落在 feh 列）
        "AFe_sun": float(row[feh_col]),
        "teff": float(row[teff_col]) if teff_col else np.nan,
        "logg": float(row[logg_col]) if logg_col else np.nan,
        "vt":   float(row[vt_col])   if vt_col else np.nan,
    }

    if extra_meta:
        rec["meta"] = dict(extra_meta)

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2, ensure_ascii=False)

    print(f"[save_solar_afe_reference] Saved -> {out_json}")
    return rec


def load_solar_afe_reference(out_json):
    """读取 save_solar_afe_reference_from_q2_solution 写出的 JSON"""
    with open(out_json, "r", encoding="utf-8") as f:
        rec = json.load(f)
    return rec


def afe_to_feh(AFe_star, AFe_sun):
    """把绝对丰度 A(Fe) 转成 dex [Fe/H]： [Fe/H] = A(Fe)_* - A(Fe)_sun。"""
    return float(AFe_star) - float(AFe_sun)


# log10(N_Fe/N_H)+12 at [Fe/H]=0（与 iSpec/MARCS 常用太阳丰度标尺对应）
SOLAR_AFE_AT_FEH_ZERO = {
    "Grevesse1998": 7.50,
    "Grevesse2007": 7.45,
    "Asplund2009": 7.50,
}


def q2_single_star_feh_column_to_mh(feh_column, afe_sun_reference):
    """
    q2 单星解（无 reference star）时 solution.csv 的 ``feh`` 列为 A(Fe)，
    不是 dex [Fe/H]。换算为 [Fe/H] 后与 iSpec/MOOG 的 MH 量纲一致。
    """
    return afe_to_feh(feh_column, afe_sun_reference)


def load_solar_afe_reference(in_json, instrument=None, resolution=None):
    """
    读取 0_Sun 保存的 Sun_AFe_reference.json，并可按 instrument/resolution 做一致性检查。
    返回 dict，其中至少包含 AFe_sun。
    """
    with open(in_json, "r", encoding="utf-8") as f:
        rec = json.load(f)

    if instrument is not None and "instrument" in rec:
        if str(rec["instrument"]) != str(instrument):
            raise ValueError(f"Solar ref instrument mismatch: {rec['instrument']} vs {instrument}")

    if resolution is not None and "resolution" in rec:
        if int(rec["resolution"]) != int(resolution):
            raise ValueError(f"Solar ref resolution mismatch: {rec['resolution']} vs {resolution}")

    if "AFe_sun" not in rec:
        raise KeyError(f"'AFe_sun' not found in {in_json}")

    return rec


def read_feh_from_abundance_with_gaia(
    star: str,
    csv_path: str,
    source_hint: str = "Bedell_2018",
):
    """
    从 long-format 表 abundance_with_gaia.csv 里读取某颗星的 [Fe/H] 及不确定度。
    期望列名：star_name, element_code, abundance, uncertainty, source, element_str

    返回：
        feh (float), e_feh (float)
    """
    df = pd.read_csv(csv_path)

    # 1) 选星
    star_key = str(star).strip()
    df = df[df["star_name"].astype(str).str.strip() == star_key]
    if df.empty:
        raise RuntimeError(f"[read_feh] 在 {csv_path} 找不到 star='{star_key}'")

    # 2) 优先选 Bedell 来源（如果同一星有多来源）
    if source_hint and "source" in df.columns:
        df_src = df[df["source"].astype(str).str.contains(source_hint, na=False)]
        if not df_src.empty:
            df = df_src

    # 3) 找 Fe：element_str == 'Fe' 或 element_code == 26
    fe = df[
        (df["element_str"].astype(str).str.strip().str.upper() == "FE")
        | (pd.to_numeric(df["element_code"], errors="coerce") == 26.0)
    ]
    if fe.empty:
        raise RuntimeError(f"[read_feh] star='{star_key}' 在 {csv_path} 里没找到 Fe 行（source_hint='{source_hint}'）")

    # 4) 若有多行：取第一行；你也可以改成取均值
    r = fe.iloc[0]
    feh = float(pd.to_numeric(r["abundance"], errors="coerce"))
    e_feh = float(pd.to_numeric(r["uncertainty"], errors="coerce")) if "uncertainty" in fe.columns else np.nan
    return feh, e_feh