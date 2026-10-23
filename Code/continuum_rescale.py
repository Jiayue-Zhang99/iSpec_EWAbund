"""
Continuum-window QA + optional global re-scaling for already-normalized spectra.

Intended use
------------
Apply **after** the final continuum normalization is done (often a manual / GUI
continuum fit). Call this when *reading* ``spectrum_norm_path`` for EW /
atmosphere / abundance steps — **not** immediately after automatic
``template`` / ``continuum_regions`` normalization in Step1 data processing.

Use continuum region masks (e.g. fe_lines_continuum.txt) to estimate the flux
histogram peak; if it is offset from 1.0 beyond tolerance, multiply flux/err by
``target / peak`` so continuum windows sit near 1.0.

Controlled by ``PIPELINE_OPTIONS`` keys prefixed with ``continuum_rescale_``.
"""

from __future__ import annotations

import glob
import logging
import os
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_CONTINUUM_REGION_FILES = [
    "input/regions/fe_lines_continuum.txt",
]


def _as_list(x) -> list:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def resolve_continuum_region_paths(
    ispec_dir: str,
    pipeline_options: dict[str, Any] | None = None,
) -> list[str]:
    """Resolve continuum-window files from PIPELINE_OPTIONS (relative to ispec_dir)."""
    opts = pipeline_options or {}
    files = _as_list(opts.get("continuum_region_files"))
    if not files:
        files = list(DEFAULT_CONTINUUM_REGION_FILES)

    paths: list[str] = []
    for rel in files:
        p = rel if os.path.isabs(rel) else os.path.join(ispec_dir, rel)
        if os.path.isfile(p):
            paths.append(p)
        else:
            logger.warning("[continuum_rescale] missing continuum file: %s", p)

    extra_glob = opts.get("continuum_region_glob")
    if extra_glob:
        pattern = (
            extra_glob
            if os.path.isabs(extra_glob)
            else os.path.join(ispec_dir, extra_glob)
        )
        for p in sorted(glob.glob(pattern, recursive=True)):
            if os.path.isfile(p) and p not in paths:
                # Skip non-window files that happen to match "*continuum*"
                base = os.path.basename(p).lower()
                if "segment" in base:
                    continue
                paths.append(p)

    return paths


def load_continuum_regions(ispec, paths: list[str]):
    """Load and concatenate continuum region tables from one or more files."""
    chunks = []
    used = []
    for p in paths:
        try:
            reg = ispec.read_continuum_regions(p)
        except Exception as ex:
            logger.warning("[continuum_rescale] failed to read %s: %s", p, ex)
            continue
        if reg is None or len(reg) == 0:
            continue
        chunks.append(reg)
        used.append(p)
    if not chunks:
        raise FileNotFoundError(
            "No usable continuum region files. "
            "Check PIPELINE_OPTIONS['continuum_region_files']."
        )
    # numpy recarray concatenate
    return np.concatenate(chunks), used


def fluxes_in_continuum_windows(spectrum, continuum_regions) -> np.ndarray:
    """Return flux samples that fall inside any continuum window."""
    wave = np.asarray(spectrum["waveobs"], dtype=float)
    flux = np.asarray(spectrum["flux"], dtype=float)
    mask = np.zeros(len(wave), dtype=bool)
    for region in continuum_regions:
        mask |= (wave >= float(region["wave_base"])) & (wave <= float(region["wave_top"]))
    vals = flux[mask]
    vals = vals[np.isfinite(vals)]
    return vals


def estimate_continuum_peak(
    fluxes: np.ndarray,
    *,
    method: str = "hist_peak",
    hist_bins: int = 80,
    flux_range: tuple[float, float] = (0.85, 1.15),
    percentile: float = 95.0,
) -> dict[str, Any]:
    """
    Estimate continuum level from window fluxes.

    Methods
    -------
    hist_peak :
        Mode of histogram within ``flux_range`` (robust for slightly line-contaminated windows).
    median :
        Median within ``flux_range``.
    percentile :
        Given percentile within ``flux_range`` (e.g. 95 for upper envelope).
    """
    method = str(method).strip().lower()
    lo, hi = float(flux_range[0]), float(flux_range[1])
    clipped = fluxes[(fluxes >= lo) & (fluxes <= hi)]
    if clipped.size < 20:
        # Fall back to all finite fluxes if range is too strict
        clipped = fluxes
    if clipped.size == 0:
        raise ValueError("No finite continuum-window fluxes to estimate peak.")

    stats = {
        "n_all": int(fluxes.size),
        "n_used": int(clipped.size),
        "median": float(np.median(clipped)),
        "mean": float(np.mean(clipped)),
        "p16": float(np.percentile(clipped, 16)),
        "p84": float(np.percentile(clipped, 84)),
        "method": method,
        "flux_range": (lo, hi),
    }

    if method in ("hist_peak", "hist", "mode"):
        n_bins = max(10, int(hist_bins))
        counts, edges = np.histogram(clipped, bins=n_bins, range=(lo, hi) if clipped.size == fluxes.size or (clipped.min() >= lo and clipped.max() <= hi) else None)
        if counts.size == 0 or counts.max() <= 0:
            peak = float(np.median(clipped))
        else:
            i = int(np.argmax(counts))
            peak = float(0.5 * (edges[i] + edges[i + 1]))
        stats["hist_counts_max"] = int(counts.max()) if counts.size else 0
    elif method in ("median",):
        peak = float(np.median(clipped))
    elif method in ("percentile", "p", "pct"):
        peak = float(np.percentile(clipped, float(percentile)))
        stats["percentile"] = float(percentile)
    else:
        raise ValueError(
            f"Unknown continuum_rescale_method={method!r}; "
            "use hist_peak | median | percentile"
        )

    stats["peak"] = float(peak)
    return stats


def diagnose_continuum_level(
    spectrum,
    *,
    ispec,
    ispec_dir: str,
    pipeline_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Quantitative continuum QA using continuum-window flux histogram peak."""
    opts = pipeline_options or {}
    paths = resolve_continuum_region_paths(ispec_dir, opts)
    regions, used_paths = load_continuum_regions(ispec, paths)
    fluxes = fluxes_in_continuum_windows(spectrum, regions)

    method = opts.get("continuum_rescale_method", "hist_peak")
    hist_bins = int(opts.get("continuum_rescale_hist_bins", 80))
    fr = opts.get("continuum_rescale_flux_range", [0.85, 1.15])
    if not (isinstance(fr, (list, tuple)) and len(fr) == 2):
        fr = [0.85, 1.15]
    percentile = float(opts.get("continuum_rescale_percentile", 95.0))

    peak_info = estimate_continuum_peak(
        fluxes,
        method=method,
        hist_bins=hist_bins,
        flux_range=(float(fr[0]), float(fr[1])),
        percentile=percentile,
    )
    target = float(opts.get("continuum_rescale_target", 1.0))
    tol = float(opts.get("continuum_rescale_tol", 0.01))
    peak = float(peak_info["peak"])
    offset = peak - target
    needs_rescale = abs(offset) > tol and peak > 0

    report = {
        **peak_info,
        "target": target,
        "tol": tol,
        "offset": float(offset),
        "needs_rescale": bool(needs_rescale),
        "scale": float(target / peak) if peak > 0 else 1.0,
        "n_regions": int(len(regions)),
        "continuum_files": used_paths,
        "wave_min": float(np.min(spectrum["waveobs"])),
        "wave_max": float(np.max(spectrum["waveobs"])),
    }
    return report


def apply_continuum_scale(spectrum, scale: float):
    """Return a copy of spectrum with flux (and err if present) multiplied by scale."""
    scale = float(scale)
    out = spectrum.copy()
    out["flux"] = np.asarray(out["flux"], dtype=float) * scale
    if "err" in out.dtype.names:
        out["err"] = np.asarray(out["err"], dtype=float) * scale
    return out


def maybe_rescale_normalized_spectrum(
    spectrum,
    *,
    ispec,
    ispec_dir: str,
    pipeline_options: dict[str, Any] | None = None,
    spectrum_path: str | None = None,
    save: bool | None = None,
    verbose: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """
    Diagnose continuum windows and optionally re-scale the spectrum to target=1.

    Returns
    -------
    spectrum, report
    """
    opts = dict(pipeline_options or {})
    enabled = bool(opts.get("continuum_rescale_enabled", False))
    report: dict[str, Any] = {
        "enabled": enabled,
        "applied": False,
        "saved_path": None,
        "input_path": spectrum_path,
    }

    if not enabled:
        if verbose:
            print("[continuum_rescale] disabled (PIPELINE_OPTIONS['continuum_rescale_enabled']=False)")
        return spectrum, report

    diag = diagnose_continuum_level(
        spectrum,
        ispec=ispec,
        ispec_dir=ispec_dir,
        pipeline_options=opts,
    )
    report.update(diag)

    if verbose:
        print(
            "[continuum_rescale] "
            f"peak={diag['peak']:.5f} target={diag['target']:.3f} "
            f"offset={diag['offset']:+.5f} tol={diag['tol']:.3f} "
            f"n_win_flux={diag['n_used']} files={len(diag['continuum_files'])}"
        )

    apply = bool(opts.get("continuum_rescale_apply", True))
    if not diag["needs_rescale"]:
        if verbose:
            print("[continuum_rescale] within tolerance; no re-scaling.")
        return spectrum, report

    if not apply:
        if verbose:
            print(
                f"[continuum_rescale] would scale by {diag['scale']:.6f}, "
                "but continuum_rescale_apply=False"
            )
        return spectrum, report

    scaled = apply_continuum_scale(spectrum, diag["scale"])
    report["applied"] = True

    # Re-diagnose after scaling (sanity)
    post = diagnose_continuum_level(
        scaled,
        ispec=ispec,
        ispec_dir=ispec_dir,
        pipeline_options=opts,
    )
    report["peak_after"] = post["peak"]
    report["offset_after"] = post["offset"]

    if verbose:
        print(
            f"[continuum_rescale] applied scale={diag['scale']:.6f}; "
            f"peak {diag['peak']:.5f} -> {post['peak']:.5f}"
        )

    do_save = bool(opts.get("continuum_rescale_save", True)) if save is None else bool(save)
    write_back = bool(opts.get("continuum_rescale_write_back", False))
    if do_save and spectrum_path:
        if write_back:
            out_path = spectrum_path
        else:
            root, ext = os.path.splitext(spectrum_path)
            suffix = str(opts.get("continuum_rescale_save_suffix", "_resc"))
            out_path = f"{root}{suffix}{ext or '.fits'}"
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        ispec.write_spectrum(scaled, out_path)
        report["saved_path"] = out_path
        if verbose:
            print(f"[continuum_rescale] saved -> {out_path}")

    return scaled, report


def read_normalized_spectrum_with_rescale(
    ispec,
    spectrum_path: str,
    *,
    ispec_dir: str,
    pipeline_options: dict[str, Any] | None = None,
    verbose: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """``ispec.read_spectrum`` + optional continuum-window re-scaling."""
    spectrum = ispec.read_spectrum(spectrum_path)
    return maybe_rescale_normalized_spectrum(
        spectrum,
        ispec=ispec,
        ispec_dir=ispec_dir,
        pipeline_options=pipeline_options,
        spectrum_path=spectrum_path,
        verbose=verbose,
    )
