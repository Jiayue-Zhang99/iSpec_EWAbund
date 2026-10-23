"""One human-approved local continuum/EW rescue; never change the global spectrum.

Wavelengths are nm, EW and EW errors are mAngstrom. No abundance/reference
information enters the fit. Accepted fits are replayed after Sun's second find.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from line_review_ui import (
    DEFAULT_REASON_OPTIONS, FAILED_BOTH, LOCAL_RENORMALIZE, ReviewCandidate,
    ORIGIN_EW_OK, NEW_EW_OK, EW_CHOICES,
    _apply_existing_decisions, _build_new_deleted_rows, _load_linelist_reference,
    _refresh_deleted_summary, _sync_images_for_items, _upsert_deleted_table,
    collect_review_candidates, run_line_review,
)


def line_key(element, wave_nm):
    return f'{str(element).strip()}|{float(wave_nm):.4f}'


def spectrum_signature(spectrum):
    h = hashlib.sha256()
    for name in ('waveobs', 'flux', 'err'):
        h.update(np.asarray(spectrum[name], dtype='<f8').tobytes())
    return h.hexdigest()


def _state_path(output_folder, target):
    return Path(output_folder) / 'linemasks' / f'local_ew_retry_{target}.json'


def _read_state(output_folder, target):
    p = _state_path(output_folder, target)
    return json.loads(p.read_text()) if p.exists() else {'version': 1, 'lines': {}}


def _write_state(output_folder, target, state):
    p = _state_path(output_folder, target)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(p)


def apply_local_refits(linemasks, *, output_folder, target, star_spectrum=None):
    """Replay the chosen original/new fit, including profile fields and EW error."""
    state = _read_state(output_folder, target)
    signature = spectrum_signature(star_spectrum) if star_spectrum is not None else None
    out = linemasks.copy()
    for i, row in enumerate(out):
        record = state['lines'].get(line_key(row['element'], row['wave_nm']))
        if not record or record['status'] not in ('accepted', 'original_accepted'):
            continue
        if signature is not None and record['spectrum_sha256'] != signature:
            raise ValueError('Local EW fit belongs to a different spectrum. '
                             'Archive/reset its local_ew_retry JSON before reusing this target.')
        values = record['original_fit_values'] if record['status'] == 'original_accepted' else record['fit_values']
        for name, value in values.items():
            if name in out.dtype.names:
                out[name][i] = value
    return out


def local_plot_inputs(spectrum, line, output_folder):
    """Use the saved local flux as well as the saved fit when regenerating QA PNGs."""
    for path in (Path(output_folder) / 'linemasks').glob('local_ew_retry_*.json'):
        record = json.loads(path.read_text())['lines'].get(line_key(line['element'], line['wave_nm']))
        if record and record['status'] == 'accepted':
            row = line.copy()
            for k, value in record['fit_values'].items():
                if k in row.dtype.names:
                    row[k] = value
            with np.load(record['profile_path']) as data:
                return data['spectrum'].copy(), row
        if record and record['status'] == 'original_accepted':
            row = line.copy()
            for k, value in record['original_fit_values'].items():
                if k in row.dtype.names:
                    row[k] = value
            return spectrum, row
    return spectrum, line


def _row_values(row):
    """JSON-safe snapshot (null preserves an unknown floating-point error)."""
    values = {}
    for name in row.dtype.names:
        value = row[name].item() if hasattr(row[name], 'item') else row[name]
        values[name] = None if isinstance(value, float) and not np.isfinite(value) else value
    return values


def _restore_original_metadata(record, row, *, ispec, audit_dir):
    """Migrate old pending records without inventing missing original fit fields."""
    if record.get('original_fit_values'):
        return
    original_ew = record['original_ew_mA']
    if original_ew is None:
        return
    if np.isclose(row['ew'], original_ew, rtol=0, atol=1e-8):
        record['original_fit_values'] = _row_values(row)
        return
    snapshot = audit_dir/'fitted_linemasks_before_retry.txt'
    if snapshot.exists():
        masks = np.atleast_1d(ispec.read_line_regions(str(snapshot)))
        hit = np.flatnonzero((masks['element'] == row['element']) &
                            (abs(masks['wave_nm']-row['wave_nm']) < .0001))
        if len(hit) == 1 and np.isclose(masks['ew'][hit[0]], original_ew, rtol=0, atol=.051):
            record['original_fit_values'] = _row_values(masks[hit[0]])
            # The audit's EW is more precise than iSpec's 0.1 mA text export.
            # Keep the number actually offered by the Origin EW button.
            record['original_fit_values']['ew'] = float(original_ew)
            if 'ewr' in masks.dtype.names and masks['wave_nm'][hit[0]] > 0 and original_ew > 0:
                record['original_fit_values']['ewr'] = float(np.log10(original_ew/10000/masks['wave_nm'][hit[0]]))


def _plot_original(path, spectrum, values):
    """Fallback if an old run has fitted parameters but no original PNG backup."""
    import matplotlib.pyplot as plt
    mu, sig, amplitude, baseline = (values[k] for k in ('mu','sig','A','baseline'))
    mask = abs(spectrum['waveobs']-mu) <= .25
    wave = spectrum['waveobs'][mask]
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(wave, spectrum['flux'][mask], lw=.7, label='Original spectrum')
    ax.plot(wave, baseline + amplitude*np.exp(-.5*((wave-mu)/sig)**2), 'r--', label='Original Gaussian')
    ax.set(xlabel='Wavelength (nm)', ylabel='Normalized flux',
           title=f"First QA: {values['element']} {values['wave_A']:.3f} A; EW = {values['ew']:.2f} mA")
    ax.legend(); fig.tight_layout(); fig.savefig(path,dpi=120); plt.close(fig)


def _match_line(item, linemasks):
    hit = np.flatnonzero((np.char.strip(linemasks['element'].astype(str)) == item.element.strip())
                         & (np.abs(linemasks['wave_nm'] - item.wave_nm) <= 0.0006))
    if len(hit) != 1:
        raise ValueError(f'Expected one fitted mask for {item.element} {item.wave_nm:.4f}; got {len(hit)}')
    return int(hit[0])


def choose_retry_lines(items, linemasks, *, attempted=(), deleted_keys=()):
    """Manual union automatic: all rejected, some EW>20, no manual request per element.

    Group by chemical element (Fe I and Fe II together), but identify individual
    transitions by species and exact fitted-mask wavelength. At least one finite
    EW must be strictly greater than 20 mA. Never make a third attempt.
    """
    attempted, deleted_keys = set(attempted), set(deleted_keys)
    by_key = {line_key(x.element, x.wave_nm): x for x in items}
    manual_symbols = {x.element.split()[0] for x in items if x.reason == LOCAL_RENORMALIZE}
    selected = {k: 'manual' for k, x in by_key.items()
                if x.reason == LOCAL_RENORMALIZE and k not in attempted}
    symbols = {x.element.split()[0] for x in items}
    for sym in symbols - manual_symbols:
        rows = [r for r in linemasks if str(r['element']).split()[0] == sym]
        if not any(np.isfinite(r['ew']) and r['ew'] > 20 for r in rows):
            continue
        keys = [line_key(r['element'], r['wave_nm']) for r in rows]
        # Include surviving lines outside this QA subset; they must prevent rescue.
        def retained(k, r):
            if k in by_key:
                return by_key[k].reason == 'good' and r['ew'] > 0
            return k not in deleted_keys and r['ew'] > 0
        if any(retained(k, r) for k, r in zip(keys, rows)):
            continue
        selected.update({k: 'automatic_has_ew_above_20_rejected' for k in keys
                         if k in by_key and k not in attempted})
    return [(by_key[k], trigger) for k, trigger in selected.items()]


def refit_local_line(spectrum, line, *, all_linemasks=None, half_window_nm=0.20,
                     continuum_windows=None):
    """Fit sideband C(lambda), then C(lambda)*(1-depth*Gaussian) in the line.

    C is linear. Automatic continuum anchors are quiet upper-envelope *bins*,
    not the highest individual noisy pixels. Known neighboring line intervals
    are excluded. Both sides must provide anchors. Optional explicit continuum
    windows (nm pairs) override automatic selection for difficult regions.
    The core cannot pull down the sideband continuum. A full covariance,
    including implicit shape response to continuum errors, gives EW error.
    Gaussian assumption matches the existing pipeline; blends still need QA.
    """
    nominal = float(line['wave_nm'])
    centre = float(line['mu'])
    if not np.isfinite(centre) or abs(centre - nominal) > 0.01:
        centre = nominal
    mask = np.abs(spectrum['waveobs'] - centre) <= half_window_nm
    indices = np.flatnonzero(mask)
    local = spectrum[mask].copy()
    valid = np.isfinite(local['waveobs']) & np.isfinite(local['flux']) & (local['flux'] > 0)
    local, indices = local[valid], indices[valid]
    w, f = local['waveobs'], local['flux']
    if len(w) < 40 or np.any(np.diff(w) <= 0):
        raise ValueError('Insufficient samples or non-monotonic local wavelength grid')
    step = float(np.median(np.diff(w)))
    if np.max(np.diff(w)) > 5 * step:
        raise ValueError('Gap in local spectral window')
    noise = float(1.4826 * np.median(np.abs(np.diff(f) - np.median(np.diff(f)))) / np.sqrt(2))
    e = np.asarray(local['err'], dtype=float).copy()
    valid_err = np.isfinite(e) & (e > 0)
    if not np.isfinite(noise) or noise <= 0:
        noise = float(np.median(e[valid_err])) if valid_err.any() else np.nan
    if not np.isfinite(noise) or noise <= 0:
        raise ValueError('Cannot estimate positive pixel noise')
    estimated_noise = not valid_err.all()
    e[~valid_err] = noise
    sigma0 = float(line['sig'])
    sigma0 = np.clip(sigma0 if np.isfinite(sigma0) and sigma0 > 0 else centre / 45500 / 2.355,
                     max(step, centre / 150000 / 2.355), centre / 15000 / 2.355)
    core_exclusion = max(4 * sigma0, 0.035)
    side = np.abs(w - centre) > core_exclusion
    if all_linemasks is not None:
        for other in all_linemasks:
            if abs(other['wave_nm'] - nominal) < 0.001:
                continue
            lo, hi = float(other['wave_base']), float(other['wave_top'])
            if lo < hi:
                side &= ~((w >= lo - step) & (w <= hi + step))
    anchors = np.zeros(len(w), dtype=bool)
    if continuum_windows is not None:
        for lo, hi in continuum_windows:
            anchors |= (w >= lo) & (w <= hi)
        anchors &= side
    else:
        # Selecting quiet bins reduces upward noise selection. The final QA
        # plots all anchors so residual blends/pseudo-continuum are visible.
        bin_width = max(0.02, 6 * step)
        for direction in (-1, 1):
            bins = []
            for lo in np.arange(w[0], w[-1], bin_width):
                b = side & ((w - centre) * direction > 0) & (w >= lo) & (w < lo + bin_width)
                if b.sum() < 4:
                    continue
                median = np.median(f[b])
                scatter = 1.4826 * np.median(np.abs(f[b] - median))
                # Use observed pixel-to-pixel noise for anchor selection. Large
                # extraction error bars must not turn real absorption into continuum.
                if scatter <= 2.5 * noise:
                    bins.append((b, median, noise / np.sqrt(b.sum())))
            if len(bins) < 2:
                raise ValueError('Insufficient quiet continuum bins on both sides')
            level = np.percentile([x[1] for x in bins], 80)
            for b, median, se in bins:
                if abs(median - level) <= max(3 * se, 0.004):
                    anchors |= b
    if min(np.sum(anchors & (w < centre)), np.sum(anchors & (w > centre))) < 8:
        raise ValueError('Fewer than eight continuum anchors on either side')
    x = (w - centre) / half_window_nm
    design = np.column_stack([np.ones(len(w)), x])
    for _ in range(4):
        coef = np.linalg.lstsq(design[anchors] / e[anchors, None], f[anchors] / e[anchors], rcond=None)[0]
        keep = np.abs(f - design @ coef) <= 3 * noise
        anchors &= keep
    if min(np.sum(anchors & (w < centre)), np.sum(anchors & (w > centre))) < 8:
        raise ValueError('Continuum anchors failed outlier review')
    lo, hi = float(line['wave_base']), float(line['wave_top'])
    if not lo < centre < hi:
        lo, hi = centre - 3 * sigma0, centre + 3 * sigma0
    core = (w >= lo) & (w <= hi)
    if core.sum() < 7:
        raise ValueError('Insufficient pixels in original line interval')
    # Recompute after the last clipping iteration.
    weighted_design = design[anchors] / e[anchors, None]
    coef = np.linalg.lstsq(weighted_design, f[anchors] / e[anchors], rcond=None)[0]
    continuum = design @ coef
    if not .75 < coef[0] < 1.25 or abs(coef[1]) >= .2 or np.min(continuum) <= 0:
        raise ValueError('Sideband continuum outside safe bounds')
    # A depressed pseudo-continuum in a crowded region must not be accepted
    # merely because the few selected anchors admit a straight line.
    excess = (f-continuum)[side]
    if len(excess) and np.percentile(excess, 99) > max(5*noise, .025*np.median(continuum)):
        raise ValueError('Continuum anchors underestimate the local upper envelope')
    anchor_chi2 = np.sum(((f-continuum)/e)[anchors]**2) / (anchors.sum()-2)
    continuum_cov = np.linalg.inv(weighted_design.T @ weighted_design) * max(1., anchor_chi2)
    def model(p):
        depth, mu, sig = p
        return continuum * (1 - depth * np.exp(-0.5 * ((w - mu) / sig)**2))
    depth0 = np.clip(1 - np.min(f[core] / continuum[core]), 0.001, 0.95)
    sig_min = max(step / 2, centre / 200000 / 2.355)
    sig_max = min((hi - lo) / 2, centre / 12000 / 2.355)
    mu_span = min(0.01, (hi - lo) / 3)
    lower = [0.000001, max(lo, centre - mu_span), sig_min]
    upper = [0.98, min(hi, centre + mu_span), sig_max]
    initial = [depth0, centre, np.clip(sigma0, sig_min*1.01, sig_max*.99)]
    if np.any(np.asarray(initial) <= lower) or np.any(np.asarray(initial) >= upper):
        raise ValueError('Continuum or initial line fit outside safe bounds')
    fit = least_squares(lambda p: ((model(p) - f) / e)[core], initial,
                        bounds=(lower, upper), x_scale='jac', max_nfev=3000)
    if not fit.success or np.any(fit.active_mask):
        raise ValueError('Local Gaussian fit failed or reached a bound')
    dof = int(core.sum() - 3)
    chi2 = float(np.sum(fit.fun**2) / dof)
    jtj = fit.jac.T @ fit.jac
    if np.linalg.matrix_rank(jtj) < 3:
        raise ValueError('Local continuum/line covariance is singular')
    inverse = np.linalg.inv(jtj)
    conditional_cov = inverse * max(1., chi2)
    c0, c1 = coef
    depth, mu, sig = fit.x
    gaussian = np.exp(-0.5 * ((w-mu)/sig)**2)
    continuum_jac = design[core] * ((1-depth*gaussian)/e)[core, None]
    # d(shape)/d(continuum) from the conditional least-squares normal equations.
    response = -inverse @ fit.jac.T @ continuum_jac
    cross_cov = continuum_cov @ response.T
    cov = np.block([[continuum_cov, cross_cov],
                    [cross_cov.T, conditional_cov + response @ continuum_cov @ response.T]])
    area_scale = np.sqrt(2 * np.pi) * 10000
    ew = float(depth * sig * area_scale)
    gradient = np.array([0., 0., sig * area_scale, 0., depth * area_scale])
    ew_err = float(np.sqrt(gradient @ cov @ gradient))
    if not np.isfinite(ew_err) or ew_err <= 0:
        raise ValueError('Invalid propagated EW uncertainty')
    if np.min(continuum) <= 0:
        raise ValueError('Nonpositive fitted continuum')
    corrected = local.copy()
    corrected['flux'] = f / continuum
    corrected['err'] = e / continuum
    cont_var = np.einsum('ij,jk,ik->i', design, cov[:2, :2], design)
    fwhm = 2 * np.sqrt(2 * np.log(2)) * sig
    fit_values = dict(mu=float(mu), sig=float(sig), A=float(-depth), baseline=1., gamma=9999.,
        ew=ew, ew_err=ew_err, ewr=float(np.log10(ew / 10000 / nominal)),
        integrated_flux=ew / 10000, fwhm=float(fwhm), fwhm_kms=float(fwhm / mu * 299792.458),
        R=float(mu / fwhm), mu_err=float(np.sqrt(cov[3, 3])),
        wave_peak=float(mu), peak=int(indices[np.argmin(abs(w-mu))]),
        base_fit=int(indices[np.flatnonzero(core)[0]]), top_fit=int(indices[np.flatnonzero(core)[-1]]),
        wave_base_fit=float(w[core][0]), wave_top_fit=float(w[core][-1]),
        depth=float(depth), depth_fit=float(depth), relative_depth_fit=float(depth),
        rms=float(np.sqrt(np.mean(((model(fit.x)-f)[core] / continuum[core])**2))),
        mean_flux=float(np.mean(corrected['flux'][core])), mean_flux_continuum=1.,
        diff_wavelength=float(hi-lo), snr=float(np.median(continuum/e)), discarded='False')
    return dict(fit_values=fit_values, spectrum=corrected, original_flux=f,
                continuum=continuum, continuum_sigma=np.sqrt(np.maximum(0, cont_var)),
                anchors=anchors, core=core, fit_parameters=[float(c0), float(c1), *fit.x.tolist()],
                covariance=cov.tolist(), reduced_chi2=chi2, noise_estimated=estimated_noise,
                half_window_nm=float(half_window_nm))


def refit_with_continuum_fallback(spectrum, line, *, all_linemasks=None, options=None):
    """Widen the local window only when too few continuum anchors are available."""
    opts = options or {}
    base = float(opts.get('local_ew_half_window_nm', .20))
    windows = sorted(set([base, min(base*1.5, .5), min(base*2, .5)])) if 0 < base <= .5 else [base]
    key = line_key(line['element'], line['wave_nm'])
    for width in windows:
        try:
            return refit_local_line(spectrum, line, all_linemasks=all_linemasks, half_window_nm=width,
                continuum_windows=opts.get('local_ew_continuum_windows', {}).get(key))
        except ValueError as exc:
            if width == windows[-1] or not any(s in str(exc).lower() for s in ('anchor', 'continuum bins')):
                raise


def _plot_retry(path, line, result=None, failure=None):
    import matplotlib.pyplot as plt
    from atmos_lbl_pipeline import silence_matplotlib_font_warnings
    silence_matplotlib_font_warnings()
    fig, axes = plt.subplots(2 if result else 1, 1, figsize=(9, 7 if result else 4))
    if result:
        upper, lower = axes
        w = result['spectrum']['waveobs']
        upper.plot(w, result['original_flux'], lw=.7, label='Before local normalization')
        upper.plot(w, result['continuum'], 'r-', label='Fitted local continuum')
        a = result['anchors']
        upper.scatter(w[a], result['original_flux'][a], s=6, c='green', label='Continuum anchors')
        upper.legend(fontsize=9)
        p = result['fit_values']
        lower.plot(w, result['spectrum']['flux'], lw=.7, label='Locally normalized')
        lower.plot(w, 1 + p['A'] * np.exp(-.5*((w-p['mu'])/p['sig'])**2), 'r--', label='New Gaussian')
        lower.axvspan(float(line['wave_base']), float(line['wave_top']), alpha=.08)
        lower.set_title(f"EW: {line['ew']:.2f} -> {p['ew']:.2f} +/- {p['ew_err']:.2f} mA")
        lower.legend(fontsize=9)
    else:
        axes.text(.05, .5, 'Local refit failed; this line cannot be accepted.\n' + str(failure),
                  transform=axes.transAxes, wrap=True)
    fig.suptitle(f"Second QA: {line['element']} {float(line['wave_A']):.3f} A")
    fig.supxlabel('Wavelength (nm)'); fig.supylabel('Flux')
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def run_review_with_local_retry(*, ispec, star_spectrum, linemasks, output_folder,
                                target, figure_dir, deleted_tsv_path=None,
                                linelist_reference_path=None, pipeline_options=None,
                                reviewer=None):
    """Blocking QA -> local fits -> blocking subset QA -> persist final decisions."""
    from run_context import local_ew_retry_enabled
    opts = pipeline_options or {}
    reviewer = reviewer or run_line_review
    reasons = list(opts.get('line_review_reason_options') or DEFAULT_REASON_OPTIONS)
    reasons = [r for r in reasons if r not in (LOCAL_RENORMALIZE, FAILED_BOTH)]
    enabled = local_ew_retry_enabled(opts) and not opts.get('input_has_external_ews', False)
    if enabled:
        reasons.insert(reasons.index('other') + 1 if 'other' in reasons else len(reasons), LOCAL_RENORMALIZE)
    parse_reasons = list(dict.fromkeys(reasons + [LOCAL_RENORMALIZE, FAILED_BOTH]))
    deleted_path = deleted_tsv_path or str(Path(output_folder)/'linemasks'/'Lines_deleted.tsv')
    state = _read_state(output_folder, target)
    linemasks = apply_local_refits(linemasks, output_folder=output_folder, target=target,
                                 star_spectrum=star_spectrum)
    if any(r['status'] == 'pending' and r['spectrum_sha256'] != spectrum_signature(star_spectrum)
           for r in state['lines'].values()):
        raise ValueError('Pending local fit belongs to a different spectrum; archive/reset its audit first')
    items = collect_review_candidates(figure_dir, parse_reasons)
    items = _apply_existing_decisions(items, deleted_path, parse_reasons)
    matched, terminal = [], []
    for item in items:
        try:
            row = linemasks[_match_line(item, linemasks)]
        except ValueError:
            continue  # Old PNGs without a current fitted mask are not reviewable.
        item.wave_nm, item.wave_a = float(row['wave_nm']), float(row['wave_A'])
        record = state['lines'].get(line_key(item.element, item.wave_nm))
        if item.reason == FAILED_BOTH or (record and record['status'] in ('rejected', 'fit_failed')):
            terminal.append(replace(item, reason=FAILED_BOTH))
            continue
        item.rescue_attempted = bool(record)
        if item.reason == LOCAL_RENORMALIZE and not enabled:
            item.reason, item.note = 'other', 'Local retry disabled; line remains rejected'
        matched.append(item)
    if terminal:
        # The audit is authoritative even if someone regenerated good PNGs or
        # cleared the deletion TSV without resetting this rescue history.
        ref = _load_linelist_reference(linelist_reference_path)
        _upsert_deleted_table(deleted_path, _build_new_deleted_rows(terminal, ref), reviewed_items=terminal)
        _sync_images_for_items(terminal, figure_dir, parse_reasons)
        _refresh_deleted_summary(figure_dir)
    common = dict(output_folder=output_folder, target=target, figure_dir=figure_dir,
                  deleted_tsv_path=deleted_path, linelist_reference_path=linelist_reference_path,
                  ui_mode=opts.get('line_review_ui_mode', 'inline'), load_existing_decisions=False)
    pending_items = [x for x in matched if state['lines'].get(
        line_key(x.element, x.wave_nm), {}).get('status') == 'pending']
    matched = [x for x in matched if x not in pending_items]
    first = (reviewer(**common, candidates_override=matched, reason_options=reasons)
             if matched or not pending_items else dict(status='saved', items=[]))
    if first['status'] == 'cancelled':
        raise RuntimeError('Line QA cancelled; stopping before abundance calculation')
    if first['status'] != 'saved' or not enabled:
        return first
    reviewed = first['items']
    deleted = pd.read_csv(deleted_path, sep='\t') if Path(deleted_path).exists() else pd.DataFrame()
    deleted_keys = [line_key(r.element, r.wave_nm) for r in deleted.itertuples()]
    todo = choose_retry_lines(reviewed, linemasks, attempted=state['lines'], deleted_keys=deleted_keys)
    todo += [(x, state['lines'][line_key(x.element, x.wave_nm)]['trigger']) for x in pending_items]
    if not todo:
        return first
    signature = spectrum_signature(star_spectrum)
    ref = _load_linelist_reference(linelist_reference_path)
    audit_dir = Path(output_folder) / 'local_ew_retry'
    audit_dir.mkdir(parents=True, exist_ok=True)
    # Original fitted table remains available even after accepted results are saved.
    lm_path = Path(output_folder)/'linemasks'/f'{target}_melendez2014_star_fitted_linemasks.txt'
    if lm_path.exists() and not (audit_dir/'fitted_linemasks_before_retry.txt').exists():
        shutil.copy2(lm_path, audit_dir/'fitted_linemasks_before_retry.txt')
    retry_items = []
    for item, trigger in todo:
        row = linemasks[_match_line(item, linemasks)]
        key = line_key(item.element, item.wave_nm)
        previous = state['lines'].get(key)
        record = dict(status='pending', trigger=trigger,
                      original_ew_mA=float(row['ew']) if np.isfinite(row['ew']) else None,
                      original_fit_values=_row_values(row),
                      first_qa_reason=item.reason, first_qa_note=item.note, spectrum_sha256=signature)
        folder = Path(item.image_path).parent
        if folder.name == 'deleted':
            folder = folder.parent
        plot_path = folder / item.image_name
        original_png = audit_dir / (key.replace('|', '_') + '_before.png')
        if Path(item.image_path).exists() and not previous:
            shutil.copy2(item.image_path, original_png)
        failed = False
        try:
            if previous and previous['status'] == 'pending':
                if previous['spectrum_sha256'] != signature:
                    raise RuntimeError('Pending local fit belongs to a different spectrum; reset its audit first')
                record = previous
                if previous['fit_failed']:
                    raise ValueError(previous['fit_error'])
                with np.load(previous['profile_path']) as data:
                    result = {k: data[k].copy() for k in data.files}
                result.update({k: previous[k] for k in ('fit_values', 'fit_parameters', 'covariance',
                                                       'reduced_chi2', 'noise_estimated', 'half_window_nm')})
            else:
                result = refit_with_continuum_fallback(star_spectrum, row,
                    all_linemasks=linemasks, options=opts)
            record.update({k: result[k] for k in ('fit_values', 'fit_parameters', 'covariance',
                                                 'reduced_chi2', 'noise_estimated', 'half_window_nm')})
            profile = audit_dir / (key.replace('|', '_') + '_profile.npz')
            np.savez_compressed(profile, spectrum=result['spectrum'], original_flux=result['original_flux'],
                continuum=result['continuum'], continuum_sigma=result['continuum_sigma'],
                anchors=result['anchors'], core=result['core'])
            record['profile_path'] = str(profile)
            _plot_retry(plot_path, row, result)
            note = 'Compare both fits; select Origin EW ok or new EW ok, or choose a rejection reason'
        except (ValueError, np.linalg.LinAlgError, RuntimeError) as exc:
            failed = True
            record.update(fit_error=str(exc))
            note = f'Local refit failed: {exc}'
            _plot_retry(plot_path, row, failure=exc)
        record['fit_failed'] = failed
        _restore_original_metadata(record, row, ispec=ispec, audit_dir=audit_dir)
        origin_available = (bool(record.get('original_fit_values')) and record['original_ew_mA'] is not None
                            and record['original_ew_mA'] > 0)
        if not original_png.exists() and origin_available:
            _plot_original(original_png, star_spectrum, record['original_fit_values'])
        record['original_image_path'] = str(original_png) if original_png.exists() else ''
        new_png = audit_dir / (key.replace('|', '_') + '_after.png')
        shutil.copy2(plot_path, new_png)
        record['new_image_path'] = str(new_png)
        state['lines'][key] = record
        retry_items.append(replace(item, image_path=str(plot_path), reason='other', note=note,
                                   fit_failed=failed, rescue_attempted=True,
                                   original_image_path=record['original_image_path'],
                                   original_ew=record['original_ew_mA'],
                                   new_ew=None if failed else record['fit_values']['ew'],
                                   origin_available=origin_available,
                                   ew_choice=''))
        print(f'[LocalEW] {key}: {trigger}; {note}')
    _write_state(output_folder, target, state)
    # Fail closed while the second QA is pending (including Cancel/interruption).
    pending = [replace(x, reason=LOCAL_RENORMALIZE) for x in retry_items]
    _upsert_deleted_table(deleted_path, _build_new_deleted_rows(pending, ref), reviewed_items=pending)
    second_reasons = [r for r in reasons if r not in (LOCAL_RENORMALIZE, 'good')]
    second = reviewer(**common, candidates_override=retry_items,
                      reason_options=second_reasons, second_pass=True)
    if second['status'] != 'saved':
        raise RuntimeError('Second QA not saved; local lines remain excluded. Rerun QA to finish.')
    final = second['items']
    for item in final:
        record = state['lines'][line_key(item.element, item.wave_nm)]
        choice = item.ew_choice or (item.reason if item.reason in EW_CHOICES else '')
        if choice == ORIGIN_EW_OK and item.origin_available and record.get('original_fit_values'):
            record['status'] = 'original_accepted'
        elif choice == NEW_EW_OK and not record['fit_failed']:
            record['status'] = 'accepted'
        else:
            record['status'] = 'fit_failed' if record['fit_failed'] else 'rejected'
        approved = record['status'] in ('accepted', 'original_accepted')
        if approved:
            item.reason, item.ew_choice = 'good', choice
        if item.reason != 'good' and item.reason != FAILED_BOTH:
            item.note = f'Second QA: {item.reason}. {item.note}'.strip()
        record['ew_choice'] = item.ew_choice if approved else ''
        record['second_qa_reason'], record['second_qa_note'] = item.reason, item.note
        if not approved:
            item.reason = FAILED_BOTH
    _write_state(output_folder, target, state)
    _upsert_deleted_table(deleted_path, _build_new_deleted_rows(final, ref), reviewed_items=final)
    _sync_images_for_items(final, figure_dir, parse_reasons)
    for item in final:
        if item.reason == 'good' and item.ew_choice == ORIGIN_EW_OK:
            record = state['lines'][line_key(item.element, item.wave_nm)]
            if record['original_image_path']:
                shutil.copy2(record['original_image_path'], item.image_path)
    _refresh_deleted_summary(figure_dir)
    # Remove superseded manual EW edits for the newly accepted lines only.
    modified_path = Path(output_folder)/'linemasks'/'Lines_EW_modified.tsv'
    accepted = {line_key(x.element, x.wave_nm) for x in final if x.reason == 'good'}
    if accepted and modified_path.exists():
        edits = pd.read_csv(modified_path, sep='\t')
        if not edits.empty:
            shutil.copy2(modified_path, audit_dir/'Lines_EW_modified_before_retry.tsv')
            edits = edits[[line_key(r.element, r.wave_nm) not in accepted for r in edits.itertuples()]]
            edits.to_csv(modified_path, sep='\t', index=False)
    updated = apply_local_refits(linemasks, output_folder=output_folder, target=target,
                                star_spectrum=star_spectrum)
    ispec.write_line_regions(updated, str(lm_path), extended=True)
    return dict(status='saved', first=first, second=second, retried=len(todo),
                accepted=len(accepted), accepted_original=sum(x.ew_choice == ORIGIN_EW_OK for x in final),
                accepted_new=sum(x.ew_choice == NEW_EW_OK for x in final),
                audit_path=str(_state_path(output_folder, target)))
