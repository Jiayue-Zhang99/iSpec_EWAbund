"""Shared path, species and uncertainty conventions for the EW workflow."""
from pathlib import Path
import re
import numpy as np
import pandas as pd

def q2_dump_path(path):
    """Canonicalize only trailing q2 suffixes; safe on repeated saves."""
    p=Path(path)
    return str(p.with_name(re.sub(r'(?:_q2)+$', '', p.stem)+'.dump'))

def species_token(value):
    parts=str(value).strip().split()
    if not parts:return ''
    if parts[0]=='CH':return 'C CH'
    ion=parts[1] if len(parts)>1 else '1'
    return parts[0]+' '+{'I':'1','II':'2','III':'3'}.get(ion.upper(),ion)

def strict_quadrature(a,b):
    """Unknown uncertainties stay unknown; assumes independence if both known."""
    a,b=np.broadcast_arrays(np.asarray(a,float),np.asarray(b,float))
    return np.where(np.isfinite(a)&np.isfinite(b)&(a>=0)&(b>=0),np.hypot(a,b),np.nan)

def summarize_lines(values):
    x=np.asarray(values,float);x=x[np.isfinite(x)];n=len(x)
    scatter=float(np.std(x,ddof=1)) if n>1 else np.nan
    return dict(n_lines=n,mean=float(np.mean(x)) if n else np.nan,
                scatter=scatter,sem=scatter/np.sqrt(n) if n>1 else np.nan)

def combine_species_statistics(g):
    """Combine target lines equally, using n_i/N weights on ion summaries.

    Recover pooled line scatter from within-ion and between-ion sums of
    squares. Shared parameter responses use the same n_i/N weights, while
    independent mean measurement errors combine in quadrature. References
    marked ``reported`` are summary estimates without measured line counts:
    retain their documented summary combination, never invent line weights.
    The plotting column ``std_[X/H]`` remains the final mean uncertainty;
    ``line_scatter_[X/H]`` explicitly holds pooled line scatter.
    """
    vals=pd.to_numeric(g['[X/H]'],errors='coerce').to_numpy(float)
    keep=np.isfinite(vals);g=g.loc[keep];vals=vals[keep];k=len(vals)
    reported=g.get('error_kind',pd.Series('',index=g.index)).eq('reported').to_numpy()
    if reported.any() and not reported.all():
        raise ValueError('Do not mix published summary errors with target line statistics')
    reference=bool(k and reported.all())
    counts=pd.to_numeric(g.get('n_lines',pd.Series(np.nan,index=g.index)),errors='coerce').to_numpy(float)
    if not reference:
        if np.any(~np.isfinite(counts)) or np.any(counts<0) or np.any(counts!=np.floor(counts)):
            raise ValueError('Equal-line abundance combination requires valid integer n_lines')
        positive=counts>0;g=g.loc[positive];vals=vals[positive];counts=counts[positive];k=len(vals)
    n=float(np.sum(counts)) if k else 0.
    weights=np.full(k,1/k) if reference else counts/n if n else np.array([])
    mean=float(np.sum(weights*vals)) if k else np.nan
    errors=[]
    for _,r in g.iterrows():
        if str(r.get('error_kind',''))=='reported':e=r.get('std_[X/H]',np.nan)
        elif 'ew_err_mean_[X/H]' in g.columns:e=r.get('ew_err_mean_[X/H]',np.nan)
        elif 'stat_err_[X/H]' in g.columns:e=r.get('stat_err_[X/H]',np.nan)
        elif 'sem_[X/H]' in g.columns:e=r.get('sem_[X/H]',np.nan)
        else:
            s=pd.to_numeric(r.get('std_[X/H]',np.nan),errors='coerce')
            ni=pd.to_numeric(r.get('n_lines',np.nan),errors='coerce')
            e=s/np.sqrt(ni) if np.isfinite(s) and np.isfinite(ni) and ni>1 else np.nan
        errors.append(pd.to_numeric(e,errors='coerce'))
    errors=np.asarray(errors,float)
    errors=np.where(errors>=0,errors,np.nan)
    propagated=float(np.linalg.norm(weights*errors)) if k and np.isfinite(errors).all() else np.nan
    scatter=np.nan;pooled_sem=np.nan
    if not reference and n>1:
        s=pd.to_numeric(g.get('std_[X/H]',pd.Series(np.nan,index=g.index)),errors='coerce').to_numpy(float)
        # A one-line group has zero within-group sum of squares, even though
        # its separately estimated scatter is undefined. Unknown multi-line
        # scatter cannot be reconstructed and must not be replaced with zero.
        valid=(counts==1)|(np.isfinite(s)&(s>=0))
        if valid.all():
            within=np.sum((counts[counts>1]-1)*s[counts>1]**2)
            between=np.sum(counts*(vals-mean)**2)
            scatter=float(np.sqrt((within+between)/(n-1)))
            pooled_sem=scatter/np.sqrt(n)
    # Group SEMs are not a second, independent measurement-noise floor.
    # Reconstruct the pooled SEM instead. Explicit measurement/statistical
    # floor fields from the uncertainty pipeline remain additional evidence.
    candidates=([propagated] if reference else [pooled_sem,propagated]
                if 'ew_err_mean_[X/H]' in g or 'stat_err_[X/H]' in g
                else [pooled_sem] if np.isfinite(pooled_sem) else [propagated])
    available=[x for x in candidates if np.isfinite(x)]
    sem=max(available) if available else np.nan
    result={'[X/H]':mean,'std_[X/H]':sem,'stat_err_[X/H]':sem,
            'line_scatter_[X/H]':scatter,'sem_[X/H]':pooled_sem,
            'n_lines':n,'n_ions':k,'uncertainty_kind':'reported_summary_error' if reference else 'equal_line_statistical',
            'abundance_weighting':'reported_summary_equal' if reference else 'equal_per_line'}
    keys=['sensitivity_'+x for x in ['teff','logg','MH','vmic']]
    if all(key in g.columns for key in keys):
        means=[]
        for key in keys:
            arr=pd.to_numeric(g[key],errors='coerce').to_numpy(float)
            value=float(np.sum(weights*arr)) if len(arr) and np.isfinite(arr).all() else np.nan
            result[key]=value;means.append(value)
        pe=np.linalg.norm(means) if np.isfinite(means).all() else np.nan
        result.update({'stat_err_[X/H]':sem,'param_err_[X/H]':pe,
                       'std_[X/H]':float(np.hypot(sem,pe)),
                       'uncertainty_kind':'parameter_diagonal_plus_statistical_floor'})
    return pd.Series(result)

def xfe_uncertainty(frame,fe_row):
    """Subtract common parameter responses before squaring for X/Fe."""
    keys=['sensitivity_'+x for x in ['teff','logg','MH','vmic']]
    if all(k in frame.columns for k in keys) and 'stat_err_[X/H]' in frame:
        stat=strict_quadrature(frame['stat_err_[X/H]'],fe_row['stat_err_[X/H]'])
        delta=frame[keys].to_numpy(float)-np.asarray([fe_row[k] for k in keys],float)
        params=np.sqrt(np.sum(delta**2,axis=1))
        return strict_quadrature(stat,params)
    return strict_quadrature(frame['std_[X/H]'],fe_row['std_[X/H]'])
