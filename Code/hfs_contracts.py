"""Explicit paired solar/stellar HFS baselines and traceable quality checks."""
from pathlib import Path
import hashlib,json,re
import numpy as np
import pandas as pd
from abundance_contracts import species_token

def no_hfs_logeps(row):
    explicit=pd.to_numeric(row.get('logeps_star_nohfs',np.nan),errors='coerce')
    if np.isfinite(explicit):return float(explicit)
    old=pd.to_numeric(row.get('hfs',np.nan),errors='coerce')
    if np.isfinite(old):
        # Legacy hfs stores the pre-replacement differential, not a correction.
        return float(row['logeps_sun'])+float(old)
    return float(row['logeps_star'])

def component_signature(line_file,n_components):
    """Hash the actual normalized HFS components, excluding EW/background."""
    rows=[]
    for line in Path(line_file).read_text().splitlines()[1:]:
        try:values=list(map(float,line.split()[:4]))
        except ValueError:continue
        if len(values)==4:rows.append([abs(values[0]),*values[1:]])
        if len(rows)==int(n_components):break
    if len(rows)!=int(n_components) or len(rows)<2:return ''
    return hashlib.sha256(json.dumps(sorted(rows),separators=(',',':')).encode()).hexdigest()

def active_line_file(run_dir):
    run=Path(run_dir)
    par=run/'batch.par'
    if par.exists():
        m=re.search(r'^\s*lines_in\s+[\'\"]?([^\s\'\"]+)',par.read_text(),re.M)
        if m:return run/m.group(1)
    return run/'lines.in'

def footer_consistent(model):
    """Reject a solar HFS model whose composition differs from its own layers."""
    try:
        text=Path(model).read_text();lines=text.splitlines()
        header=re.search(r'\[FE/H\]=\s*([+-]?[\d.]+)',text,re.I)
        i=next(i for i,l in enumerate(lines) if l.startswith('NATOMS'))
        footer=float(lines[i].split()[-1])
        return header is not None and abs(float(header.group(1))-footer)<=.011
    except (OSError,ValueError,StopIteration):return False

def profile_quality(output_folder,target,element,wave_A,observed_ew):
    """Audit the measured profile, without inventing a universal RMS threshold.

    Gaussian residuals alone cannot reject an HFS/non-Gaussian line. Missing
    profiles (e.g. published external EWs) remain explicitly unavailable.
    """
    folder=Path(output_folder)/'linemasks'
    candidates=[folder/f'{target}_melendez2014_star_fitted_linemasks.filtered_for_abund.txt',
                folder/f'{target}_melendez2014_star_fitted_linemasks.txt']
    result={'profile_quality':'unavailable','profile_rms':np.nan,'profile_ew_err_mA':np.nan,
            'profile_fwhm_kms':np.nan,'profile_ew_mA':np.nan,'profile_source':''}
    for path in candidates:
        if not path.exists():continue
        frame=pd.read_csv(path,sep='\t')
        hit=frame[(frame.element.map(species_token)==species_token(element)) &
                  (abs(frame.wave_A-float(wave_A))<=.003)]
        if len(hit)!=1:return result
        row=hit.iloc[0];result['profile_source']=str(path)
        for dst,src in [('profile_rms','rms'),('profile_ew_err_mA','ew_err'),
                        ('profile_fwhm_kms','fwhm_kms'),('profile_ew_mA','ew')]:
            result[dst]=pd.to_numeric(row.get(src,np.nan),errors='coerce')
        # Manually replaced/external EWs are not the same measurement as this fit.
        if not np.isclose(result['profile_ew_mA'],float(observed_ew),atol=.01,rtol=0):
            result['profile_quality']='different_ew_requires_review';return result
        discarded=str(row.get('discarded','False')).strip().lower() in {'true','1','yes'}
        invalid=discarded or any(np.isfinite(result[k]) and result[k]<=0
                                for k in ['profile_fwhm_kms','profile_ew_mA'])
        invalid=invalid or any(np.isfinite(result[k]) and result[k]<0
                               for k in ['profile_rms','profile_ew_err_mA'])
        result['profile_quality']='invalid_fit' if invalid else 'metrics_available'
        return result
    return result

def attach_solar_baselines(lines,ispec_dir,solar_label=None,is_solar_reference=False):
    out=lines.copy()
    for col in ['logeps_sun_nohfs','logeps_sun_hfs','d_xh_nohfs_paired']:
        out[col]=np.nan
    out['solar_component_signature']='';out['solar_pair_status']='missing'
    out['solar_label_used']=''
    if is_solar_reference:
        out['logeps_sun_nohfs']=out.logeps_sun_ref
        out['logeps_sun_hfs']=out.logeps_sun_ref
        out['d_xh_nohfs_paired']=out.logeps_star_ref-out.logeps_sun_ref
        out['solar_pair_status']='solar_absolute_reference'
        return out
    root=Path(ispec_dir)
    master=pd.read_csv(root/'input/solar_lines_instru.csv')
    master['species_token']=master.element.map(species_token)
    cache={}
    for i,r in out.iterrows():
        token=species_token(f'{r.element} {r.ion}')
        m=master[(master.species_token==token)&(abs(master.wave_A-float(r.wave_A))<=.003)]
        if solar_label is not None:m=m[m.solar_label==solar_label]
        else:m=m[np.isclose(m.logeps_sun,float(r.logeps_sun_ref),atol=.0011,rtol=0)]
        labels=m.solar_label.dropna().unique()
        if len(labels)!=1:
            out.at[i,'solar_pair_status']='ambiguous_or_missing_solar_label';continue
        label=str(labels[0]);out.at[i,'solar_label_used']=label
        if label not in cache:
            folder=root/'output/Sun'/label
            raw=folder/f'abundances_linebyline_{label}_raw.csv'
            hf=folder/'HFS/tables'/f'abundances_hfs_blends_{label}.csv'
            cache[label]=(pd.read_csv(raw) if raw.exists() else pd.DataFrame(),
                          pd.read_csv(hf) if hf.exists() else pd.DataFrame(),folder)
        raw,hf,folder=cache[label]
        if raw.empty or hf.empty:continue
        sr=raw[(raw.element.map(species_token)==token)&(abs(raw.wave_A-r.wave_A)<=.003)]
        sh=hf[(hf.element==r.element)&(abs(hf.wave_A-r.wave_A)<=.003)&(hf.status=='ok')]
        if len(sr)!=1:continue
        a0=no_hfs_logeps(sr.iloc[0]);out.at[i,'logeps_sun_nohfs']=a0
        out.at[i,'d_xh_nohfs_paired']=float(r.logeps_star_ref)-a0
        if len(sh)!=1:
            out.at[i,'solar_pair_status']='solar_hfs_not_ok';continue
        sh=sh.iloc[0];ah=float(sh.logeps_HFS)
        if abs(ah-float(r.logeps_sun_ref))>.0011:
            out.at[i,'solar_pair_status']='solar_library_not_synced';continue
        # Old run directories may have moved; resolve against this reference root.
        run=folder/'HFS/runs'/Path(str(sh.run_dir)).name
        if not footer_consistent(run/'model.in'):
            out.at[i,'solar_pair_status']='solar_model_requires_refresh';continue
        try:sig=component_signature(active_line_file(run),int(sh.n_hfs_components))
        except (OSError,ValueError):sig=''
        if not sig:continue
        out.at[i,'logeps_sun_hfs']=ah
        out.at[i,'solar_component_signature']=sig
        out.at[i,'solar_pair_status']='ok'
    return out

def apply_pair_quality(result,is_solar_reference=False,review_decisions=None):
    """No abundance-amplitude cut: use model/component/convergence provenance.

    A robust 4-MAD screen (at least four usable lines) marks candidates for
    review, not physically invalid lines. They are held out of a final summary
    until an explicit decision is recorded. This is not a calibrated p-value.
    """
    df=result.copy();df['quality_flags']='';df['used_in_summary']=False
    for i,r in df.iterrows():
        flags=[]
        if r.status!='ok':flags.append('moog_or_component_failure')
        if not is_solar_reference:
            if r.get('solar_pair_status')!='ok':flags.append(str(r.get('solar_pair_status','missing_solar_pair')))
            elif r.get('component_signature','')!=r.get('solar_component_signature',''):
                flags.append('solar_target_component_mismatch')
        if not np.isfinite(pd.to_numeric(r.get('[X/H]_HFS'),errors='coerce')):flags.append('nonfinite_abundance')
        if r.get('profile_quality')=='invalid_fit':flags.append('invalid_measured_profile')
        if flags and r.status=='ok':df.at[i,'status']='needs_reference_or_component_review'
        df.at[i,'quality_flags']=';'.join(flags)
        df.at[i,'used_in_summary']=not flags and r.status=='ok'
    for _,g in df[df.used_in_summary].groupby(['element','ion']):
        if len(g)<4:continue
        x=g['[X/H]_HFS'].to_numpy(float);med=np.median(x);mad=1.4826*np.median(abs(x-med))
        if mad<=0:continue
        for idx,value in zip(g.index,x):
            if abs(value-med)>4*mad:
                key=(str(df.at[idx,'element']),round(float(df.at[idx,'wave_A']),3))
                decision=(review_decisions or {}).get(key)
                df.at[idx,'quality_flags']='line_abundance_outlier_review'
                if decision=='accept':
                    df.at[idx,'quality_flags']+=';explicitly_accepted'
                else:
                    df.at[idx,'status']='needs_line_review'
                    df.at[idx,'used_in_summary']=False
    # An explicit reject is honored even when no statistical screen flags it.
    for idx,r in df.iterrows():
        if (review_decisions or {}).get((str(r.element),round(float(r.wave_A),3)))=='reject':
            df.at[idx,'status']='rejected_by_review';df.at[idx,'used_in_summary']=False
            df.at[idx,'quality_flags']+=';explicitly_rejected'
    return df
