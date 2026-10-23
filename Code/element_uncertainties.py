"""Preserve the iSpec abundance estimator; propagate measurement/parameter errors.

The Sun is a fixed atmospheric anchor. Target parameter errors are marginal,
independent approximations, as in q2's sensitivity calculation; covariance
between different elements induced by the same parameter is retained.
"""
from pathlib import Path
import os,tempfile,json,shutil,contextlib
import numpy as np
import pandas as pd
from abundance_contracts import species_token

PARAMETERS=('teff','logg','MH','vmic')
TRIPLET=np.array([7771.94,7774.16,7775.39])

def q2_oxygen(raw,params,sun_params,workdir,star_ews=None,sun_ews=None):
    """Call the installed q2 element interface on both complete O triplets.

    q2 currently assumes its O I input consists of these three transitions.
    No extrapolation/nonfinite NLTE result is silently accepted.
    """
    import q2
    rows=raw.copy()
    selected=[]
    for wave in TRIPLET:
        hit=rows[(rows.element.map(species_token)=='O 1') & (abs(rows.wave_A-wave)<.05)]
        if len(hit)!=1:raise ValueError('run_O_triple requires exactly one measured star/Sun match for each 777 nm triplet line')
        selected.append(hit.iloc[0])
    r=pd.DataFrame(selected);indices=r.index.to_numpy()
    # The q2 interface hardcodes these wavelength identities; preserve measured
    # wavelengths in the exported table and use canonical IDs inside q2 only.
    def make_star(name,p,ews):
        s=q2.Star(name,teff=p['teff'],logg=p['logg'],feh=p['MH'],vt=p['vmic'])
        s.linelist={'wavelength':TRIPLET.copy(),'species':np.full(3,8.),
                    'ep':r.EP.to_numpy(float),'gf':r.loggf.to_numpy(float),'ew':np.asarray(ews,float)}
        if not np.isfinite(s.linelist['ew']).all() or (s.linelist['ew']<=0).any():raise ValueError('Invalid oxygen EW')
        s.get_model_atmosphere(grid='marcs');return s
    a=make_star('Target',params,r.ew_mA if star_ews is None else np.asarray(star_ews)[indices])
    b=make_star('Sun',sun_params,r.ew_sun_mA if sun_ews is None else np.asarray(sun_ews)[indices])
    work=Path(workdir);work.mkdir(parents=True,exist_ok=True)
    cwd=os.getcwd();old_alias=np.__dict__.get('float',None);had_alias='float' in np.__dict__
    original_nlte=q2.abundances.nlte_triplet;corrections=[]
    def capture_nlte(*args,**kwargs):
        result=original_nlte(*args,**kwargs)
        corrections.append((np.asarray(args[3],float),np.asarray(result,float)))
        return result
    # Local compatibility for installed q2's removed numpy alias, never edit q2.
    np.float=float
    q2.abundances.nlte_triplet=capture_nlte
    try:
        os.chdir(work)
        with open(work/'q2_interface.log','w') as log,contextlib.redirect_stdout(log):
            q2.abundances.get_one(a,['OI'],b,silent=True,errors=False)
    finally:
        os.chdir(cwd)
        q2.abundances.nlte_triplet=original_nlte
        if had_alias:np.float=old_alias
        else:delattr(np,'float')
    star=np.asarray(a.OI['ab'],float);sun=np.asarray(b.OI['ab'],float)
    if len(star)!=3 or not np.isfinite(star).all() or not np.isfinite(sun).all():
        raise ValueError('q2 oxygen NLTE interpolation failed/outside supported grid; no LTE fallback')
    if len(corrections)!=2:raise RuntimeError('Unexpected q2 oxygen reference correction sequence')
    audit=pd.DataFrame({'wave_A':TRIPLET,'star_LTE':corrections[0][0],
        'star_NLTE':star,'sun_LTE':corrections[1][0],'sun_NLTE':sun})
    audit['differential_NLTE_correction']=(audit.star_NLTE-audit.sun_NLTE)-(audit.star_LTE-audit.sun_LTE)
    audit.to_csv(work/'oxygen_q2_audit.csv',index=False)
    return indices,star,sun

class FixedLineEvaluator:
    def __init__(self,ispec,raw,linemasks,ispec_dir,output_folder,solar_label,oxygen=False):
        self.ispec=ispec;self.raw=raw.reset_index(drop=True);self.root=Path(ispec_dir)
        self.folder=Path(output_folder);self.label=solar_label;self.oxygen=oxygen
        self.masks=linemasks;self.sun_params=dict(teff=5772.,logg=4.44,MH=0.,alpha=0.,vmic=1.)
        from abundance_contracts import q2_dump_path
        sf=self.root/'output/Sun'/solar_label
        for candidate in [sf/f'atmos_params_{solar_label}.dump',sf/f'atmos_params_{solar_label}_q2.dump']:
            if candidate.exists():self.sun_params.update(ispec.restore_results(str(candidate))[0]);break
        self.sun_folder=sf
        self.line_masks=[];self.star_ew_error=[];self.sun_ew_error=[]
        smp=sf/'linemasks'/f'{solar_label}_melendez2014_star_fitted_linemasks.filtered_for_abund.txt'
        if not smp.exists():smp=sf/'linemasks'/f'{solar_label}_melendez2014_star_fitted_linemasks.txt'
        sm=ispec.read_line_regions(str(smp)) if smp.exists() else None
        if sm is not None:
            # iSpec text masks round EW to 0.1 mA. The rescue audit retains full
            # precision and the matching new error for the solar-library EW.
            from local_ew_retry import apply_local_refits
            sm=apply_local_refits(sm,output_folder=str(sf),target=solar_label)
        for _,row in self.raw.iterrows():
            match=np.where((abs(linemasks['wave_A']-row.wave_A)<.003) &
                           (np.array([species_token(x) for x in linemasks['element']])==species_token(row.element)))[0]
            if len(match)!=1:raise ValueError(f'Nonunique uncertainty mask: {row.element} {row.wave_A}')
            self.line_masks.append(linemasks[match[0]])
            # A different manually adopted EW must not inherit the old fit error.
            e=float(linemasks['ew_err'][match[0]]) if np.isclose(linemasks['ew'][match[0]],row.ew_mA,atol=.01,rtol=0) else np.nan
            self.star_ew_error.append(e if np.isfinite(e) and e>0 else np.nan)
            se=np.nan
            if sm is not None:
                hit=np.where((abs(sm['wave_A']-row.wave_A)<.003)&(np.array([species_token(x) for x in sm['element']])==species_token(row.element)))[0]
                if len(hit)==1 and np.isclose(sm['ew'][hit[0]],row.ew_sun_mA,atol=.01,rtol=0):se=float(sm['ew_err'][hit[0]])
            self.sun_ew_error.append(se if np.isfinite(se) and se>0 else np.nan)
        self.line_masks=np.array(self.line_masks,dtype=linemasks.dtype)

    def evaluate(self,params,work,*,solar=False,ews=None):
        from abund_lbl_pipeline import prepare_abundance_engine
        from HFSBlends import (_ensure_moog_model_footer,_find_moog_executable,
            run_moog_blends_batch,parse_moog_blends_summary,write_batch_par_from_template,
            _assess_moog_run_quality)
        from hfs_contracts import active_line_file
        work=Path(work);work.mkdir(parents=True,exist_ok=True)
        p=self.sun_params.copy() if solar else params.copy()
        eng=prepare_abundance_engine(ispec=self.ispec,ispec_dir=str(self.root),teff=p['teff'],logg=p['logg'],mh=p['MH'],alpha=p.get('alpha',0))
        values=np.full(len(self.raw),np.nan);masks=self.line_masks.copy()
        masks['ew']=np.asarray(ews if ews is not None else self.raw.ew_sun_mA if solar else self.raw.ew_mA,float)
        hfs=self.raw.get('hfs_status',pd.Series('',index=self.raw.index)).eq('ok').to_numpy()
        oi=(self.raw.element.map(species_token).eq('O 1') & self.raw.wave_A.map(lambda x:np.min(abs(TRIPLET-x))<.05)).to_numpy() if self.oxygen else np.zeros(len(values),bool)
        for token in sorted(set(self.raw.element.map(species_token))):
            sel=(self.raw.element.map(species_token).eq(token)).to_numpy() & ~hfs & ~oi
            if not sel.any():continue
            a=self.ispec.determine_abundances(eng['atmosphere_layers'],p['teff'],p['logg'],p['MH'],p.get('alpha',0),masks[sel],eng['solar_abundances'],microturbulence_vel=p['vmic'],code='moog',verbose=0)[0]
            values[sel]=np.asarray(a)+12
        if hfs.any():
            folder=self.sun_folder if solar else self.folder
            name=self.label if solar else str(self.raw.id.iloc[0])
            hf=pd.read_csv(folder/'HFS/tables'/f'abundances_hfs_blends_{name}.csv')
            model=work/'model.in';self.ispec.write_atmosphere(eng['atmosphere_layers'],p['teff'],p['logg'],p['MH'],atmosphere_filename=str(model),code='moog')
            _ensure_moog_model_footer(model,p['MH'],p['vmic'])
            for i in np.where(hfs)[0]:
                row=self.raw.iloc[i];hit=hf[(hf.element==str(row.element).split()[0])&(abs(hf.wave_A-row.wave_A)<.003)&(hf.status=='ok')]
                if len(hit)!=1:raise ValueError(f'Missing fixed HFS line for error propagation: {row.wave_A}')
                run=folder/'HFS/runs'/Path(str(hit.iloc[0].run_dir)).name
                source=active_line_file(run)
                dest=work/f'hfs_{i}';dest.mkdir(exist_ok=True);shutil.copyfile(model,dest/'model.in')
                lines=source.read_text().splitlines();v=lines[1].split();v[6]=f'{masks["ew"][i]:.3f}'
                lines[1]=''.join(f'{float(x):10.3f}' for x in v[:7])
                (dest/'lines.in').write_text('\n'.join(lines)+'\n')
                write_batch_par_from_template(template_path=run/'batch.par',out_path=dest/'batch.par',lines_in='lines.in',model_in='model.in',standard_out='moog.std',summary_out='moog.sum',window_A=2.,blen_species=float(hit.iloc[0].species))
                stdout,stderr,rc=run_moog_blends_batch(dest,moog_bin=_find_moog_executable(),timeout_sec=90)
                parsed=parse_moog_blends_summary(dest/'moog.sum',expected_element=str(row.element).split()[0],expected_wave_A=float(row.wave_A),expected_species=float(hit.iloc[0].species))
                qc=_assess_moog_run_quality(run_dir=dest,stdout=stdout,stderr=stderr,expected_wave_A=float(row.wave_A))
                if rc or parsed['status']!='ok' or qc.get('non_converged',False):raise RuntimeError(f'HFS uncertainty perturbation failed: {dest}: {parsed}')
                values[i]=parsed['logeps_hfs']
        if oi.any():
            ix,a,b=q2_oxygen(self.raw,params,self.sun_params,work/'oxygen',
                star_ews=None if solar else masks['ew'],sun_ews=masks['ew'] if solar else None)
            values[ix]=b if solar else a
        if not np.isfinite(values).all():raise ValueError(f'Nonfinite abundance perturbation: {work}')
        return values

def finalize_element_outputs(*,ispec,target,linemasks,ispec_dir,output_folder,solar_label,
                             summary_path,params,parameter_errors,pipeline_options):
    """Run after HFS: same successful lines, same estimator, optional O reroute."""
    summary_path=Path(summary_path);raw_path=summary_path.with_name(summary_path.stem+'_raw.csv')
    raw=pd.read_csv(raw_path);summary=pd.read_csv(summary_path)
    if 'abundance_method' not in raw:raw['abundance_method']='ispec_MOOG_LTE'
    if 'hfs_status' in raw:raw.loc[raw.hfs_status.eq('ok'),'abundance_method']='MOOG_blends_HFS_LTE'
    master=pd.read_csv(Path(ispec_dir)/'input/solar_lines_instru.csv')
    master=master[master.solar_label==solar_label].copy();master['token']=master.element.map(species_token)
    raw['ew_sun_mA']=np.nan
    for i,row in raw.iterrows():
        m=master[(master.token==species_token(row.element))&(abs(master.wave_A-row.wave_A)<.003)]
        if len(m)!=1:raise ValueError(f'Uncertainty solar pairing is nonunique: {row.wave_A}')
        raw.at[i,'ew_sun_mA']=float(m.ew_mA.iloc[0])
    use=raw.get('used_in_summary',pd.Series(True,index=raw.index)).astype(str).str.lower().isin(['true','1'])
    use &= np.isfinite(pd.to_numeric(raw['d[X/H]'],errors='coerce'))
    selected=raw.loc[use].reset_index().rename(columns={'index':'raw_index'})
    work=Path(output_folder)/'uncertainties';work.mkdir(exist_ok=True)
    oxygen=bool(pipeline_options.get('run_O_triple',False))
    if not oxygen and raw.abundance_method.eq('q2_O777_NLTE').any():
        raise ValueError('Turning oxygen NLTE off requires rerunning the normal 4a abundance step first')
    ev=FixedLineEvaluator(ispec,selected,linemasks,ispec_dir,output_folder,solar_label,oxygen)
    if oxygen:
        if 'oxygen_original_ispec_logeps' not in raw:raw['oxygen_original_ispec_logeps']=np.nan
        ix,a,b=q2_oxygen(selected,params,ev.sun_params,work/'oxygen_nominal')
        oxygen_audit=pd.read_csv(work/'oxygen_nominal/oxygen_q2_audit.csv')
        for j,sa,su in zip(ix,a,b):
            ri=int(selected.loc[j,'raw_index'])
            if pd.isna(raw.at[ri,'oxygen_original_ispec_logeps']):raw.at[ri,'oxygen_original_ispec_logeps']=raw.at[ri,'logeps_star']
            raw.at[ri,'logeps_star']=sa;raw.at[ri,'logeps_sun']=su
            raw.at[ri,'d[X/H]']=sa-su;raw.at[ri,'dlogeps']=sa-su
            raw.at[ri,'abundance_method']='q2_O777_NLTE'
            oi=oxygen_audit.iloc[list(ix).index(j)]
            for src,dst in [('star_LTE','oxygen_q2_star_LTE'),('sun_LTE','oxygen_q2_sun_LTE'),
                            ('differential_NLTE_correction','oxygen_differential_NLTE_correction')]:
                raw.at[ri,dst]=oi[src]
            # Grevesse absolute fields must describe the same recomputed Sun.
            gref=float(selected.loc[j,'logeps_sun']-selected.loc[j,'[X/H]_sun_Grevesse'])
            raw.at[ri,'[X/H]_sun_Grevesse']=su-gref;raw.at[ri,'[X/H]_star_Grevesse']=sa-gref
        selected=raw.loc[use].reset_index().rename(columns={'index':'raw_index'});ev.raw=selected
    shifts={}
    for key in PARAMETERS:
        sigma=float(parameter_errors.get(key,np.nan))
        if not np.isfinite(sigma) or sigma<0:shifts[key]=np.full(len(selected),np.nan);continue
        if sigma==0:shifts[key]=np.zeros(len(selected));continue
        plus=params.copy();minus=params.copy();plus[key]+=sigma;minus[key]-=sigma
        if key=='vmic' and minus[key]<0:raise ValueError('Symmetric vt uncertainty crosses zero; supply a suitable posterior/error model')
        ap=ev.evaluate(plus,work/f'{key}_plus');am=ev.evaluate(minus,work/f'{key}_minus')
        shifts[key]=(ap-am)/2
        print(f'[errors] {target}: {key} +/- {sigma} complete',flush=True)
    # Independent EW-noise floor: vary each line by its own fitted EW sigma.
    noise=[]
    for solar,err,col in [(False,np.asarray(ev.star_ew_error), 'ew_mA'),(True,np.asarray(ev.sun_ew_error),'ew_sun_mA')]:
        ew=selected[col].to_numpy(float);known=np.isfinite(err)&(err>0)&(ew>err)
        if known.any():
            delta=np.where(known,err,0)
            ap=ev.evaluate(params,work/('solar_EW_plus' if solar else 'star_EW_plus'),solar=solar,ews=ew+delta)
            am=ev.evaluate(params,work/('solar_EW_minus' if solar else 'star_EW_minus'),solar=solar,ews=ew-delta)
            noise.append(np.where(known,abs(ap-am)/2,np.nan))
        else:noise.append(np.full(len(selected),np.nan))
    line_noise=np.hypot(noise[0],noise[1])
    audit=selected[['element','wave_A','raw_index']].copy();audit['ew_noise_dex']=line_noise
    for k,v in shifts.items():audit['sensitivity_'+k]=v
    audit.to_csv(work/'line_sensitivities.csv',index=False)
    for i,row in summary.iterrows():
        idx=np.where(selected.element.map(species_token).eq(species_token(row.element)))[0]
        if not len(idx):continue
        vals=selected.iloc[idx]['d[X/H]'].to_numpy(float);n=len(idx)
        scatter=np.std(vals,ddof=1) if n>1 else np.nan
        empirical=scatter/np.sqrt(n) if n>1 else np.nan
        measured=np.sqrt(np.sum(line_noise[idx]**2))/n if np.isfinite(line_noise[idx]).all() else np.nan
        # The empirical scatter already contains measurement noise; avoid adding
        # that noise twice. Use the larger available estimate as a floor.
        finite=[v for v in [empirical,measured] if np.isfinite(v)]
        statistical=max(finite) if finite else np.nan
        ds=[]
        for k in PARAMETERS:
            value=np.mean(shifts[k][idx]);summary.at[i,'sensitivity_'+k]=value;ds.append(value)
        pe=np.linalg.norm(ds) if np.isfinite(ds).all() else np.nan
        summary.at[i,'[X/H]']=np.mean(vals);summary.at[i,'n_lines']=n
        summary.at[i,'std_[X/H]']=scatter;summary.at[i,'sem_[X/H]']=empirical
        summary.at[i,'stat_err_[X/H]']=statistical;summary.at[i,'ew_err_mean_[X/H]']=measured
        summary.at[i,'param_err_[X/H]']=pe
        summary.at[i,'total_err_[X/H]']=np.hypot(statistical,pe)
        summary.at[i,'uncertainty_kind']='parameter_diagonal_plus_statistical_floor'
    raw.to_csv(raw_path,index=False);summary.to_csv(summary_path,index=False)
    (work/'assumptions.json').write_text(json.dumps({'parameters':params,'parameter_errors':parameter_errors,
        'solar_atmosphere_fixed':ev.sun_params,'parameter_covariance':'diagonal approximation; not a full covariance matrix',
        'statistical':'max(empirical SEM, independently propagated star+Sun EW noise), where available',
        'run_O_triple':oxygen,'fixed_successful_line_set':True},indent=2))
    return {'summary_path':str(summary_path),'uncertainty_folder':str(work)}
