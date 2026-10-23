"""Regression tests for scientific data contracts, with real reference fixtures."""
import unittest,tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from abundance_contracts import q2_dump_path,strict_quadrature,combine_species_statistics
from hfs_contracts import no_hfs_logeps,component_signature,apply_pair_quality

ROOT=Path(__file__).resolve().parent.parent

class PipelineContracts(unittest.TestCase):
    def test_q2_path_is_idempotent(self):
        for p in ['atmos_X.dump','atmos_X_q2.dump','atmos_X_q2_q2.dump']:
            out=q2_dump_path(p)
            self.assertEqual(out,'atmos_X.dump');self.assertEqual(q2_dump_path(out),out)

    def test_unknown_uncertainty_is_not_zero(self):
        self.assertTrue(np.isnan(strict_quadrature(np.nan,.02)))
        self.assertAlmostEqual(float(strict_quadrature(.03,.04)),.05)

    def test_single_line_retains_unknown_sem(self):
        g=pd.DataFrame({'[X/H]':[.1],'std_[X/H]':[np.nan],'n_lines':[1]})
        self.assertTrue(np.isnan(combine_species_statistics(g)['std_[X/H]']))

    def test_within_species_errors_not_between_species_scatter(self):
        g=pd.DataFrame({'[X/H]':[.1,.1],'std_[X/H]':[.1,.2],'n_lines':[4,4]})
        self.assertAlmostEqual(combine_species_statistics(g)['std_[X/H]'],np.sqrt((3*.1**2+3*.2**2)/7/8))

    def test_published_single_species_error_is_preserved(self):
        g=pd.DataFrame({'[X/H]':[.1],'std_[X/H]':[.02],'n_lines':[1],'error_kind':['reported']})
        self.assertAlmostEqual(combine_species_statistics(g)['std_[X/H]'],.02)

    def test_bedell_matching_excludes_ch_and_scI(self):
        from abund_plot import build_ref_for_star_bedell2018_table2
        td=tempfile.TemporaryDirectory();self.addCleanup(td.cleanup)
        p=str(Path(td.name)/'reference.csv')
        # Synthetic species fixture; no external literature database required.
        pd.DataFrame([{'Star':'HIP42333','[CI/H]':-.026,'e_[CI/H]':.01,
                       '[CH/H]':.038,'e_[CH/H]':.01,'[ScII/H]':.087,
                       'e_[ScII/H]':.01,'[ScI/H]':.2,'e_[ScI/H]':.02}]).to_csv(p,index=False)
        x,_=build_ref_for_star_bedell2018_table2('HIP42333',p,feh_override=.132,match_target_species=True,target_species={'C 1','Sc 2'})
        self.assertAlmostEqual(x.set_index('Element').loc['C','[X/H]'],-.026)
        self.assertAlmostEqual(x.set_index('Element').loc['Sc','[X/H]'],.087)
        y,_=build_ref_for_star_bedell2018_table2('HIP42333',p,feh_override=.132)
        self.assertAlmostEqual(y.set_index('Element').loc['C','[X/H]'],.006)

    def test_run_context_reference_switch(self):
        from run_context import load_run_context,REF_MATCH_SPECIES
        self.assertTrue(REF_MATCH_SPECIES)
        self.assertTrue(load_run_context(target='HD73350')['PIPELINE_OPTIONS']['ref_match_species'])
        ctx=load_run_context(target='HD73350',run_context_override={'ref_match_species':False})
        self.assertFalse(ctx['PIPELINE_OPTIONS']['ref_match_species'])

    def test_dec_sign_and_barycentric_interface(self):
        import DataPreProcess as d
        dec=d.parse_dec_to_dms('-06:48:27.9');self.assertEqual(dec[0],-6)
        tiny=d.parse_dec_to_dms('-00:30:00');self.assertTrue(np.signbit(tiny[0]))
        class Fake:
            def calculate_barycentric_velocity_correction(self,time,coords):
                self.coords=coords;return 1.
            def correct_velocity(self,s,v):return s
        fake=Fake();previous=d.ispec;d.ispec=fake
        try:d.calculate_barycentric_velocity(None,[2026,2,20,6,0,0],[8,0,0],dec)
        finally:d.ispec=previous
        self.assertAlmostEqual(sum([fake.coords[3],fake.coords[4]/60,fake.coords[5]/3600]),-(6+48/60+27.9/3600))

    def test_legacy_nohfs_is_recovered_without_mixed_baseline(self):
        row={'logeps_star':5.358,'logeps_sun':5.35,'hfs':.188}
        self.assertAlmostEqual(no_hfs_logeps(row),5.538)

    def test_component_signature_ignores_ew_but_not_physics(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'a.in';q=Path(td)/'b.in'
            p.write_text('header\n6000 23 1 -1 0 0 30\n-6000.01 23 1 -2 0 0 0\n')
            q.write_text('header\n6000 23 1 -1 0 0 40\n-6000.01 23 1 -2 0 0 0\n')
            self.assertEqual(component_signature(p,2),component_signature(q,2))
            q.write_text(q.read_text().replace('1 -2','1 -3'))
            self.assertNotEqual(component_signature(p,2),component_signature(q,2))

    def test_large_hfs_correction_does_not_automatically_reject(self):
        f=pd.DataFrame([dict(element='V',ion='1',wave_A=6000.,status='ok',component_signature='a',solar_component_signature='a',solar_pair_status='ok',**{'[X/H]_HFS':.05,'[X/H]_noHFS_paired':.25})])
        out=apply_pair_quality(f);self.assertTrue(out.used_in_summary.iloc[0])
        f.loc[0,'solar_component_signature']='b'
        self.assertFalse(apply_pair_quality(f).used_in_summary.iloc[0])

    def test_existing_footer_is_updated(self):
        from HFSBlends import _ensure_moog_model_footer
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'model.in';p.write_text('layers\n  1.00\nNATOMS=   0 0.11\nNMOL 0\n')
            _ensure_moog_model_footer(p,.097,.87)
            self.assertIn('0.87\nNATOMS=   0 0.10',p.read_text())

    def test_tc_merge_is_idempotent(self):
        from abund_plot import add_tc_to
        x=pd.DataFrame({'Element':['C'],'Tc_K':[123.]})
        tc=pd.DataFrame({'Element':['C'],'Tc_K':[40.]})
        out=add_tc_to(add_tc_to(x,tc),tc)
        self.assertEqual(list(out.columns),['Element','Tc_K'])
        self.assertEqual(out.Tc_K.iloc[0],40.)

    def test_line_review_cannot_override_physical_failure(self):
        rows=[dict(element='V',ion='1',wave_A=5000.+i,status='ok',component_signature='a',
                   solar_component_signature='a',solar_pair_status='ok',**{'[X/H]_HFS':v})
              for i,v in enumerate([0.,.01,.02,.03,1.])]
        f=pd.DataFrame(rows);out=apply_pair_quality(f)
        self.assertEqual(out.status.iloc[-1],'needs_line_review')
        decisions={('V',5004.):'accept'}
        self.assertTrue(apply_pair_quality(f,review_decisions=decisions).used_in_summary.iloc[-1])
        f.loc[4,'solar_pair_status']='missing'
        self.assertFalse(apply_pair_quality(f,review_decisions=decisions).used_in_summary.iloc[-1])

    def test_profile_failure_is_excluded(self):
        f=pd.DataFrame([dict(element='Cu',ion='1',wave_A=5100.,status='ok',
            component_signature='a',solar_component_signature='a',solar_pair_status='ok',
            profile_quality='invalid_fit',**{'[X/H]_HFS':.1})])
        self.assertFalse(apply_pair_quality(f).used_in_summary.iloc[0])

    def test_real_q2_save_and_reload_uses_same_path(self):
        import ispec
        from atmos_lbl_pipeline import save_q2_dump,export_q2_dump_csv
        row=pd.Series(dict(teff=5772.,logg=4.44,feh=.05,vt=.87))
        with tempfile.TemporaryDirectory() as td:
            path=str(Path(td)/'atmos_X_q2_q2.dump')
            out=save_q2_dump(ispec=ispec,output_atmos_params_dumpfile_path=path,
                            solution_row=row,linemasks_fe=None,target='X',solution_csv='solution.csv')
            save_q2_dump(ispec=ispec,output_atmos_params_dumpfile_path=out['dump_file_q2'],
                         solution_row=row,linemasks_fe=None,target='X',solution_csv='solution.csv')
            export_q2_dump_csv(ispec=ispec,target='X',output_atmos_params_dumpfile_path=path)
            self.assertTrue((Path(td)/'atmos_X.csv').exists())
            self.assertEqual(len(list(Path(td).glob('*.dump'))),1)
            self.assertAlmostEqual(ispec.restore_results(out['dump_file_q2'])[0]['vmic'],.87)

    def test_oxygen_default_and_override(self):
        from run_context import load_run_context
        self.assertFalse(load_run_context(target='HD73350')['run_O_triple'])
        self.assertTrue(load_run_context(target='HD73350',run_context_override={'run_O_triple':True})['run_O_triple'])

    def test_shared_parameter_error_does_not_average_away(self):
        g=pd.DataFrame({'[X/H]':[.1,.1],'n_lines':[2,2],
            'std_[X/H]':[.02,.02],'stat_err_[X/H]':[.01,.01],
            'sensitivity_teff':[.1,.1],'sensitivity_logg':[0.,0.],
            'sensitivity_MH':[0.,0.],'sensitivity_vmic':[0.,0.]})
        out=combine_species_statistics(g)
        self.assertAlmostEqual(out['param_err_[X/H]'],.1)
        self.assertAlmostEqual(out['std_[X/H]'],np.hypot(.1,.02/np.sqrt(6)))

    def test_xfe_cancels_shared_parameter_response(self):
        from abundance_contracts import xfe_uncertainty
        f=pd.DataFrame({'[X/H]':[.1,.2],'std_[X/H]':[.1,.1],
            'stat_err_[X/H]':[.01,.02],'sensitivity_teff':[.1,.1],
            'sensitivity_logg':[0.,0.],'sensitivity_MH':[0.,0.],'sensitivity_vmic':[0.,0.]})
        result=xfe_uncertainty(f,f.iloc[0]);self.assertAlmostEqual(result[1],np.hypot(.01,.02))

    def test_oxygen_requires_all_three_lines(self):
        from element_uncertainties import q2_oxygen
        with tempfile.TemporaryDirectory() as td:
            r=pd.DataFrame({'element':['O I'],'wave_A':[7771.94]})
            with self.assertRaisesRegex(ValueError,'each 777 nm triplet'):
                q2_oxygen(r,{}, {}, td)

    def test_equal_line_mean_and_scatter_match_flattened_lines(self):
        groups=[np.array([.1,.2,.3]),np.array([.8])]
        g=pd.DataFrame({'[X/H]':[a.mean() for a in groups],'n_lines':[len(a) for a in groups],
                        'std_[X/H]':[np.std(a,ddof=1) if len(a)>1 else np.nan for a in groups]})
        result=combine_species_statistics(g);flat=np.concatenate(groups)
        self.assertAlmostEqual(result['[X/H]'],np.mean(flat))
        self.assertAlmostEqual(result['line_scatter_[X/H]'],np.std(flat,ddof=1))
        self.assertAlmostEqual(result['sem_[X/H]'],np.std(flat,ddof=1)/np.sqrt(len(flat)))
        self.assertEqual(result.n_lines,4)

    def test_two_single_line_groups_have_a_defined_pooled_sem(self):
        g=pd.DataFrame({'[X/H]':[.1,.3],'n_lines':[1,1],'std_[X/H]':[np.nan,np.nan]})
        out=combine_species_statistics(g)
        self.assertAlmostEqual(out['sem_[X/H]'],.1)

    def test_parameter_responses_use_line_counts(self):
        g=pd.DataFrame({'[X/H]':[.1,.3],'n_lines':[3,1],'std_[X/H]':[.01,np.nan],
            'stat_err_[X/H]':[.01,.02],'sensitivity_teff':[.02,.1],
            'sensitivity_logg':[0.,0.],'sensitivity_MH':[0.,0.],'sensitivity_vmic':[0.,0.]})
        out=combine_species_statistics(g)
        self.assertAlmostEqual(out['[X/H]'],.15)
        self.assertAlmostEqual(out.sensitivity_teff,.04)
        self.assertAlmostEqual(out['param_err_[X/H]'],.04)

    def test_missing_counts_are_not_invented(self):
        g=pd.DataFrame({'[X/H]':[.1],'std_[X/H]':[.02]})
        with self.assertRaisesRegex(ValueError,'integer n_lines'):combine_species_statistics(g)

    def test_zero_count_group_is_not_used(self):
        g=pd.DataFrame({'[X/H]':[99.,.2],'n_lines':[0,2],'std_[X/H]':[np.nan,.01]})
        out=combine_species_statistics(g)
        self.assertAlmostEqual(out['[X/H]'],.2);self.assertEqual(out.n_lines,2)

    def test_hfs_fe_reference_is_the_same_pooled_fe(self):
        from HFSBlends import _compute_fe_reference_from_ew_csv
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'iron.csv'
            pd.DataFrame({'element':['Fe 1','Fe 2'],'n_lines':[3,1],
                          '[X/H]':[.1,.3],'std_[X/H]':[.02,np.nan]}).to_csv(p,index=False)
            a,b=_compute_fe_reference_from_ew_csv(p)
            self.assertAlmostEqual(a,.15);self.assertAlmostEqual(b,.15)

    def test_measurement_errors_use_same_unequal_line_weights(self):
        g=pd.DataFrame({'[X/H]':[.1,.1],'n_lines':[3,1],
                        'std_[X/H]':[.01,np.nan],'ew_err_mean_[X/H]':[.02,.1]})
        out=combine_species_statistics(g)
        self.assertAlmostEqual(out['stat_err_[X/H]'],np.hypot(.75*.02,.25*.1))

if __name__=='__main__':unittest.main()
