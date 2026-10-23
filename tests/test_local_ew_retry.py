"""Local-continuum science and QA lifecycle regression tests (isolated outputs)."""
import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

import local_ew_retry as retry
import line_review_ui as ui


def fixtures():
    wave = np.arange(499.65, 500.351, .001)
    spectrum = np.zeros(len(wave), dtype=[('waveobs', float), ('flux', float), ('err', float)])
    spectrum['waveobs'], spectrum['err'] = wave, .002
    spectrum['flux'] = ((.98 + .01*(wave-500)/.35) *
        (1-.075*np.exp(-.5*((wave-500)/.006)**2)) + np.random.default_rng(4).normal(0,.002,len(wave)))
    dtype = [('element','U8'),('wave_nm',float),('wave_A',float),('mu',float),
        ('sig',float),('ew',float),('wave_base',float),('wave_top',float),('ew_err',float),
        ('ewr',float),('A',float),('baseline',float),('rms',float),('loggf',float)]
    lines = np.array([('C 1',500.,5000.,500.,.006,12.,499.978,500.022,4.,0.,-.1,1.,.02,-1.)], dtype=dtype)
    return spectrum, lines


def item(element='C 1', wave=500., reason='other'):
    return ui.ReviewCandidate('', f'{element}_1_{wave*10:.2f}.png', element, wave*10, wave,
                              reason=reason, note='test reason')


class FakeIspec:
    def __init__(self):
        self.tables={}

    def write_line_regions(self, arr, path, **kwargs):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame.from_records(arr).to_csv(path, sep='\t', index=False)
        self.tables[str(path)]=arr.copy()

    def read_line_regions(self,path):
        return self.tables[str(path)].copy()

    def reset_fitted_data_fields(self,arr):
        return arr.copy()

    def write_atomic_linelist(self,arr,path):
        self.write_line_regions(arr,path)


class ScriptedReviewer:
    def __init__(self, rounds):
        self.rounds = list(rounds)
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if not kwargs['candidates_override']:
            return dict(status='empty', items=[])
        choices = self.rounds.pop(0)
        if choices == 'cancel':
            return dict(status='cancelled', items=kwargs['candidates_override'])
        items = [ui._review_decision(x, choices.get(x.element, 'other' if kwargs.get('second_pass') else 'good'),
                 x.note, second_pass=kwargs.get('second_pass', False), final=True)
                 for x in kwargs['candidates_override']]
        # Use real persistence helpers, without opening a human review UI.
        ui._upsert_deleted_table(kwargs['deleted_tsv_path'],
            ui._build_new_deleted_rows(items, pd.DataFrame()), reviewed_items=items)
        ui._sync_images_for_items(items, kwargs['figure_dir'],
            kwargs['reason_options'] + [ui.FAILED_BOTH, ui.LOCAL_RENORMALIZE])
        return dict(status='saved', items=items)


class RetryTests(unittest.TestCase):
    def setup_run(self, folder, lines=None):
        spectrum, original = fixtures()
        lines = original if lines is None else lines
        figures = Path(folder)/'figs_ele_GaussianFits'
        for r in lines:
            p = figures/str(r['element']).replace(' ','_')/f"{r['element']}_1_{r['wave_A']:.2f}.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            # No need to render the original fixture image; refit replaces it.
            p.write_bytes(b'fixture')
        return dict(ispec=FakeIspec(), star_spectrum=spectrum, linemasks=lines,
                    output_folder=folder, target='Test', figure_dir=str(figures),
                    pipeline_options={'line_review_ui_mode':'browser'})

    def test_switch_default_and_override(self):
        from run_context import run_local_ew_retry, local_ew_retry_enabled, load_run_context
        self.assertTrue(run_local_ew_retry)
        self.assertTrue(local_ew_retry_enabled({}))  # also the Sun path
        self.assertFalse(local_ew_retry_enabled({'run_local_ew_retry':False}))
        self.assertTrue(load_run_context('HD73350')['run_local_ew_retry'])
        self.assertFalse(load_run_context('HD73350',run_context_override={'run_local_ew_retry':False})['run_local_ew_retry'])

    def test_manual_and_automatic_selection(self):
        _, lines = fixtures()
        x = item(reason=ui.LOCAL_RENORMALIZE)
        self.assertEqual(retry.choose_retry_lines([x],lines)[0][1], 'manual')
        x.reason = 'continuum_low'
        lines['ew']=25.
        self.assertEqual(retry.choose_retry_lines([x],lines)[0][1], 'automatic_has_ew_above_20_rejected')
        x.reason = 'good'
        self.assertFalse(retry.choose_retry_lines([x],lines))

    def test_only_finite_ew_strictly_above_20_triggers_automatic_retry(self):
        _, lines = fixtures()
        for ew in [0.,12.,20.,np.nan,np.inf,-1.]:
            lines['ew']=ew
            self.assertFalse(retry.choose_retry_lines([item()],lines))
        for ew in [20.001,25.]:
            lines['ew']=ew
            self.assertTrue(retry.choose_retry_lines([item()],lines))
        lines['ew']=12.
        self.assertTrue(retry.choose_retry_lines([item(reason=ui.LOCAL_RENORMALIZE)],lines))

    def test_mixed_ews_need_only_one_line_above_20(self):
        _, line=fixtures()
        lines=np.repeat(line,3);lines['wave_nm']=[500.,500.1,500.2]
        items=[item(wave=w) for w in lines['wave_nm']]
        for ews in [[12.,25.,18.],[np.nan,25.,18.]]:
            lines['ew']=ews
            self.assertEqual(len(retry.choose_retry_lines(items,lines)),3)
        lines['ew']=[12.,20.,18.]
        self.assertFalse(retry.choose_retry_lines(items,lines))

    def test_element_groups_and_manual_suppression(self):
        _, l = fixtures()
        lines=np.repeat(l,2);lines['element']=['Fe 1','Fe 2'];lines['wave_nm']=[500.,500.1]
        lines['ew']=[12.,25.]
        items=[item('Fe 1',500.,ui.LOCAL_RENORMALIZE),item('Fe 2',500.1)]
        chosen=retry.choose_retry_lines(items,lines)
        self.assertEqual(len(chosen),1)
        items[0].reason='other'
        self.assertEqual(len(retry.choose_retry_lines(items,lines)),2)
        # A usable Fe II line outside this page prevents an automatic Fe I rescue.
        self.assertFalse(retry.choose_retry_lines(items[:1],lines))

    def test_no_third_attempt(self):
        _, lines=fixtures();key=retry.line_key('C 1',500.)
        self.assertFalse(retry.choose_retry_lines([item(reason=ui.LOCAL_RENORMALIZE)],lines,attempted=[key]))

    def test_recovery_and_uncertainty(self):
        s,l=fixtures();old=s.copy();r=retry.refit_local_line(s,l[0])
        truth=.075*.006*np.sqrt(2*np.pi)*10000
        self.assertLess(abs(r['fit_values']['ew']-truth),3*r['fit_values']['ew_err'])
        self.assertAlmostEqual(r['fit_parameters'][0],.98,delta=.001)
        self.assertGreater(r['fit_values']['ew_err'],0)
        self.assertTrue(np.array_equal(s,old))
        noisier=s.copy();noisier['err']*=2
        self.assertGreater(retry.refit_local_line(noisier,l[0])['fit_values']['ew_err'],r['fit_values']['ew_err'])
        self.assertNotEqual(r['covariance'][0][2],0) # continuum-depth covariance retained

    def test_missing_pixel_errors_estimated_and_flagged(self):
        s,l=fixtures();s['err']=0
        r=retry.refit_local_line(s,l[0]);self.assertTrue(r['noise_estimated'])
        self.assertGreater(r['fit_values']['ew_err'],0)

    def test_insufficient_anchors_rejected(self):
        s,l=fixtures()
        with self.assertRaisesRegex(ValueError,'anchors'):
            retry.refit_local_line(s,l[0],continuum_windows=[(499.8,499.9)])

    def test_manual_two_round_accept_and_replay(self):
        with tempfile.TemporaryDirectory() as td:
            args=self.setup_run(td)
            modified=Path(td)/'linemasks'/'Lines_EW_modified.tsv';modified.parent.mkdir()
            pd.DataFrame([dict(element='C 1',wave_nm=500.,ew_mA_new=15.),dict(element='O 1',wave_nm=777.,ew_mA_new=30.)]).to_csv(modified,sep='\t',index=False)
            r=ScriptedReviewer([{'C 1':ui.LOCAL_RENORMALIZE},{'C 1':ui.NEW_EW_OK}])
            result=retry.run_review_with_local_retry(**args,reviewer=r)
            self.assertEqual(result['accepted'],1)
            self.assertEqual(r.calls[0]['reason_options'][-1],ui.LOCAL_RENORMALIZE)
            self.assertNotIn(ui.LOCAL_RENORMALIZE,r.calls[1]['reason_options'])
            self.assertNotIn('good',r.calls[1]['reason_options'])
            self.assertTrue(r.calls[1]['second_pass'])
            self.assertEqual(len(pd.read_csv(Path(td)/'linemasks'/'Lines_deleted.tsv',sep='\t')),0)
            replay=retry.apply_local_refits(args['linemasks'],output_folder=td,target='Test',star_spectrum=args['star_spectrum'])
            self.assertAlmostEqual(replay['ew'][0],11.26,delta=.1)
            self.assertLess(replay['ew_err'][0],.2)
            self.assertAlmostEqual(replay['ewr'][0],np.log10(replay['ew'][0]/10000/500))
            self.assertEqual(list(pd.read_csv(modified,sep='\t').element),['O 1'])
            # Full abundance filtering consumes the new EW and its error, not just a new PNG.
            from abund_lbl_pipeline import apply_lineqa_for_abundance
            actual=apply_lineqa_for_abundance(ispec=FakeIspec(),linemasks=args['linemasks'],output_folder=td,
                linemask_output_folder=str(Path(td)/'linemasks'),target='Test',pipeline_options={})
            self.assertAlmostEqual(actual['ew'][0],replay['ew'][0])
            self.assertAlmostEqual(actual['ew_err'][0],replay['ew_err'][0])

    def test_automatic_second_failure_is_terminal(self):
        with tempfile.TemporaryDirectory() as td:
            args=self.setup_run(td)
            args['linemasks']['ew']=25.
            r=ScriptedReviewer([{'C 1':'continuum_low'},{'C 1':'crowded_region'}])
            result=retry.run_review_with_local_retry(**args,reviewer=r)
            self.assertEqual(result['accepted'],0)
            deleted=pd.read_csv(Path(td)/'linemasks'/'Lines_deleted.tsv',sep='\t')
            self.assertEqual(deleted.qa_reason.iloc[0],ui.FAILED_BOTH)
            state=retry._read_state(td,'Test')['lines']['C 1|500.0000']
            self.assertEqual(state['trigger'],'automatic_has_ew_above_20_rejected')
            r2=ScriptedReviewer([])
            retry.run_review_with_local_retry(**args,reviewer=r2)
            self.assertFalse(r2.calls[0]['candidates_override'])

    def test_sun_second_measurement_does_not_overwrite_local_ew(self):
        with tempfile.TemporaryDirectory() as td:
            args=self.setup_run(td)
            retry.run_review_with_local_retry(**args,reviewer=ScriptedReviewer([
                {'C 1':ui.LOCAL_RENORMALIZE},{'C 1':ui.NEW_EW_OK}]))
            lm_path=Path(td)/'linemasks'/'Test_melendez2014_star_fitted_linemasks.txt'
            # 0_Sun calls find_linemasks again after the Fe QA.
            args['ispec'].write_line_regions(args['linemasks'],str(lm_path),extended=True)
            from atmos_lbl_pipeline import fast_apply_lineqa_to_linemasks
            actual=fast_apply_lineqa_to_linemasks(ispec=args['ispec'],target='Test',output_folder=td,
                star_spectrum=args['star_spectrum'],star_linemasks=args['linemasks'])['star_linemasks']
            self.assertAlmostEqual(actual['ew'][0],11.26,delta=.2)
            self.assertLess(actual['ew_err'][0],.2)

    def test_failure_survives_regenerated_png_and_cleared_deletion_table(self):
        with tempfile.TemporaryDirectory() as td:
            args=self.setup_run(td)
            retry.run_review_with_local_retry(**args,reviewer=ScriptedReviewer([
                {'C 1':ui.LOCAL_RENORMALIZE},{'C 1':'other'}]))
            (Path(td)/'linemasks'/'Lines_deleted.tsv').unlink()
            args=self.setup_run(td) # regenerate a root/good PNG
            retry.run_review_with_local_retry(**args,reviewer=ScriptedReviewer([]))
            deleted=pd.read_csv(Path(td)/'linemasks'/'Lines_deleted.tsv',sep='\t')
            self.assertEqual(deleted.qa_reason.iloc[0],ui.FAILED_BOTH)

    def test_solar_error_replayed_despite_text_mask_rounding(self):
        with tempfile.TemporaryDirectory() as td:
            solar=Path(td)/'output'/'Sun'/'Test'
            args=self.setup_run(str(solar))
            retry.run_review_with_local_retry(**args,reviewer=ScriptedReviewer([
                {'C 1':ui.LOCAL_RENORMALIZE},{'C 1':ui.NEW_EW_OK}]))
            actual=retry.apply_local_refits(args['linemasks'],output_folder=str(solar),target='Test')
            rounded=actual.copy();rounded['ew']=np.round(rounded['ew'],1)
            path=solar/'linemasks'/'Test_melendez2014_star_fitted_linemasks.filtered_for_abund.txt'
            args['ispec'].write_line_regions(rounded,str(path))
            raw=pd.DataFrame([dict(element='C I',wave_A=5000.,ew_mA=actual['ew'][0],ew_sun_mA=actual['ew'][0])])
            from element_uncertainties import FixedLineEvaluator
            ev=FixedLineEvaluator(args['ispec'],raw,actual,td,str(Path(td)/'target'),'Test')
            self.assertTrue(np.isfinite(ev.sun_ew_error[0]))
            self.assertAlmostEqual(ev.sun_ew_error[0],actual['ew_err'][0])

    def test_fit_failure_cannot_accept(self):
        with tempfile.TemporaryDirectory() as td:
            args=self.setup_run(td)
            args['pipeline_options']['local_ew_continuum_windows']={'C 1|500.0000':[]}
            r=ScriptedReviewer([{'C 1':ui.LOCAL_RENORMALIZE},{'C 1':ui.NEW_EW_OK}])
            result=retry.run_review_with_local_retry(**args,reviewer=r)
            self.assertEqual(result['accepted'],0)
            self.assertTrue(r.calls[1]['candidates_override'][0].fit_failed)
            self.assertEqual(retry._read_state(td,'Test')['lines']['C 1|500.0000']['status'],'fit_failed')

    def test_switch_off_and_external_ew_never_refit(self):
        for extra in [{'run_local_ew_retry':False},{'input_has_external_ews':True}]:
            with self.subTest(extra=extra),tempfile.TemporaryDirectory() as td:
                args=self.setup_run(td);args['pipeline_options'].update(extra)
                r=ScriptedReviewer([{'C 1':'other'}])
                with patch.object(retry,'refit_local_line') as fit:
                    retry.run_review_with_local_retry(**args,reviewer=r)
                    fit.assert_not_called()
                self.assertNotIn(ui.LOCAL_RENORMALIZE,r.calls[0]['reason_options'])

    def test_cancel_and_resume_without_third_fit(self):
        with tempfile.TemporaryDirectory() as td:
            args=self.setup_run(td)
            r=ScriptedReviewer([{'C 1':ui.LOCAL_RENORMALIZE},'cancel'])
            with self.assertRaisesRegex(RuntimeError,'Second QA'):
                retry.run_review_with_local_retry(**args,reviewer=r)
            self.assertEqual(retry._read_state(td,'Test')['lines']['C 1|500.0000']['status'],'pending')
            with patch.object(retry,'refit_local_line',side_effect=AssertionError('Should reuse pending fit')):
                result=retry.run_review_with_local_retry(**args,reviewer=ScriptedReviewer([{'C 1':ui.NEW_EW_OK}]))
            self.assertEqual(result['accepted'],1)

    def test_changed_spectrum_not_replayed(self):
        with tempfile.TemporaryDirectory() as td:
            args=self.setup_run(td)
            retry.run_review_with_local_retry(**args,reviewer=ScriptedReviewer([{'C 1':ui.LOCAL_RENORMALIZE},{'C 1':ui.NEW_EW_OK}]))
            s=args['star_spectrum'].copy();s['flux']*=1.01
            with self.assertRaisesRegex(ValueError,'different spectrum'):
                retry.apply_local_refits(args['linemasks'],output_folder=td,target='Test',star_spectrum=s)

    def test_http_ui_second_save_maps_bad_reason_and_locks_failed_fit(self):
        with tempfile.TemporaryDirectory() as td:
            args=self.setup_run(td)
            candidates=ui.collect_review_candidates(args['figure_dir'],ui.DEFAULT_REASON_OPTIONS)
            candidates[0].fit_failed=True
            htmls=[]
            def post_save(url,*args,**kw):
                htmls.append(urlopen(url).read().decode())
                data=asdict(candidates[0]);data['reason']=ui.NEW_EW_OK
                req=Request(url+'save',data=json.dumps({'items':[data]}).encode(),headers={'Content-Type':'application/json'})
                self.assertIn('Saved',urlopen(req).read().decode())
            with patch.object(ui.webbrowser,'open',side_effect=post_save):
                r=ui.run_line_review(output_folder=td,target='Test',figure_dir=args['figure_dir'],
                    ui_mode='browser',candidates_override=candidates,load_existing_decisions=False,second_pass=True)
            self.assertEqual(r['items'][0].reason,ui.FAILED_BOTH)
            self.assertIn('Failed both QA rounds',htmls[0])
            self.assertIn('await doSave()',htmls[0])

    def test_http_full_first_save_refit_second_save(self):
        with tempfile.TemporaryDirectory() as td:
            args=self.setup_run(td);pages=[]
            def save_page(url,*args,**kw):
                html=urlopen(url).read().decode()
                start=html.index('const payload = ')+len('const payload = ')
                payload,_=json.JSONDecoder().raw_decode(html[start:])
                pages.append(payload)
                if len(pages)==1:
                    reasons=payload['reason_options']
                    self.assertEqual(reasons[reasons.index('other')+1],ui.LOCAL_RENORMALIZE)
                    reason=ui.LOCAL_RENORMALIZE
                else:
                    self.assertNotIn(ui.LOCAL_RENORMALIZE,payload['reason_options'])
                    self.assertNotIn('good',payload['reason_options'])
                    self.assertTrue(payload['second_pass'])
                    self.assertTrue(payload['items'][0]['original_image_path'])
                    self.assertTrue(payload['items'][0]['origin_available'])
                    self.assertEqual(payload['items'][0]['original_ew'],12.)
                    self.assertGreater(payload['items'][0]['new_ew'],0)
                    self.assertEqual(urlopen(url+'image?i=0&original=1').read(),b'fixture')
                    self.assertEqual(payload['items'][0]['reason'],'other')
                    reason=ui.NEW_EW_OK
                data=payload['items'];data[0]['reason']=reason
                # Exercise the immediate image-moving /mark endpoint as a click would.
                mark=Request(url+'mark',data=json.dumps(dict(idx=0,reason=reason,note='')).encode(),
                             headers={'Content-Type':'application/json'})
                urlopen(mark).read()
                request=Request(url+'save',data=json.dumps({'items':data}).encode(),
                                headers={'Content-Type':'application/json'})
                urlopen(request).read()
            with patch.object(ui.webbrowser,'open',side_effect=save_page):
                result=retry.run_review_with_local_retry(**args)
            self.assertEqual(len(pages),2)
            self.assertEqual(result['accepted'],1)

    def test_http_second_round_original_new_and_rejection(self):
        with tempfile.TemporaryDirectory() as td:
            args=self.setup_run(td)
            first=ui.collect_review_candidates(args['figure_dir'],ui.DEFAULT_REASON_OPTIONS)[0]
            candidates=[replace(first,image_name=f'C 1_{i+1}_{5000+i:.2f}.png',wave_nm=500+i/10,
                        wave_a=5000+i,original_image_path=first.image_path,original_ew=12.,new_ew=11.,
                        origin_available=True,reason='other',note='Pending comparison') for i in range(3)]
            # Distinct images prevent one item's immediate mark from moving another's PNG.
            for x in candidates:
                x.image_path=str(Path(args['figure_dir'])/x.image_name)
                Path(x.image_path).write_bytes(b'new-figure')
            def save_comparison(url,*args,**kw):
                html=urlopen(url).read().decode()
                start=html.index('const payload = ')+len('const payload = ')
                payload,_=json.JSONDecoder().raw_decode(html[start:])
                self.assertNotIn('good',payload['reason_options'])
                self.assertIn('Origin EW ok',html);self.assertIn('new EW ok',html)
                data=payload['items']
                for it,reason in zip(data,[ui.ORIGIN_EW_OK,ui.NEW_EW_OK,'crowded_region']):
                    it['reason']=reason
                req=Request(url+'save',data=json.dumps({'items':data}).encode(),headers={'Content-Type':'application/json'})
                self.assertIn('Saved',urlopen(req).read().decode())
            with patch.object(ui.webbrowser,'open',side_effect=save_comparison):
                result=ui.run_line_review(output_folder=td,target='Test',figure_dir=args['figure_dir'],
                    ui_mode='browser',candidates_override=candidates,load_existing_decisions=False,second_pass=True)
            self.assertEqual([x.reason for x in result['items']],['good','good',ui.FAILED_BOTH])
            self.assertEqual([x.ew_choice for x in result['items']],[ui.ORIGIN_EW_OK,ui.NEW_EW_OK,''])

    def test_origin_choice_restores_original_ew_error_profile_and_image(self):
        with tempfile.TemporaryDirectory() as td:
            args=self.setup_run(td)
            result=retry.run_review_with_local_retry(**args,reviewer=ScriptedReviewer([
                {'C 1':ui.LOCAL_RENORMALIZE},{'C 1':ui.ORIGIN_EW_OK}]))
            self.assertEqual(result['accepted_original'],1)
            self.assertEqual(result['accepted_new'],0)
            state=retry._read_state(td,'Test')['lines']['C 1|500.0000']
            self.assertEqual(state['status'],'original_accepted')
            self.assertEqual(state['ew_choice'],ui.ORIGIN_EW_OK)
            altered=args['linemasks'].copy();altered['ew']=99.;altered['ew_err']=.1
            actual=retry.apply_local_refits(altered,output_folder=td,target='Test')
            for field in args['linemasks'].dtype.names:
                np.testing.assert_array_equal(actual[field],args['linemasks'][field])
            with patch.object(retry.np,'load',side_effect=AssertionError('New local flux must not be used')):
                spectrum,line=retry.local_plot_inputs(args['star_spectrum'],actual[0],td)
            self.assertIs(spectrum,args['star_spectrum'])
            self.assertEqual(line['ew'],12.)
            good=Path(args['figure_dir'])/'C_1'/'C 1_1_5000.00.png'
            self.assertEqual(good.read_bytes(),b'fixture')
            self.assertTrue(pd.read_csv(Path(td)/'linemasks'/'Lines_deleted.tsv',sep='\t').empty)

    def test_failed_new_fit_can_still_keep_original(self):
        with tempfile.TemporaryDirectory() as td:
            args=self.setup_run(td)
            args['pipeline_options']['local_ew_continuum_windows']={'C 1|500.0000':[]}
            result=retry.run_review_with_local_retry(**args,reviewer=ScriptedReviewer([
                {'C 1':ui.LOCAL_RENORMALIZE},{'C 1':ui.ORIGIN_EW_OK}]))
            self.assertEqual(result['accepted_original'],1)
            state=retry._read_state(td,'Test')['lines']['C 1|500.0000']
            self.assertTrue(state['fit_failed'])
            self.assertEqual(state['status'],'original_accepted')

    def test_old_pending_record_migrates_and_accepts_original(self):
        with tempfile.TemporaryDirectory() as td:
            args=self.setup_run(td)
            with self.assertRaises(RuntimeError):
                retry.run_review_with_local_retry(**args,reviewer=ScriptedReviewer([
                    {'C 1':ui.LOCAL_RENORMALIZE},'cancel']))
            state=retry._read_state(td,'Test')
            record=state['lines']['C 1|500.0000']
            record.pop('original_fit_values');record.pop('original_image_path')
            retry._write_state(td,'Test',state)
            with patch.object(retry,'refit_with_continuum_fallback',side_effect=AssertionError('No third fit')):
                result=retry.run_review_with_local_retry(**args,reviewer=ScriptedReviewer([{'C 1':ui.ORIGIN_EW_OK}]))
            self.assertEqual(result['accepted_original'],1)

    def test_old_snapshot_migration_preserves_precise_original_ew(self):
        with tempfile.TemporaryDirectory() as td:
            _,lines=fixtures();api=FakeIspec()
            snapshot=Path(td)/'fitted_linemasks_before_retry.txt'
            api.write_line_regions(lines,str(snapshot))
            current=lines[0].copy();current['ew']=99.
            record={'original_ew_mA':12.037}
            retry._restore_original_metadata(record,current,ispec=api,audit_dir=Path(td))
            self.assertEqual(record['original_fit_values']['ew'],12.037)

    def test_second_round_requires_explicit_original_or_new_choice(self):
        candidate=replace(item(),origin_available=True,fit_failed=False)
        for choice in ui.EW_CHOICES:
            selected=ui._review_decision(candidate,choice,'',second_pass=True,final=True)
            self.assertEqual(selected.reason,'good')
            self.assertEqual(selected.ew_choice,choice)
        bad=ui._review_decision(candidate,'good','',second_pass=True,final=True)
        self.assertEqual(bad.reason,ui.FAILED_BOTH)
        rejected=ui._review_decision(candidate,'crowded_region','',second_pass=True,final=True)
        self.assertEqual(rejected.reason,ui.FAILED_BOTH)
        self.assertFalse(rejected.ew_choice)
        unavailable=replace(candidate,origin_available=False)
        self.assertEqual(ui._review_decision(unavailable,ui.ORIGIN_EW_OK,'',second_pass=True,final=True).reason,ui.FAILED_BOTH)

    def test_notebook_entrypoints_pass_spectrum(self):
        root=Path(__file__).resolve().parents[1]/"Code"
        for name in ['0_Sun_abundance.ipynb','3a_atmos_params_iteration_lbl.ipynb','4a_ele_abundance_lbl.ipynb']:
            n=json.loads((root/name).read_text())
            source='\n'.join(''.join(c.get('source',[])) for c in n['cells'] if c['cell_type']=='code')
            expected=('run_review_with_local_retry' if name.startswith('0_') else
                      'prepare_manual_line_edit_tables' if name.startswith('3a') else 'run_nonfe_line_review')
            self.assertIn(expected+'(\n',source)
            self.assertIn('ispec=ispec, star_spectrum=star_spectrum',source)


if __name__=='__main__':
    unittest.main()
