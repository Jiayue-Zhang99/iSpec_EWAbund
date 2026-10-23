"""Scientific-interface regression tests (run under ispec_env_2025)."""
import sys
import tempfile
import unittest
import ast
import json
import hashlib
from types import SimpleNamespace
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ispec
from spectrum_io import read_spectrum_for_pipeline


class SpectrumIOTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def write(self, name, text):
        path = self.root / name
        path.write_text(text)
        return path

    def test_legacy_ivar_preserves_first_rows_and_converts(self):
        p = self.write('test_stitched_auto.txt', '5000 1 10000\n5001 .5 2500\n5002 .9 40000\n')
        spectrum, report = read_spectrum_for_pipeline(ispec, p)
        np.testing.assert_allclose(spectrum['waveobs'], [5000, 5001, 5002])
        np.testing.assert_allclose(spectrum['err'], [.01, .02, .005])
        self.assertEqual(report['input_pixels'], 3)
        self.assertTrue(report['converted_ivar'])

    def test_header_sigma_not_converted_twice_and_fits_roundtrip(self):
        p = self.write('test_stitched_auto.txt', 'waveobs\tflux\terr\n5000\t1\t.01\n5001\t.5\t.02\n5002\t.9\t.005\n')
        spectrum, report = read_spectrum_for_pipeline(ispec, p)
        self.assertFalse(report['converted_ivar'])
        spectrum['waveobs'] /= 10
        output = self.root / 'converted.fits'
        ispec.write_spectrum(spectrum, str(output))
        back, _ = read_spectrum_for_pipeline(ispec, output)
        np.testing.assert_allclose(back['err'], [.01, .02, .005], rtol=1e-6)
        np.testing.assert_allclose(back['flux'], [1, .5, .9], rtol=1e-6)

    def test_bad_ivar_pixels_removed_not_zero_weight_errors(self):
        p = self.write('test_stitched_auto.txt', '5000 1 100\n5001 1 0\n5002 1 -2\n5003 1 nan\n5004 1 inf\n5005 .8 400\n')
        spectrum, report = read_spectrum_for_pipeline(ispec, p)
        np.testing.assert_allclose(spectrum['err'], [.1, .05])
        self.assertEqual(report['invalid_pixels_removed'], 4)

    def test_unknown_headerless_requires_explicit_semantics(self):
        p = self.write('unknown.txt', '5000 1 .01\n5001 .9 .02\n')
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            read_spectrum_for_pipeline(ispec, p)
        spectrum, _ = read_spectrum_for_pipeline(ispec, p, error_kind='sigma')
        np.testing.assert_allclose(spectrum['err'], [.01, .02])

    def test_header_declares_ivar_and_conflicting_override_rejected(self):
        p = self.write('named.txt', 'waveobs flux ivar\n5000 1 100\n5001 .9 400\n')
        spectrum, _ = read_spectrum_for_pipeline(ispec, p)
        np.testing.assert_allclose(spectrum['err'], [.1, .05])
        with self.assertRaisesRegex(ValueError, 'contradicts'):
            read_spectrum_for_pipeline(ispec, p, error_kind='sigma')

    def test_unknown_errors_and_duplicates(self):
        p = self.write('plain.txt', 'waveobs flux err\n5001 .9 0\n5000 1 0\n5000 .7 0\n')
        spectrum, report = read_spectrum_for_pipeline(ispec, p)
        np.testing.assert_allclose(spectrum['waveobs'], [5000, 5001])
        np.testing.assert_allclose(spectrum['flux'], [1, .9])
        self.assertEqual(report['duplicate_pixels_removed'], 1)

    def test_pipeline_wrapper_units_and_provenance(self):
        from DataPreProcess import read_write_spectrum
        path = self.write('small_stitched_auto.txt', '5000 1 10000\n5001 .8 2500\n5002 .9 40000\n')
        spectrum = read_write_spectrum(str(path), 'test', str(self.root), True)
        np.testing.assert_allclose(spectrum['waveobs'], [500, 500.1, 500.2])
        np.testing.assert_allclose(spectrum['err'], [.01, .02, .005])
        provenance = json.loads((self.root/'spectra_test.fits.provenance.json').read_text())
        self.assertEqual(provenance['source_sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(provenance['source_error_kind'], 'ivar')


if __name__ == '__main__':
    unittest.main()
