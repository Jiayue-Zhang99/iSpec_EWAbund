# Local iSpec compatibility delta

`ispec-local-compat.patch` records the source workspace's existing changes against iSpec commit `502974867b3a3313f561a3aafb31b1336eba5026`:

1. Expose `plot_spectra` from iSpec's existing plotting module.
2. Guard MOOG-summary wavelength matching against an out-of-range mask index and improve mismatch diagnostics.

The pipeline migration neither changes the original installation nor silently applies this patch to another installation. To reproduce it on a separate compatible checkout, inspect it, then run:

```bash
git -C /path/to/iSpec apply --check /path/to/iSpec_EWAbund/patches/ispec-local-compat.patch
git -C /path/to/iSpec apply /path/to/iSpec_EWAbund/patches/ispec-local-compat.patch
```

Do not reapply to an already patched installation. If the check fails, compare the installed engine source; a different release may already implement the fix. This historical delta is not a full review of all upstream MOOG parsing edge cases. It does not make q2 and iSpec use identical damping or abundance estimators. Follow iSpec's AGPL license for derived engine changes; a license copy is included here.
