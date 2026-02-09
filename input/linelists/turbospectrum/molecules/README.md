# Description

Molecules were compiled by Thomas Masseron and supplemented to cover 420-920nm, and they were used in Gerber et al. (2023). This effort has made use of the VALD database, operated at Uppsala University, the Institute of Astronomy RAS in Moscow, and the University of Vienna.

A first compilation was provided by Thomas Masseron in 2015 (files in `2015/` folder, shared via private communication), the updated compilation in this root folder was downloaded from https://keeper.mpdl.mpg.de/d/6eaecbf95b88448f98a4/, where they were uploaded in 2022 (Gerber et al., 2023), and splitted per wavelength regions.

# iSpec

These files can be automatically used only with turbospectrum code if the `use_molecules=True` named argument is passed (see `ispec/synth/turbospectrum.py`) when `generate_spectrum` or `model_spectrum` functions are called. This behavior is experimental and it may change in future iSpec releases.


# References

- Gerber, J. M., Magg, E., Plez, B., et al. 2023, A&A, 669, A43

