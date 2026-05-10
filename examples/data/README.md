# Data
Contained here is a set of partial-wave analysis results to a set of pseudodata
in three independent neighboring bins. These were produced using a conversion
script (TODO: link once pushed to master) within the GlueX
[halld_sim](https://github.com/JeffersonLab/halld_sim) repository. This converter 
changes [AmpTools](https://github.com/mashephe/AmpTools) `.fit` results into csv
files. A short description of each file is found below 

*Note that more file types will be added in future updates*

## Fit
The "bread and butter" of our partial-wave analysis. It contains AmpTools status
outputs, production coefficients, and intensities and phase differences of all
the user-defined amplitudes. Practically all the results we are interested in
live within this output.

## Data
A simplified form of the ROOT files the partial-wave analysis was performed on.
This includes the bin information of the mass, beam energy, and -t
distributions. We'll use this to "stitch together" our binned fits into a smooth
mass spectrum, and for labelling our fit results.

## Covariance / Correlation
These contain the full covariance and correlation matrices between all the
AmpTools parameters, including our production coefficients. Connecting these up
to the intensities and phase differences themselves is difficult, and better
left to AmpTools, but they can be useful (optional) additions to diagnose
problematic fits.

## Norm_int (Normalization Integrals)
These handle the normalization that come from the acceptance estimation that is
built off the generated and phasespace MC. It is an optional file that can be 
used for fit diagnostics. See the AmpTools documentation for more details.