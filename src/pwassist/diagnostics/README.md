TODO: This directory will contain diagnostics classes that will run over a fit result and, given some input configuration, provide a report to the user of all potentially worrisome fit results. The idea is that instead of having to make hundreds of plots and analyze them all, this class will provide some guidance on which bins are problematic.
This is distinct from the Preprocessor, which mostly focuses on data validation and preparation, and only flags results that are concerning enough to redo a fit (bad eMatrixStatus in final result). The structure will be very similar to the preprocessor though, with steps and reports constructed.

Some ideas for the steps to run:

1. NormInt diagnostic
    - Check if $N_{ij} / \sqrt{N_{ii}N_{jj}} \approx 1$
    - Check if $N^{det}_{ij} / N^{acc}_{ij}$ has large variations, indicating how the acceptance mixes amplitudes
2. Bootstrap diagnostic
    - Check for multi-modal distributions
3. Rand diagnostic
    - Do several randomized fits have identical likelihoods, but whose parameter results are "far" from each other?