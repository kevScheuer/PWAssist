# PWAssist
PWAssist is a python package designed for users of the GlueX collaboration to analyze their [AmpTools](https://github.com/mashephe/AmpTools)-based partial-wave analysis (PWA) results. It gives the analyzer the abilities to:
1. Identify and collect fit results across mass, $-t$, or other kinematic bins
2. Process and aggregate those results into one flattened file
3. View and plot their results in jupyter notebooks using pandas + matplotlib 

See the [quick start guide](./examples/notebooks/quick_start.ipynb) for an example of how to execute a built-in pipeline and immediately begin viewing your results. A thorough exploration of this package's capabilities can be found in the [walkthrough](./examples/notebooks/walkthrough.ipynb).

## Install



## Documentation
By default, the documentation is built via *GitHub Actions* and posted to the associated [github pages]()  TODO: create link. To build the documentation yourself, run
```shell
uv run --group docs sphinx-build -b html docs/source docs/build
```