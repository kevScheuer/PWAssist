from pwassist.core.result import Results
from pwassist.plotting.base import BasePWAPlotter


class ScanPlotter(BasePWAPlotter):
    """Plotter for all results that scan across a range of bins, e.g. mass, t, etc."""

    # TODO: make sure to run in format:
    # with self._style():
    #   ...
    # return ax
    #
    # and to always prepare common plot data as dataframe, with optional fit_index
    # style mass bin picking
