from pwassist.core.result import Results
from pwassist.plotting.base import BasePWAPlotter


class ScanPlotter(BasePWAPlotter):
    """Plotter for all results that scan across a range of bins, e.g. mass, t, etc."""
