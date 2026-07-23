from pwassist.core.result import Results
from pwassist.plotting.base import BasePWAPlotter
from pwassist.plotting.bin import BinPlotter
from pwassist.plotting.scan import ScanPlotter


class FactoryPlotter:
    """Factory class to interface with sub-plotters

    Caches each sub-plotter instance to avoid re-initialization and allow for easy
    access to shared data.
    """

    def __init__(self, results: Results) -> None:
        self.results = results
        self._cache: dict[str, BasePWAPlotter] = {}

    def _get(self, name: str, cls: type[BasePWAPlotter]) -> BasePWAPlotter:
        if name not in self._cache:
            self._cache[name] = cls(self.results)
        return self._cache[name]

    @property
    def scan(self) -> ScanPlotter:
        return self._get("scan", ScanPlotter)  # type: ignore

    @property
    def bin(self) -> BinPlotter:
        return self._get("bin", BinPlotter)  # type: ignore
