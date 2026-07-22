"""Classes for storing and interacting with groups of PWA result files in a mass bin."""

import pathlib
from dataclasses import dataclass, field

import pandas as pd

from pwassist.io.catalog import (
    Catalog,
    CorrelationFile,
    CovarianceFile,
    DataFile,
    FitFile,
    NormIntFile,
    ResultsFile,
)

FILE_TYPE_MAP: dict[str, type[ResultsFile]] = {
    cls.__name__: cls for cls in Catalog.RESULT_FILE_TYPES
}


@dataclass(frozen=True, slots=True)
class MassBin:
    low: float
    high: float

    @classmethod
    def from_bin_id(cls, bin_id: str) -> "MassBin":
        """Create a MassBin from a bin ID string of the form 'mass_low-high'."""
        low, high = map(float, bin_id.split("_")[1].split("-"))
        if not low or not high:
            raise ValueError(f"Invalid bin ID: {bin_id}")

        return cls(low=low, high=high)

    @property
    def center(self) -> float:
        """Return the center of the mass bin."""
        return (self.low + self.high) / 2.0

    @property
    def width(self) -> float:
        """Return the width of the mass bin."""
        return self.high - self.low

    def __str__(self) -> str:
        """Return a string representation of the mass bin."""
        return f"{self.low}-{self.high})"

    def __lt__(self, other: "MassBin") -> bool:
        """Compare two MassBin instances based on their low values."""
        return self.low < other.low


@dataclass(slots=True)
class BinBundle:
    """Contains set of ResultsFile DataFrames for a mass bin."""

    mass_bin: MassBin
    bin_id: str
    paths: dict[str, pathlib.Path]  # file_type -> path

    # Loaded ResultsFile DataFrames: file_type -> ResultsFile instance
    _loaded: dict[str, ResultsFile] = field(default_factory=dict, init=False)

    def get(self, file_type: str) -> ResultsFile | None:
        """Get loaded ResultsFile for the given file type, or None if not found."""
        if file_type not in self.paths:
            return None
        if file_type not in self._loaded:
            cls = FILE_TYPE_MAP[file_type]
            self._loaded[file_type] = cls.from_path(self.paths[file_type])
        return self._loaded[file_type]

    @property
    def fit(self) -> FitFile:
        return self.get("FitFile")  # type: ignore

    @property
    def data(self) -> DataFile:
        return self.get("DataFile")  # type: ignore

    @property
    def correlation(self) -> CorrelationFile | None:
        return self.get("CorrelationFile")  # type: ignore

    @property
    def covariance(self) -> CovarianceFile | None:
        return self.get("CovarianceFile")  # type: ignore

    @property
    def norm_int(self) -> NormIntFile | None:
        return self.get("NormIntFile")  # type: ignore

    def unload(self, file_type: str | None = None) -> None:
        """Drop loaded DataFrame(s) after preprocessing to save memory."""
        if file_type is None:
            self._loaded.clear()
        else:
            self._loaded.pop(file_type, None)


class BinCollection:
    """Collection of mass bins and their associated bundle of fit results / data"""

    def __init__(self, manifest: pd.DataFrame):
        mass_to_bundles: dict[MassBin, BinBundle] = {}
        for bin_id, group in manifest.groupby("bin_id"):
            mass_bin = MassBin.from_bin_id(str(bin_id))
            paths = {
                row["file_type"]: pathlib.Path(row["file_path"])
                for _, row in group.iterrows()
            }
            mass_to_bundles[mass_bin] = BinBundle(
                mass_bin=mass_bin, bin_id=str(bin_id), paths=paths
            )

        self._mass_to_bundles = mass_to_bundles
        self._order = sorted(mass_to_bundles)  # sorts on MassBin.__lt__

    def __iter__(self):
        for mass_bin in self._order:
            yield mass_bin, self._mass_to_bundles[mass_bin]

    def __getitem__(self, mass_bin: MassBin) -> BinBundle:
        return self._mass_to_bundles[mass_bin]

    def __len__(self) -> int:
        return len(self._mass_to_bundles)

    @classmethod
    def from_catalog(cls, catalog: Catalog) -> "BinCollection":
        return cls(catalog.manifest)
