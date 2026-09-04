"""Classes for handling groups of result files in a kinematic bin."""

import pathlib
from dataclasses import dataclass, field

import pandas as pd

from pwassist.io.catalog import (
    BootstrapFile,
    Catalog,
    CorrelationFile,
    CovarianceFile,
    DataFile,
    FitFile,
    NormIntFile,
    RandomizedFile,
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
    def from_tuple(cls, mass_tuple: tuple[float, float]) -> "MassBin":
        """Create a MassBin from a tuple of (low, high)."""
        low, high = mass_tuple
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
        return f"{self.low}-{self.high}"

    def __lt__(self, other: "MassBin") -> bool:
        """Compare two MassBin instances based on their low values."""
        return self.low < other.low


@dataclass(frozen=True, slots=True)
class TBin:
    low: float
    high: float

    @classmethod
    def from_tuple(cls, t_tuple: tuple[float, float]) -> "TBin":
        """Create a TBin from a tuple of (low, high)."""
        low, high = t_tuple
        return cls(low=low, high=high)

    @property
    def center(self) -> float:
        """Return the center of the t bin."""
        return (self.low + self.high) / 2.0

    @property
    def width(self) -> float:
        """Return the width of the t bin."""
        return self.high - self.low

    def __str__(self) -> str:
        """Return a string representation of the t bin."""
        return f"{self.low}-{self.high}"

    def __lt__(self, other: "TBin") -> bool:
        """Compare two TBin instances based on their low values."""
        return self.low < other.low


@dataclass(frozen=True, slots=True)
class EnergyBin:
    low: float
    high: float

    @classmethod
    def from_tuple(cls, energy_tuple: tuple[float, float]) -> "EnergyBin":
        """Create an EnergyBin from a tuple of (low, high)."""
        low, high = energy_tuple
        return cls(low=low, high=high)

    @property
    def center(self) -> float:
        """Return the center of the energy bin."""
        return (self.low + self.high) / 2.0

    @property
    def width(self) -> float:
        """Return the width of the energy bin."""
        return self.high - self.low

    def __str__(self) -> str:
        """Return a string representation of the energy bin."""
        return f"{self.low}-{self.high}"

    def __lt__(self, other: "EnergyBin") -> bool:
        """Compare two EnergyBin instances based on their low values."""
        return self.low < other.low


@dataclass(frozen=True, slots=True)
class KinematicBin:
    """Represents a kinematic bin defined by mass, t, and energy ranges."""

    mass_bin: MassBin
    t_bin: TBin
    energy_bin: EnergyBin
    bin_id: str

    @classmethod
    def from_bin_id(cls, bin_id: str) -> "KinematicBin":
        """Create a KinematicBin from a bin_id

        Expects the form 'T=<low>,<high>-E=<low>,<high>-M=<low>,<high>'.
        """
        try:
            t_str, energy_str, mass_str = bin_id.split("-")
            t_low, t_high = map(float, t_str.split("=")[1].split(","))
            energy_low, energy_high = map(float, energy_str.split("=")[1].split(","))
            mass_low, mass_high = map(float, mass_str.split("=")[1].split(","))
        except ValueError as e:
            raise ValueError(f"Invalid bin_id format: {bin_id}") from e

        return cls(
            mass_bin=MassBin(mass_low, mass_high),
            t_bin=TBin(t_low, t_high),
            energy_bin=EnergyBin(energy_low, energy_high),
            bin_id=bin_id,
        )

    @classmethod
    def from_tuples(
        cls,
        t_tuple: tuple[float, float],
        energy_tuple: tuple[float, float],
        mass_tuple: tuple[float, float],
    ) -> "KinematicBin":
        """Create a KinematicBin from tuples of (low, high) for t, energy, and mass."""
        t_bin = TBin.from_tuple(t_tuple)
        energy_bin = EnergyBin.from_tuple(energy_tuple)
        mass_bin = MassBin.from_tuple(mass_tuple)
        bin_id = (
            f"T={t_bin.low},{t_bin.high}-"
            f"E={energy_bin.low},{energy_bin.high}-"
            f"M={mass_bin.low},{mass_bin.high}"
        )
        return cls(mass_bin=mass_bin, t_bin=t_bin, energy_bin=energy_bin, bin_id=bin_id)

    def __str__(self) -> str:
        """Return a string representation of the kinematic bin."""
        return (
            f"T={self.t_bin.low},{self.t_bin.high}-"
            f"E={self.energy_bin.low},{self.energy_bin.high}-"
            f"M={self.mass_bin.low},{self.mass_bin.high}"
        )

    def __lt__(self, other: "KinematicBin") -> bool:
        """Compare two KinematicBin instances based on their t, energy, and mass bins.

        Orders by t_bin first, then energy_bin, then mass_bin.
        """
        if self.t_bin != other.t_bin:
            return self.t_bin < other.t_bin
        if self.energy_bin != other.energy_bin:
            return self.energy_bin < other.energy_bin
        return self.mass_bin < other.mass_bin


@dataclass(slots=True)
class BinBundle:
    """Contains set of ResultsFile DataFrames for a kinematic bin."""

    kinematic_bin: KinematicBin
    bin_id: str
    paths: dict[str, pathlib.Path]  # file_type -> path

    # Loaded ResultsFile DataFrames: file_type -> ResultsFile instance
    # This is a cache to avoid reloading the same file multiple times.
    _loaded: dict[str, ResultsFile] = field(default_factory=dict, init=False)

    def get(self, file_type: str | type[ResultsFile]) -> ResultsFile | None:
        """Get loaded ResultsFile for the given file type, or None if not found."""

        if isinstance(file_type, type) and issubclass(file_type, ResultsFile):
            cls = file_type
            file_label = cls.__name__
        elif isinstance(file_type, str):
            cls = FILE_TYPE_MAP[file_type]
            file_label = file_type
        else:
            raise TypeError(
                f"file_type must be str or ResultsFile, got {type(file_type)}"
            )

        if file_label not in self.paths:
            return None
        if file_label not in self._loaded:
            self._loaded[file_label] = cls.from_path(self.paths[file_label])
        return self._loaded[file_label]

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

    @property
    def randomized(self) -> RandomizedFile | None:
        return self.get("RandomizedFile")  # type: ignore

    @property
    def bootstrap(self) -> BootstrapFile | None:
        return self.get("BootstrapFile")  # type: ignore

    def unload(self, file_type: str | type[ResultsFile] | None = None) -> None:
        """Drop loaded DataFrame(s) after preprocessing to save memory."""
        if file_type is None:
            self._loaded.clear()
        elif isinstance(file_type, type) and issubclass(file_type, ResultsFile):
            self._loaded.pop(file_type.__name__, None)
        elif isinstance(file_type, str):
            self._loaded.pop(file_type, None)
        else:
            raise TypeError(
                f"file_type must be str or ResultsFile, got {type(file_type)}"
            )


class BinCollection:
    """Collection of kinematic bins and their associated bundle of fit results / data"""

    def __init__(self, manifest: pd.DataFrame):
        kinematic_bin_to_bundles: dict[KinematicBin, BinBundle] = {}
        for bin_id, group in manifest.groupby("bin_id"):
            kinematic_bin = KinematicBin.from_bin_id(str(bin_id))
            paths = {
                row["file_type"]: pathlib.Path(row["file_path"])
                for _, row in group.iterrows()
            }
            kinematic_bin_to_bundles[kinematic_bin] = BinBundle(
                kinematic_bin=kinematic_bin, bin_id=str(bin_id), paths=paths
            )

        self._kinematic_bin_to_bundles = kinematic_bin_to_bundles
        self._order = sorted(kinematic_bin_to_bundles)  # sorts on KinematicBin.__lt__

    def __iter__(self):
        for kinematic_bin in self._order:
            yield kinematic_bin, self._kinematic_bin_to_bundles[kinematic_bin]

    def __getitem__(self, kinematic_bin: KinematicBin) -> BinBundle:
        return self._kinematic_bin_to_bundles[kinematic_bin]

    def __len__(self) -> int:
        return len(self._kinematic_bin_to_bundles)

    @classmethod
    def from_catalog(cls, catalog: Catalog) -> "BinCollection":
        return cls(catalog.manifest)
