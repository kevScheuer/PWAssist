"""Classes for identifying and cataloging PWA result files in a directory structure."""

import pathlib
import warnings
from dataclasses import dataclass
from typing import ClassVar, Self

import numpy as np
import pandas as pd


@dataclass(slots=True)
class ResultsFile:
    """Abstract base class for all CSV files produced from a fit result conversion

    Returns:
        _type_: The type of the ResultsFile.
    """

    path: pathlib.Path  # the path to the CSV file
    frame: pd.DataFrame  # the contents of the CSV file as a DataFrame

    # The columns that must be present in the CSV file for it to be identified as this
    # type.
    required_columns: ClassVar[frozenset[str]] = frozenset()

    @classmethod
    def matches(cls, columns: pd.Index) -> bool:
        """Check if the required columns are present."""
        return cls.required_columns.issubset(set(columns))

    @classmethod
    def identify(cls, path: pathlib.Path) -> bool:
        """Determine if this file is of this type. Override in subclasses."""
        header = pd.read_csv(path, nrows=0)
        return cls.matches(header.columns)

    @classmethod
    def from_path(cls, path: pathlib.Path) -> Self:
        """Create an instance of the ResultsFile from a path.

        Some files (like NormIntFile) require special handling, so this method can be
        overridden in subclasses

        Note:
            This method reads the entire CSV file into memory, which may be
                inefficient for large files. Use with caution.
        """
        frame = pd.read_csv(path)
        if "file" in frame.columns:
            frame["file"] = frame["file"].astype("category")
        return cls(path=path, frame=frame)


@dataclass(slots=True)
class FitFile(ResultsFile):
    """Primary fit result file containing intensities, phases, and AmpTools status codes

    Identified by 'likelihood', 'eMatrixStatus', and 'intensity' columns.

    Todo:
        - Ideally would prefer to identify by content alone, as in other ResultsFile
        types, but currently no unique content-based identifier exists that
        distinguishes it from a RandomizedFile or BootstrapFile.
    """

    required_columns: ClassVar[frozenset[str]] = frozenset(
        {"likelihood", "eMatrixStatus", "intensity"}
    )

    @classmethod
    def identify(cls, path: pathlib.Path) -> bool:
        header = pd.read_csv(path, nrows=0)

        # distinguish from RandomizedFile and BootstrapFile by checking the file path
        if "rand" in str(path).lower() or "bootstrap" in str(path).lower():
            return False
        return cls.matches(header.columns)


@dataclass(slots=True)
class DataFile(ResultsFile):
    """Data file containing the number of events, efficiency, and bin edges.

    Identified by 'events', 'efficiency', 'm_low', and 'm_high' columns.
    """

    required_columns: ClassVar[frozenset[str]] = frozenset(
        {"events", "m_low", "m_high"}
    )

    @classmethod
    def identify(cls, path: pathlib.Path) -> bool:
        header = pd.read_csv(path, nrows=0)
        return cls.matches(header.columns)


@dataclass(slots=True)
class CorrelationFile(ResultsFile):
    """Correlation matrix between fit parameters.

    Identified by 'file' and 'parameter' columns, and if the diagonal elements of the
    numeric columns are equivalent to 1.0.
    """

    required_columns: ClassVar[frozenset[str]] = frozenset({"file", "parameter"})

    @classmethod
    def identify(cls, path: pathlib.Path) -> bool:
        # first check that required columns are present
        header = pd.read_csv(path, nrows=0)
        if not cls.matches(header.columns):
            return False

        # then use small sample of data to check that diagonal elements are 1.0
        df_sample = pd.read_csv(path, nrows=5)
        numeric_cols = df_sample.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) == 0:
            return False

        diagonal_elements = df_sample[numeric_cols].to_numpy().diagonal()

        return bool(
            len(diagonal_elements) > 0 and np.all(np.isclose(diagonal_elements, 1.0))
        )


@dataclass(slots=True)
class CovarianceFile(ResultsFile):
    """Covariance matrix between fit parameters.

    Identified by 'file' and 'parameter' columns with real numeric covariance values
    (values not bounded in [-1, 1], not complex, not 1 on diagonal).
    """

    required_columns: ClassVar[frozenset[str]] = frozenset({"file", "parameter"})

    @classmethod
    def identify(cls, path: pathlib.Path) -> bool:
        # first check that required columns are present
        header = pd.read_csv(path, nrows=0)
        if not cls.matches(header.columns):
            return False

        # then use small sample of data to check that numeric columns are not bounded in
        # [-1, 1] and are not 1 on the diagonal (true for correlation matrices)
        df_sample = pd.read_csv(path, nrows=2)
        numeric_cols = df_sample.select_dtypes(include=[np.number]).columns

        diagonal_elements = df_sample[numeric_cols].to_numpy().diagonal()
        if len(diagonal_elements) == 0 or np.all(np.isclose(diagonal_elements, 1.0)):
            return False

        if len(numeric_cols) == 0:
            return False

        # Check it's not complex
        try:
            if df_sample[numeric_cols].select_dtypes(include=[complex]).shape[1] > 0:
                return False
        except TypeError, ValueError:
            pass

        numeric_data = df_sample[numeric_cols].values.flatten()
        numeric_data = numeric_data[~np.isnan(numeric_data)]

        return bool(
            len(numeric_data) > 0
            and not np.all((numeric_data >= -1.0) & (numeric_data <= 1.0))
        )


@dataclass(slots=True)
class NormIntFile(ResultsFile):
    """Normalization integrals with complex values.

    Identified by 'file' and 'amplitude' columns containing complex-valued
    normalization integral data.
    """

    required_columns: ClassVar[frozenset[str]] = frozenset({"file", "amplitude"})

    @classmethod
    def identify(cls, path: pathlib.Path) -> bool:
        header = pd.read_csv(path, nrows=0)
        return cls.matches(header.columns)

    @classmethod
    def from_path(cls, path: pathlib.Path) -> Self:
        """Create an instance of the NormIntFile from a path.

        We need to ensure that the complex-valued entries are read correctly, so we
        override the base class method to handle this.

        Note:
            This method reads the entire CSV file into memory, which may be
                inefficient for large files. Use with caution.
        """
        frame = pd.read_csv(path)
        if "file" in frame.columns:
            frame["file"] = frame["file"].astype("category")

        # Convert complex columns to complex dtype
        str_cols = (
            frame.drop(columns=["file", "amplitude"])
            .select_dtypes(include=["object", "string"])
            .columns
        )
        # some cols have values like "1.0+-0.0j" which is not a valid complex number, so
        # we need to replace "+-" with "-" before converting
        frame[str_cols] = (
            frame[str_cols]
            .replace(r"\+-", "-", regex=True)
            .apply(lambda col: col.astype(np.complex128))
        )

        return cls(path=path, frame=frame)


@dataclass(slots=True)
class RandomizedFile(ResultsFile):
    """Fits from randomized starting values, used to explore the likelihood landscape.

    Identified by 'likelihood', 'eMatrixStatus' and 'intensity' columns, similar to
    FitFile, but also checks that the 'random' is contained somewhere in the file path.

    Todo:
        - Ideally would prefer to identify by content alone, as in other ResultsFile
        types, but currently no unique content-based identifier exists that
        distinguishes it from a BootstrapFile or FitFile.
    """

    required_columns: ClassVar[frozenset[str]] = frozenset(
        {"likelihood", "eMatrixStatus", "intensity", "file"}
    )

    @classmethod
    def identify(cls, path: pathlib.Path) -> bool:
        # first check that required columns are present
        header = pd.read_csv(path, nrows=0)
        if not cls.matches(header.columns):
            return False

        if "rand" not in str(path).lower():
            return False

        return True


@dataclass(slots=True)
class BootstrapFile(ResultsFile):
    """Fits from bootstrapped datasets, used to estimate uncertainties.

    Identified by 'likelihood', 'eMatrixStatus' and 'intensity' columns, similar to
    FitFile, but also checks that the 'bootstrap' is contained somewhere in the file
    path.

    Todo:
        - Ideally would prefer to identify by content alone, as in other ResultsFile
        types, but currently no unique content-based identifier exists that
        distinguishes it from a RandomizedFile or FitFile.
    """

    required_columns: ClassVar[frozenset[str]] = frozenset(
        {"likelihood", "eMatrixStatus", "intensity", "file"}
    )

    @classmethod
    def identify(cls, path: pathlib.Path) -> bool:
        # first check that required columns are present
        header = pd.read_csv(path, nrows=0)
        if not cls.matches(header.columns):
            return False

        if "bootstrap" not in str(path).lower():
            return False

        return True


class Catalog:
    """A catalog of PWA result files in a directory structure.

    What type of file the CSV is (fit, covariance, etc.) will primarily be
    determined by its content and structure. See ResultsFile for details.

    Attributes:
        input_dir (pathlib.Path): The directory to scan for PWA results.
    """

    RESULT_FILE_TYPES = [
        FitFile,
        DataFile,
        CorrelationFile,
        CovarianceFile,
        NormIntFile,
        RandomizedFile,
        BootstrapFile,
    ]

    REQUIRED_FILE_TYPES = [FitFile, DataFile]
    OPTIONAL_FILE_TYPES = [
        CorrelationFile,
        CovarianceFile,
        NormIntFile,
        RandomizedFile,
        BootstrapFile,
    ]

    def __init__(
        self,
        input_dir: pathlib.Path | str,
        sig_kinematic_digits: int = 3,
        ignore_files: str | list[str] | None = None,
    ):
        """
        Args:
            input_dir (pathlib.Path | str): The top level directory to scan for PWA
                results. All CSV files in this directory and its subdirectories will be
                scanned and cataloged.
            sig_kinematic_digits (int, optional): The number of significant digits to
                use when rounding kinematic bin values (mass, t, energy) for bin
                identification. Defaults to 3.
            ignore_files (str | None, optional): If provided, any CSV file whose name
                matches this string will be ignored during the scan.
        """
        self.input_dir = (
            pathlib.Path(input_dir) if isinstance(input_dir, str) else input_dir
        )

        if not self.input_dir.exists():
            raise ValueError(f"Input directory does not exist: {self.input_dir}")
        if not self.input_dir.is_dir():
            raise ValueError(f"Input path must be a directory: {self.input_dir}")

        # -- private attributes --
        self._manifest: pd.DataFrame | None = None
        self._sig_kinematic_digits = sig_kinematic_digits
        self._ignore_files: list[str] | None = None

        if isinstance(ignore_files, str):
            self._ignore_files = [ignore_files]
        else:
            self._ignore_files = ignore_files

    def scan(self) -> pd.DataFrame:
        """Scan the input directory for CSV files and catalog them.

        We assume that:
            1. There is a data CSV file in each kinematic bin directory that contains
                the binning information (t, beam energy, mass).
            2. All csv's in the same directory, or subdirectories, of the data csv
                belong to the same kinematic bin.

        Returns:
            pd.DataFrame: A DataFrame with columns 'mass', 't', 'energy',
                'file_path', and 'file_type' describing the catalog of PWA result files.

        Raises:
            ValueError: If a CSV file is found that cannot be identified as a known
                result file type.
            FileNotFoundError: If a kinematic bin is missing the required file
                types (FitFile and DataFile).
        """

        records = []
        all_csvs = list(self.input_dir.glob("**/*.csv"))
        data_csvs = [csv_file for csv_file in all_csvs if DataFile.identify(csv_file)]
        dir_to_kinematics_map = self._build_dir_to_kinematics_map(data_csvs)

        orphan_csvs = [
            csv
            for csv in all_csvs
            if not any(
                csv.parent.is_relative_to(kinematic_dir)
                for kinematic_dir in dir_to_kinematics_map.keys()
            )
        ]
        if orphan_csvs:
            warnings.warn(
                f"Found {len(orphan_csvs)} CSV file(s) that could not be associated"
                " with a kinematic bin, likely due to a missing data.csv file in the"
                " directory. These files will be ignored. Orphan files: "
                + ", ".join(str(csv) for csv in orphan_csvs),
                UserWarning,
            )

        for kinematic_dir, kinematics in dir_to_kinematics_map.items():
            mass = kinematics["mass"]
            t = kinematics["t"]
            energy = kinematics["energy"]
            if (
                (mass is None)
                or len(mass) != 2
                or (t is None)
                or len(t) != 2
                or (energy is None)
                or len(energy) != 2
            ):
                raise ValueError(
                    f"Kinematic bin information is incomplete for directory:"
                    f" {kinematic_dir}"
                )

            csv_files_in_bin = list(kinematic_dir.glob("**/*.csv"))
            csv_files_in_bin = [
                f
                for f in csv_files_in_bin
                if not self._ignore_files or f.name not in self._ignore_files
            ]

            file_types_found = {
                self.identify_file_type(csv_file) for csv_file in csv_files_in_bin
            }

            # confirm that required files are present and identifiable
            missing_required = [
                ft.__name__
                for ft in self.REQUIRED_FILE_TYPES
                if ft not in file_types_found
            ]
            if missing_required:
                raise FileNotFoundError(
                    f"Kinematic bin '{str(kinematic_dir)}' is missing required"
                    f" file types: {missing_required}"
                )

            # then catalog all files in the bin, ensuring we have one file per type
            for ft in self.REQUIRED_FILE_TYPES + self.OPTIONAL_FILE_TYPES:
                matching_files = [
                    csv_file
                    for csv_file in csv_files_in_bin
                    if self.identify_file_type(csv_file) == ft
                ]
                if len(matching_files) > 1:
                    raise ValueError(
                        f"Multiple files of type '{ft.__name__}' found in kinematic"
                        f" bin '{str(kinematic_dir)}': "
                        f"{[str(f) for f in matching_files]}"
                    )
                if len(matching_files) == 0:
                    continue
                csv_file = matching_files[0]
                file_type = ft
                size_bytes = csv_file.stat().st_size
                records.append(
                    {
                        "bin_id": (
                            f"T={t[0]},{t[1]}-"
                            f"E={energy[0]},{energy[1]}-"
                            f"M={mass[0]},{mass[1]}"
                        ),
                        "t_bin": t,
                        "energy_bin": energy,
                        "mass_bin": mass,
                        "file_path": str(csv_file.resolve()),
                        "file_type": file_type.__name__,
                        "size_bytes": size_bytes,
                    }
                )

        self._manifest = pd.DataFrame(records)

        return self._manifest

    def identify_file_type(self, path: pathlib.Path) -> type[ResultsFile]:
        """Identify file type by using class' identify method."""
        matches = [ft for ft in self.RESULT_FILE_TYPES if ft.identify(path)]
        if len(matches) == 0 or not any(matches):
            raise ValueError(f"Unknown result file type: {path}")
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous result file type: {path}. Matches multiple types:"
                f" {', '.join(ft.__name__ for ft in matches)}"
            )

        return matches[0]

    @property
    def manifest(self) -> pd.DataFrame:
        """The manifest DataFrame produced by scan().

        Returns:
            pd.DataFrame: A DataFrame with columns 'bin_id', 'file_path', and
                'file_type' describing the catalog of PWA result files.
        """
        if self._manifest is None:
            self.scan()
        if self._manifest is None:
            raise RuntimeError("Manifest is not available after scanning.")
        return self._manifest

    def _round_to_n(self, x: float, n: int) -> float:
        return x if x == 0 else round(x, -int(np.floor(np.log10(abs(x)))) + (n - 1))

    def _snap_edges(
        self, bins: list[tuple[float, float]], rel_tol: float = 0.25
    ) -> dict[float, float]:
        """Map each raw bin edge to a canonical value shared by all nearby edges

        Edges closer together than rel_tol * (narrowest bin width) are considered the
        same edge. Tolerance is derived from the data dataframes.

        Args:
            bins (list[tuple[float, float]]): low, high bin edges for a t, energy, or
                mass dimension.
            rel_tol (float, optional): Multiplies the minimum bin_width to determine
                what bins will be grouped. Defaults to 0.25.

        Returns:
            dict[float, float]: map of raw bin edge to a shared canonical value.
        """
        edges = np.unique(np.concatenate([np.asarray(b, dtype=float) for b in bins]))
        min_width = min(high - low for low, high in bins)
        tol = rel_tol * min_width

        # start a new cluster wherever the gap to the previous edge exceeds tol
        split_idx = np.where(np.diff(edges) > tol)[0] + 1
        mapping: dict[float, float] = {}
        for cluster in np.split(edges, split_idx):
            canonical = self._round_to_n(
                float(np.median(cluster)), self._sig_kinematic_digits
            )
            for edge in cluster:
                mapping[float(edge)] = canonical

        return mapping

    def _build_dir_to_kinematics_map(
        self, data_csv_files: list[pathlib.Path]
    ) -> dict[pathlib.Path, dict[str, tuple[float, float]]]:
        """Map the data file directories to kinematic bin information.

        Raw bin edges from all data files are clustered per dimension so that
        statistically blurred edges (0.0747 vs 0.0748) collapse to one value

        Args:
            data_csv_files (list[pathlib.Path]): all data CSV files discovered from a
                scan.

        Returns:
            dict[pathlib.Path, dict[str, tuple[float, float]]]: A dictionary mapping
                file paths to kinematic bin information. Structure is
                {file_path:
                    {"mass": (low, high),
                    "t": (low, high),
                    "energy": (low, high)}
                }.
        """
        dims = {
            "mass": ("m_low", "m_high"),
            "t": ("t_low", "t_high"),
            "energy": ("e_low", "e_high"),
        }

        # first pass, raw edges per directory
        raw: dict[pathlib.Path, dict[str, tuple[float, float]]] = {}
        for data_csv in data_csv_files:
            row = pd.read_csv(data_csv, nrows=1).iloc[0]
            raw[data_csv.parent] = {
                dim: (float(row[lo]), float(row[hi])) for dim, (lo, hi) in dims.items()
            }
        # second pass: cluster edges per dimension, then snap every bin to canonical
        # edges
        snaps = (
            {dim: self._snap_edges([k[dim] for k in raw.values()]) for dim in dims}
            if raw
            else {}
        )

        return {
            d: {dim: (snaps[dim][lo], snaps[dim][hi]) for dim, (lo, hi) in k.items()}
            for d, k in raw.items()
        }
