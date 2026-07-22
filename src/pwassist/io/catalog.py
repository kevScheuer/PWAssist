"""Classes for identifying and cataloging PWA result files in a directory structure."""

import pathlib
from dataclasses import dataclass
from typing import ClassVar, Self

import numpy as np
import pandas as pd


@dataclass(slots=True)
class ResultsFile:
    """Abstract base class for all CSV files produced from a fit result conversion

    Attributes:
        path (pathlib.Path): The path to the CSV file.
        frame (pd.DataFrame): The contents of the CSV file as a DataFrame.

    Returns:
        _type_: The type of the ResultsFile.
    """

    path: pathlib.Path
    frame: pd.DataFrame

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

        Note that this method reads the entire CSV file into memory, which may be
        inefficient for large files. Use with caution.
        """
        frame = pd.read_csv(path)
        return cls(path=path, frame=frame)


@dataclass(slots=True)
class FitFile(ResultsFile):
    """Primary fit result file containing intensities, phases, and AmpTools status codes

    Identified by 'likelihood', 'eMatrixStatus', and 'intensity' columns.
    """

    required_columns: ClassVar[frozenset[str]] = frozenset(
        {"likelihood", "eMatrixStatus", "intensity"}
    )

    @classmethod
    def identify(cls, path: pathlib.Path) -> bool:
        header = pd.read_csv(path, nrows=0)
        return cls.matches(header.columns)


@dataclass(slots=True)
class DataFile(ResultsFile):
    """Data file containing the number of events, efficiency, and bin edges.

    Identified by 'events', 'efficiency', 'm_low', and 'm_high' columns.
    """

    required_columns: ClassVar[frozenset[str]] = frozenset(
        {"events", "efficiency", "m_low", "m_high"}
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
    (values not bounded in [-1, 1] and not complex).
    """

    required_columns: ClassVar[frozenset[str]] = frozenset({"file", "parameter"})

    @classmethod
    def identify(cls, path: pathlib.Path) -> bool:
        # first check that required columns are present
        header = pd.read_csv(path, nrows=0)
        if not cls.matches(header.columns):
            return False

        # then use small sample of data to check that numeric columns are not bounded in
        # [-1, 1] and not complex
        df_sample = pd.read_csv(path, nrows=2)
        numeric_cols = df_sample.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) == 0:
            return False

        # Check it's not complex
        try:
            if df_sample[numeric_cols].select_dtypes(include=[complex]).shape[1] > 0:
                return False
        except (TypeError, ValueError):
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


class Catalog:
    """Scans an input directory for PWA results organized into mass bins.

    The class currently expects the following structure
    input_dir/
        mass_1.0-1.1/
            fit.csv
            fit_data.csv
            fit_correlation.csv
            ...
        mass_1.1-1.2/
            ...

    Note that the file names are not hardcoded. What type of file the CSV is will be
    determined by the column and data it contains. See ResultsFile for details.

    Attributes:
        input_dir (pathlib.Path): The directory to scan for PWA results.
        catalog (pd.DataFrame): A DataFrame containing the catalog of PWA results.
    """

    RESULT_FILE_TYPES = [
        FitFile,
        DataFile,
        CorrelationFile,
        CovarianceFile,
        NormIntFile,
    ]

    REQUIRED_FILE_TYPES = [FitFile, DataFile]
    OPTIONAL_FILE_TYPES = [CorrelationFile, CovarianceFile, NormIntFile]

    def __init__(
        self,
        input_dir: pathlib.Path | str,
    ):
        """
        Args:
            input_dir (pathlib.Path | str): The directory to scan for PWA results.
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

    def scan(self) -> pd.DataFrame:
        """Scan the input directory for CSV files and catalog them.

        Note that current structure is hard coded to expect mass bin directories
        directly under the input directory.

        Returns:
            pd.DataFrame: A DataFrame with columns 'bin_id', 'file_path', and
                'file_type' describing the catalog of PWA result files.

        Raises:
            ValueError: If a CSV file is found that cannot be identified as a known
                result file type.
            FileNotFoundError: If a mass bin directory is missing the required file
                types (FitFile and DataFile).
        """

        records = []

        for mass_bin_dir in sorted(self.input_dir.iterdir()):
            if not mass_bin_dir.is_dir():
                continue

            bin_id = mass_bin_dir.name

            csv_iterator = sorted(mass_bin_dir.glob("*.csv"))

            # first confirm that required files are present and identifiable
            file_types_found = {
                self.identify_file_type(csv_file) for csv_file in csv_iterator
            }

            missing_required = [
                ft.__name__
                for ft in self.REQUIRED_FILE_TYPES
                if ft not in file_types_found
            ]
            if missing_required:
                raise FileNotFoundError(
                    f"Mass bin '{bin_id}' is missing required file types:"
                    f" {missing_required}"
                )

            # then catalog all files in the bin
            for csv_file in csv_iterator:
                file_type = self.identify_file_type(csv_file)
                size_bytes = csv_file.stat().st_size
                records.append(
                    {
                        "bin_id": bin_id,
                        "file_path": str(csv_file.resolve()),
                        "file_type": file_type.__name__,
                        "size_bytes": size_bytes,
                    }
                )

        self._manifest = pd.DataFrame(records)

        return self._manifest

    def identify_file_type(self, path: pathlib.Path) -> type[ResultsFile]:
        """Identify file type by using class' identify method."""
        for file_type in self.RESULT_FILE_TYPES:
            if file_type.identify(path):
                return file_type

        raise ValueError(f"Unknown result file type: {path}")

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
