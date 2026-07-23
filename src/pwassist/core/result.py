import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from pwassist.io.binning import MassBin
from pwassist.plotting.factory import FactoryPlotter
from pwassist.preprocessing.preprocessor import PreprocessReport, ProcessedBin


@dataclass
class Results:
    """Assembled, preprocessed, analysis-ready results from a collection of PWA bins.

    Todo:
        - Add examples and lots of documentation, as this is the main user-facing class
    """

    fit: pd.DataFrame
    data: pd.DataFrame
    correlation: pd.DataFrame | None = None
    covariance: pd.DataFrame | None = None
    norm_int: pd.DataFrame | None = None

    mass_bins: list[MassBin] = field(default_factory=list)
    reports: list[PreprocessReport] = field(default_factory=list)
    is_acc_corrected: bool = field(init=False)

    _factory_plotter: FactoryPlotter | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        # TODO: find the coherent sums, phase differences, etc. and store them
        is_acc_corrected = self._is_fit_acc_corrected()
        _factory_plotter = FactoryPlotter(self)
        return

    # ----------------------------------------------------------------------------------
    # Constructors
    # ----------------------------------------------------------------------------------

    @classmethod
    def from_processed_bins(cls, processed_bins: list[ProcessedBin]) -> "Results":
        """Construct a Results instance from a list of ProcessedBin objects."""
        # sort the processed bins by mass bin low edge to ensure consistent ordering
        processed_bins = sorted(processed_bins, key=lambda pb: pb.mass_bin.low)

        # concatenate all dataframes for each file type across all bins
        fit_df = pd.concat([pb.fit for pb in processed_bins], ignore_index=True)
        data_df = pd.concat([pb.data for pb in processed_bins], ignore_index=True)
        correlations = (
            pd.concat(
                [pb.correlation for pb in processed_bins if pb.correlation is not None],
                ignore_index=True,
            )
            if any(pb.correlation is not None for pb in processed_bins)
            else None
        )
        covariances = (
            pd.concat(
                [pb.covariance for pb in processed_bins if pb.covariance is not None],
                ignore_index=True,
            )
            if any(pb.covariance is not None for pb in processed_bins)
            else None
        )
        norm_ints = (
            pd.concat(
                [pb.norm_int for pb in processed_bins if pb.norm_int is not None],
                ignore_index=True,
            )
            if any(pb.norm_int is not None for pb in processed_bins)
            else None
        )

        return cls(
            fit=fit_df,
            data=data_df,
            correlation=correlations,
            covariance=covariances,
            norm_int=norm_ints,
            mass_bins=[pb.mass_bin for pb in processed_bins],
            reports=[pb.report for pb in processed_bins],
        )

    @classmethod
    def load(cls, filepath: str | Path) -> "Results":
        """Load a Results instance from a pickle file."""
        filepath = Path(filepath)
        with open(filepath, "rb") as f:
            data: dict = pickle.load(f)
        return cls(**data)

    def save(self, filepath: str | Path) -> None:
        """Save the Results instance to a pickle file."""
        data = {
            "fit": self.fit,
            "data": self.data,
            "correlation": self.correlation,
            "covariance": self.covariance,
            "norm_int": self.norm_int,
            "mass_bins": self.mass_bins,
            "reports": self.reports,
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    # ----------------------------------------------------------------------------------
    # Summaries and Reports
    # ----------------------------------------------------------------------------------

    def summary(self) -> None:
        """Print a summary of the Results instance and its preprocessor warnings."""
        print(f"Results Summary:")
        print(f"  Number of mass bins: {len(self.mass_bins)}")

        for name in ("fit", "data", "correlation", "covariance", "norm_int"):
            df = getattr(self, name)
            if df is not None:
                print(f"\n{name}")
                print(df.info())

        self.warnings()

    def warnings(self) -> None:
        """Print a summary of all warnings from the preprocessing reports."""
        if not self.reports:
            print("No preprocessing reports available.")
            return

        flagged = [r for r in self.reports if r.warnings]
        total_ms = sum(r.total_time_ms for r in self.reports)
        print(f"{len(self.reports)} bins were processed in {total_ms:.2f} ms.")
        for report in flagged:
            print(f"\nBin ID: {report.bin_id}")
            for warning in report.warnings:
                print(f"\t- {warning}")

    def preprocess_summary(self) -> None:
        """Print a summary of the preprocessing reports."""
        if not self.reports:
            print("No preprocessing reports available.")
            return

        print(f"Preprocessing Summary:")
        for report in self.reports:
            print(f"\nBin ID: {report.bin_id}")
            print(f"  Applied Steps: {report.applied_steps}")
            print(f"  Warnings: {report.warnings}")
            print(f"  Timings (ms): {report.timings_ms}")
            print(f"  Total Time (ms): {report.total_time_ms}")

    # ----------------------------------------------------------------------------------
    # Significance Queries
    # ----------------------------------------------------------------------------------

    # TODO: copy over get_significant_amplitudes / phases from old results class

    # ----------------------------------------------------------------------------------
    # Data Queries
    # ----------------------------------------------------------------------------------
    def get_mass_centers(self) -> list[float]:
        """Return the list of mass bin centers."""
        return self.data["m_center"].astype(float).tolist()

    def get_mass_edges(self) -> list[tuple[float, float]]:
        """Return list of mass bin edges in (low, high) pairs"""
        low_edges = self.data["m_low"].astype(float).tolist()
        high_edges = self.data["m_high"].astype(float).tolist()
        return list(zip(low_edges, high_edges))

    def get_average_mass_bin_width(self) -> float:
        """Return average mass bin width across all bins in the results."""
        return (
            self.data["m_high"].astype(float) - self.data["m_low"].astype(float)
        ).mean()

    def get_t_edges(self) -> list[tuple[float, float]]:
        """Return list of t bin edges in (low, high) pairs"""
        low_edges = self.data["t_low"].astype(float).tolist()
        high_edges = self.data["t_high"].astype(float).tolist()
        return list(zip(low_edges, high_edges))

    def get_t_average(self) -> float:
        """Return average t_avg value across all bins in the results.

        Note: This assumes that the Result is constructed from a single t bin. If
            multiple t bins are present, this will return the average of all t_avg
            values, which may be undesired.
        """
        return float(self.data["t_avg"].mean())

    def get_t_rms(self) -> float:
        """Return RMS of t_avg values across all bins in the results.

        Note: This assumes that the Result is constructed from a single t bin. If
            multiple t bins are present, this will return the RMS of all t_avg
            values, which may be undesired.
        """
        return float(self.data["t_rms"].mean())

    def _is_fit_acc_corrected(self) -> bool:
        """Determine if the fit is acceptance-corrected

        Todo: Implement so that it checks if the sum of total reflectivities is greater
            than the number of detected events. Even if these interfere, acceptance
            correction should still recognize this.

            To do this, we will need defined amplitude naming schemes to recognize
            the reflectivity quantum number.

            The problem is that this will fail for non-reflectivity based fits, so
            we may just have to warn the user in this scenario
        """

        warnings.warn(
            "The method _is_fit_acc_corrected is not yet implemented."
            " Returning False by default.",
            UserWarning,
        )
        return False

    # ----------------------------------------------------------------------------------
    # Analysis
    # ----------------------------------------------------------------------------------
    @property
    def plot(self) -> FactoryPlotter:
        """Return a FactoryPlotter instance for plotting the results."""
        if self._factory_plotter is None:
            self._factory_plotter = FactoryPlotter(self)

        return self._factory_plotter
