import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from pwassist.io.binning import MassBin
from pwassist.parser import AmplitudeParser
from pwassist.plotting.factory import FactoryPlotter
from pwassist.preprocessing.preprocessor import PreprocessReport, ProcessedBin


@dataclass
class Results:
    """Assembled, preprocessed, analysis-ready results from a collection of PWA bins.

    Todo:
        - Add examples and lots of documentation, as this is the main user-facing class
    """

    # Result dataframes
    fit: pd.DataFrame
    data: pd.DataFrame
    correlation: pd.DataFrame | None = None
    covariance: pd.DataFrame | None = None
    norm_int: pd.DataFrame | None = None

    # Result metadata
    mass_bins: list[MassBin] = field(default_factory=list)
    reports: list[PreprocessReport] = field(default_factory=list)
    is_acc_corrected: bool = field(init=False)
    final_state_parity: int | None = field(default=None)

    # Amplitude-based attributes
    coherent_sums: dict[str, list[str]] = field(default_factory=dict, init=False)
    amplitudes: list[str] = field(default_factory=list, init=False)
    phase_differences: list[str] = field(default_factory=list, init=False)
    _phase_difference_dict: dict[tuple[str, str], str] = field(
        default_factory=dict, init=False
    )
    _naming_scheme: str | None = field(default=None)
    parser: AmplitudeParser | None = field(default=None, init=False)

    _factory_plotter: FactoryPlotter | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self.is_acc_corrected = self._is_fit_acc_corrected()
        self._factory_plotter = FactoryPlotter(self)

        if self._naming_scheme is None:
            self._naming_scheme = "auto"
        self.parser = AmplitudeParser(self._naming_scheme)

        self.amplitudes = self.parser.get_amplitudes(self.fit.columns.to_list())
        if self.amplitudes is None or len(self.amplitudes) == 0:
            warnings.warn(
                f"No amplitudes found in fit dataframe using naming scheme"
                f" '{self._naming_scheme}'. This will affect plotting scripts.",
                UserWarning,
            )

        self.coherent_sums = self.parser.get_coherent_sums(self.fit.columns.to_list())
        self.phase_differences = self.parser.get_phase_differences(
            self.fit.columns.to_list()
        )
        self._phase_difference_dict = self._build_phase_difference_dict()

        return

    # ----------------------------------------------------------------------------------
    # Constructors
    # ----------------------------------------------------------------------------------

    @classmethod
    def from_processed_bins(
        cls,
        processed_bins: list[ProcessedBin],
        naming_scheme: str | None = None,
        final_state_parity: int | None = None,
    ) -> "Results":
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
            _naming_scheme=naming_scheme,
            final_state_parity=final_state_parity,
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
            "final_state_parity": self.final_state_parity,
            "_naming_scheme": self._naming_scheme,
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

        This is done by checking if the sum of the reflectivities exceeds the number of
        detected events.

        Warning:
            If the fit does not contain reflectivity-based amplitudes, or if the naming
            scheme is not recognized, this method will return False and issue a warning.
        """

        refl_sums = self.coherent_sums.get("e")
        total_reflectivity = (
            [e.sum() for e in self.fit[refl_sums].to_numpy()] if refl_sums else None
        )
        if total_reflectivity is not None:
            detected_events = self.fit["detected_events"].to_numpy()
            if (total_reflectivity > detected_events).any():
                return True
            else:
                return False

        warnings.warn(
            "Could not determine if fit is acceptance-corrected. This may be due to"
            " the naming scheme not being recognized or the fit not containing"
            " reflectivity-based amplitudes. Please check the fit data and naming"
            " scheme to ensure that the results are interpreted correctly.",
            UserWarning,
        )
        return False

    # ----------------------------------------------------------------------------------
    # Amplitude-based Queries
    # ----------------------------------------------------------------------------------
    def find_phase(self, amp1: str, amp2: str) -> str | None:
        """Find the phase difference column name for a given pair of amplitudes.

        Args:
            amp1 (str): Name of the first amplitude.
            amp2 (str): Name of the second amplitude.

        Returns:
            str | None: The name of the phase difference column, or None if not found.
        """
        return self._phase_difference_dict.get((amp1, amp2), None)

    # ----------------------------------------------------------------------------------
    # Analysis
    # ----------------------------------------------------------------------------------
    @property
    def plot(self) -> FactoryPlotter:
        """Return a FactoryPlotter instance for plotting the results."""
        if self._factory_plotter is None:
            self._factory_plotter = FactoryPlotter(self)

        return self._factory_plotter

    # ----------------------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------------------
    def _build_phase_difference_dict(self) -> dict[tuple[str, str], str]:
        """Build dictionary of all pairs of amplitudes and corresponding phases

        Since the phase_difference orderings are not known a priori, it's unclear how
        to request a phase difference column from a dataframe. This dictionary will
        allow for easy lookup of the phase difference for a pair of amplitudes,
        regardless of their order in the dataframe.
        """

        if self.parser is None:
            raise ValueError("AmplitudeParser is not initialized.")

        possible_pairs_to_phases = {}
        for phase in self.phase_differences:
            amps = phase.split("_")
            if len(amps) != 2:
                raise ValueError(
                    f"Phase difference '{phase}' does not correspond to a pair of"
                    " amplitudes."
                )
            possible_pairs_to_phases[tuple(amps)] = phase
            possible_pairs_to_phases[tuple(amps[::-1])] = phase

        return possible_pairs_to_phases
