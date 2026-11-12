import dataclasses
import os
import pickle
import subprocess
import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import scipy

from pwassist.core.parameters import (
    coefficient_bases,
    group_by_unique,
    matches_unique,
    reaction_scale_columns,
    split_full_name,
)
from pwassist.io.binning import EnergyBin, KinematicBin, MassBin, TBin
from pwassist.parser import AmplitudeParser, NamingScheme
from pwassist.plotting.factory import FactoryPlotter
from pwassist.preprocessing.preprocessor import PreprocessReport, ProcessedBin

NormIntType = Literal["accepted", "generated", "unknown"]
CoefficientFrame = Literal["fit", "randomized", "bootstrap"]


def _get_current_git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            check=True,
            capture_output=True,
            text=True,
        )
    except OSError, subprocess.CalledProcessError:
        return None
    return result.stdout.strip()


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
    bootstrap: pd.DataFrame | None = None
    randomized: pd.DataFrame | None = None

    # Result metadata
    kinematic_bins: list[KinematicBin] = field(default_factory=list)
    unique_mass_bins: frozenset[MassBin] = field(init=False)
    unique_t_bins: frozenset[TBin] = field(init=False)
    unique_energy_bins: frozenset[EnergyBin] = field(init=False)
    reports: list[PreprocessReport] = field(default_factory=list)
    is_acc_corrected: bool = field(init=False)
    final_state_parity: int | None = field(default=None)
    conjugation_references: tuple[tuple[str, str, int], ...] = field(
        default_factory=tuple
    )
    git_commit_hash: str | None = field(
        default_factory=_get_current_git_commit_hash, repr=False
    )

    # Amplitude-based attributes
    coherent_sums: dict[str, tuple[str, ...]] = field(default_factory=dict, init=False)
    amplitudes: tuple[str, ...] = field(default_factory=tuple, init=False)
    phase_differences: tuple[str, ...] = field(default_factory=tuple, init=False)
    parser: AmplitudeParser = field(init=False)
    are_reactions_constrained: bool = False
    are_sums_constrained: bool = False

    _phase_difference_dict: dict[tuple[str, str], str] = field(
        default_factory=dict, init=False
    )
    _naming_scheme: str | None | NamingScheme = field(default=None)
    _factory_plotter: FactoryPlotter | None = field(
        default=None, init=False, repr=False
    )
    _norm_int_type: NormIntType | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._factory_plotter = FactoryPlotter(self)

        if self._naming_scheme is None:
            self._naming_scheme = "auto"
        self.parser = AmplitudeParser(
            self._naming_scheme, final_state_parity=self.final_state_parity
        )

        self.amplitudes = tuple(self.parser.get_amplitudes(self.fit.columns.to_list()))
        if self.amplitudes is None or len(self.amplitudes) == 0:
            warnings.warn(
                f"No amplitudes found in fit dataframe using naming scheme"
                f" '{self._naming_scheme}'. This will affect plotting scripts.",
                UserWarning,
            )

        # ensure that the coherent sums and phase differences are immutable
        coh_sums = self.parser.get_coherent_sums(self.fit.columns.to_list())
        self.coherent_sums = {
            sum_label: tuple(coh_sum) for sum_label, coh_sum in coh_sums.items()
        }
        self.phase_differences = tuple(
            self.parser.get_phase_differences(self.fit.columns.to_list())
        )
        self._phase_difference_dict = self._build_phase_difference_dict()

        self.is_acc_corrected = self._is_fit_acc_corrected()

        self.unique_mass_bins = frozenset(
            sorted(kb.mass_bin for kb in self.kinematic_bins)
        )
        self.unique_t_bins = frozenset(sorted(kb.t_bin for kb in self.kinematic_bins))
        self.unique_energy_bins = frozenset(
            sorted(kb.energy_bin for kb in self.kinematic_bins)
        )

        # We can infer whether reaction/sum names are constrained by counting the number
        # of "::" occurrences in the production coefficient columns.
        prod_coeff_cols = [c for c in self.fit.columns if "_re" in c or "_im" in c]
        if any([c.count("::") > 2 for c in prod_coeff_cols]):
            raise ValueError(f"Unexpected format for production coefficient columns")
        elif any([c.count("::") == 2 for c in prod_coeff_cols]):
            # labeled '<reaction>::<sum>::<amplitude>_<part>'
            self.are_sums_constrained = False
            self.are_reactions_constrained = False
        elif any([c.count("::") == 1 for c in prod_coeff_cols]):
            # labeled '<sum>::<amplitude>_<part>'
            self.are_sums_constrained = False
            self.are_reactions_constrained = True
        else:
            # labeled '<amplitude>_<part>'
            self.are_sums_constrained = True
            self.are_reactions_constrained = True

        return

    # ----------------------------------------------------------------------------------
    # Constructors and Loaders
    # ----------------------------------------------------------------------------------

    @classmethod
    def from_processed_bins(
        cls,
        processed_bins: list[ProcessedBin],
        naming_scheme: str | None | NamingScheme = None,
        final_state_parity: int | None = None,
    ) -> "Results":
        """Construct a Results instance from a list of ProcessedBin objects."""
        # sort the processed bins by kinematic bin to ensure consistent ordering
        processed_bins = sorted(processed_bins, key=lambda pb: pb.kinematic_bin)

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
        randomized = (
            pd.concat(
                [pb.randomized for pb in processed_bins if pb.randomized is not None],
                ignore_index=True,
            )
            if any(pb.randomized is not None for pb in processed_bins)
            else None
        )
        bootstrap = (
            pd.concat(
                [pb.bootstrap for pb in processed_bins if pb.bootstrap is not None],
                ignore_index=True,
            )
            if any(pb.bootstrap is not None for pb in processed_bins)
            else None
        )

        return cls(
            fit=fit_df,
            data=data_df,
            correlation=correlations,
            covariance=covariances,
            norm_int=norm_ints,
            randomized=randomized,
            bootstrap=bootstrap,
            kinematic_bins=[pb.kinematic_bin for pb in processed_bins],
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
        data.setdefault("git_commit_hash", None)
        data.setdefault("conjugation_references", ())
        return cls(**data)

    def save(self, filepath: str | Path) -> None:
        """Save the Results instance to a pickle file."""
        data = {
            "fit": self.fit,
            "data": self.data,
            "correlation": self.correlation,
            "covariance": self.covariance,
            "norm_int": self.norm_int,
            "randomized": self.randomized,
            "bootstrap": self.bootstrap,
            "kinematic_bins": self.kinematic_bins,
            "reports": self.reports,
            "final_state_parity": self.final_state_parity,
            "conjugation_references": self.conjugation_references,
            "git_commit_hash": self.git_commit_hash,
            "_naming_scheme": self._naming_scheme,
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    def _matches_current_git_commit(self) -> bool:
        current_commit = _get_current_git_commit_hash()
        return (
            self.git_commit_hash is not None
            and current_commit is not None
            and self.git_commit_hash == current_commit
        )

    def filter_by_kinematic_bins(
        self,
        kinematic_bins: KinematicBin | list[KinematicBin] | None = None,
        t_bins: (
            tuple[float, float] | TBin | list[tuple[float, float]] | list[TBin] | None
        ) = None,
        energy_bins: (
            tuple[float, float]
            | EnergyBin
            | list[tuple[float, float]]
            | list[EnergyBin]
            | None
        ) = None,
        mass_bins: (
            tuple[float, float]
            | MassBin
            | list[tuple[float, float]]
            | list[MassBin]
            | None
        ) = None,
    ) -> "Results":
        """Return a new Results instance filtered by the specified kinematic bins.

        Args:
            kinematic_bins (KinematicBin | list[KinematicBin] | None): List of
                KinematicBin instances to filter by. If specified, t_bins, energy_bins,
                and mass_bins will be ignored.
            t_bins (
                tuple[float, float] | TBin | list[tuple[float, float]] | list[TBin]
                | None): List of requested t bins, either as (low, high) tuples or TBin
                instances.
            energy_bins (
                tuple[float, float] | EnergyBin | list[tuple[float, float]]
                | list[EnergyBin] | None): List of requested energy bins, either as
                (low, high) tuples or EnergyBin instances.
            mass_bins (
                tuple[float, float] | MassBin | list[tuple[float, float]]
                | list[MassBin] | None): List of requested mass bins, either as
                (low, high) tuples or MassBin instances.
        """
        selected_bins: list[KinematicBin]

        # Get the selected kinematic bins and corresponding bin_ids
        if kinematic_bins is not None:
            if isinstance(kinematic_bins, KinematicBin):
                selected_bins = [kinematic_bins]
            else:
                selected_bins = kinematic_bins

            if any(kb not in self.kinematic_bins for kb in selected_bins):
                raise KeyError(
                    "One or more specified kinematic bins are not present in the"
                    " results."
                )
        else:
            selected_bins = self._resolve_kinematic_bins(
                t_bins=t_bins, energy_bins=energy_bins, mass_bins=mass_bins
            )

        selected_ids = {kb.bin_id for kb in selected_bins}
        if not selected_ids:
            raise KeyError(
                "No kinematic bins found matching the specified criteria. Please check"
                " the input parameters."
            )

        def _filter(df: pd.DataFrame | None) -> pd.DataFrame | None:
            if df is None:
                return None
            return df[df["bin_id"].isin(selected_ids)].reset_index(drop=True)

        return dataclasses.replace(
            self,
            fit=_filter(self.fit),
            data=_filter(self.data),
            correlation=_filter(self.correlation),
            covariance=_filter(self.covariance),
            norm_int=_filter(self.norm_int),
            bootstrap=_filter(self.bootstrap),
            randomized=_filter(self.randomized),
            kinematic_bins=selected_bins,
            reports=[r for r in self.reports if r.bin_id in selected_ids],
        )

    # ----------------------------------------------------------------------------------
    # Reports
    # ----------------------------------------------------------------------------------

    def summary(self) -> None:
        """Print a summary of the Results instance and its preprocessor warnings."""
        try:
            width = os.get_terminal_size().columns
        except OSError:
            width = 80  # default width if terminal size cannot be determined
        print(f"{'-' * ((width -16)// 2)}Results Summary:{'-' * ((width -16)// 2)}")
        print(f"Number of kinematic bins: {len(self.kinematic_bins)}")
        print(f"\tUnique mass bins: {len(self.unique_mass_bins)}")
        print(f"\tUnique t bins: {len(self.unique_t_bins)}")
        print(f"\tUnique beam energy bins: {len(self.unique_energy_bins)}")
        print(f"Number of preprocessing reports: {len(self.reports)}")

        for name in (
            "fit",
            "data",
            "correlation",
            "covariance",
            "norm_int",
            "randomized",
            "bootstrap",
        ):
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
    # Kinematic Bin Extraction
    # ----------------------------------------------------------------------------------

    def mass_kinematic_bins(
        self,
        t_bin: tuple[float, float] | TBin | None,
        energy_bin: tuple[float, float] | EnergyBin | None,
    ) -> list[KinematicBin]:
        """List of mass bins, optionally filtered by t and/or energy bins.

        Args:
            t_bin (tuple[float,float] | TBin | None): Low and high edges of the t bin to
                select, or a TBin instance. Only needs specification if the results span
                multiple t bins. If None, the sole available t bin will be used.
            energy_bin (tuple[float,float] | EnergyBin | None): Low and high edges of
                the energy bin to select, or an EnergyBin instance. Only needs
                specification if the results span multiple energy bins. If None, the
                sole available energy bin will be used.

        Returns:
            list[KinematicBin]: List of mass bins that match the specified t and energy
                bins.

        Raises:
            KeyError: If no kinematic bins match the specified t and energy bins, or if
                the results are ambiguous.
            ValueError: If multiple t or energy bins are present and none is specified,
                making the mass bin selection ambiguous.
        """
        matching_kinematic_bins: list[KinematicBin] = self._resolve_kinematic_bins(
            t_bins=t_bin, energy_bins=energy_bin
        )
        if not matching_kinematic_bins and (
            t_bin is not None or energy_bin is not None
        ):
            raise KeyError(
                f"No kinematic bins found matching t_bins={t_bin} and"
                f" energy_bins={energy_bin}."
            )
        elif not matching_kinematic_bins:
            raise KeyError(
                "No kinematic bins found in the results. Please check the data."
            )

        # If the t_bins or energy_bins are not specified, ensure that there is only one
        # unique t_bin and energy_bin in the matching kinematic bins
        matching_t_bins = {kb.t_bin for kb in matching_kinematic_bins}
        matching_energy_bins = {kb.energy_bin for kb in matching_kinematic_bins}
        if len(matching_t_bins) > 1 or len(matching_energy_bins) > 1:
            raise ValueError(
                f"Cannot determine a mass range because multiple t or energy bins are"
                f" present and must be specified. Found t_bins={matching_t_bins} and"
                f" energy_bins={matching_energy_bins}."
            )

        return sorted(matching_kinematic_bins)

    def t_kinematic_bins(
        self,
        energy_bin: tuple[float, float] | EnergyBin | None = None,
        mass_bin: tuple[float, float] | MassBin | None = None,
    ) -> list[KinematicBin]:
        """List of t bins, optionally filtered by mass and/or energy bins.

        Args:
            mass_bin (tuple[float,float] | MassBin | None): Low and high edges of the
                mass bin to select, or a MassBin instance. Only needs specification if
                the results span multiple mass bins. If None, the sole available mass
                bin will be used.
            energy_bin (tuple[float,float] | EnergyBin | None): Low and high edges of
                the energy bin to select, or an EnergyBin instance. Only needs
                specification if the results span multiple energy bins. If None, the
                sole available energy bin will be used.

        Returns:
            list[KinematicBin]: List of t bins that match the specified mass and energy
                bins.

        Raises:
            KeyError: If no kinematic bins match the specified mass and energy bins, or
            if the results are ambiguous.
            ValueError: If multiple mass or energy bins are present and none is
                specified, making the mass bin selection ambiguous.
        """
        matching_kinematic_bins = self._resolve_kinematic_bins(
            mass_bins=mass_bin, energy_bins=energy_bin
        )
        if not matching_kinematic_bins and (
            mass_bin is not None or energy_bin is not None
        ):
            raise KeyError(
                f"No kinematic bins found matching t_bins={mass_bin} and"
                f" energy_bins={energy_bin}."
            )
        elif not matching_kinematic_bins:
            raise KeyError(
                "No kinematic bins found in the results. Please check the data."
            )

        # If the mass_bins or energy_bins are not specified, ensure that there is only
        # one unique mass_bin and energy_bin in the matching kinematic bins
        matching_mass_bins = {kb.mass_bin for kb in matching_kinematic_bins}
        matching_energy_bins = {kb.energy_bin for kb in matching_kinematic_bins}
        if len(matching_mass_bins) > 1 or len(matching_energy_bins) > 1:
            raise ValueError(
                f"Cannot determine a mass range because multiple t or energy bins are"
                f" present and must be specified. Found t_bins={matching_mass_bins} and"
                f" energy_bins={matching_energy_bins}."
            )

        return sorted(matching_kinematic_bins)

    def energy_kinematic_bins(
        self,
        t_bin: tuple[float, float] | TBin | None,
        mass_bin: tuple[float, float] | MassBin | None,
    ) -> list[KinematicBin]:
        """List of energy bins, optionally filtered by t and/or mass bins.

        Args:
            t_bin (tuple[float,float] | TBin | None): Low and high edges of the t
                bin to select, or a TBin instance. Only needs specification if the
                results span multiple t bins. If None, the sole available t bin will
                be used.
            mass_bin (tuple[float,float] | MassBin | None): Low and high edges of
                the mass bin to select, or a MassBin instance. Only needs
                specification if the results span multiple mass bins. If None, the
                sole available mass bin will be used.

        Returns:
            list[KinematicBin]: List of energy bins that match the specified t and mass
                bins.

        Raises:
            KeyError: If no kinematic bins match the specified t and mass bins, or
                if the results are ambiguous.
            ValueError: If multiple t or mass bins are present and none is
                specified, making the energy bin selection ambiguous.
        """
        matching_kinematic_bins = self._resolve_kinematic_bins(
            t_bins=t_bin, mass_bins=mass_bin
        )
        if not matching_kinematic_bins and (t_bin is not None or mass_bin is not None):
            raise KeyError(
                f"No kinematic bins found matching t_bins={t_bin} and"
                f" mass_bins={mass_bin}."
            )
        elif not matching_kinematic_bins:
            raise KeyError(
                "No kinematic bins found in the results. Please check the data."
            )

        # If the t_bins or mass_bins are not specified, ensure that there is only
        # one unique t_bin and mass_bin in the matching kinematic bins
        matching_t_bins = {kb.t_bin for kb in matching_kinematic_bins}
        matching_mass_bins = {kb.mass_bin for kb in matching_kinematic_bins}
        if len(matching_t_bins) > 1 or len(matching_mass_bins) > 1:
            raise ValueError(
                f"Cannot determine a mass range because multiple t or mass bins are"
                f" present and must be specified. Found t_bins={matching_t_bins}"
                f" and mass_bins={matching_mass_bins}."
            )

        return sorted(matching_kinematic_bins)

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
    # Production Coefficients, Covariance, and Normalization
    # ----------------------------------------------------------------------------------
    @property
    def coefficient_names(self) -> list[str]:
        """Unique production coefficient names, without their '_re'/'_im' part.

        These match the amplitude names when reactions and sums are constrained, and
        otherwise carry their '<sum>::' or '<reaction>::<sum>::' prefix.
        """
        return coefficient_bases(self.fit.columns)

    def production_coefficients(
        self,
        kinematic_bins: list[KinematicBin] | None = None,
        frame: CoefficientFrame = "fit",
    ) -> pd.DataFrame:
        """Complex production coefficients, one column per unique coefficient.

        Args:
            kinematic_bins (list[KinematicBin] | None): Optional bins to select, in the
                order returned. Defaults to None, selecting all bins.
            frame (Literal['fit', 'randomized', 'bootstrap']): Which results to take
                the coefficients from. Defaults to 'fit'.

        Returns:
            pd.DataFrame: complex128 values indexed by 'bin_id'. The 'randomized' and
                'bootstrap' frames have several rows per bin.

        Raises:
            ValueError: If the requested frame is not available.
        """
        df = self._select_rows(self._coefficient_frame(frame), kinematic_bins)
        values = {
            name: df[f"{name}_re"].to_numpy(np.float64)
            + 1j * df[f"{name}_im"].to_numpy(np.float64)
            for name in self.coefficient_names
        }
        return pd.DataFrame(values, index=pd.Index(df["bin_id"], name="bin_id"))

    def covariance_matrix(
        self,
        bin_id: str | KinematicBin,
        parameters: list[str] | None = None,
        fill_fixed: bool = False,
    ) -> pd.DataFrame:
        """The covariance matrix of the fit parameters in a single bin.

        With the default preprocessing, the matrix only contains the unique, free
        parameters, named as in the fit dataframe (e.g. 'm1p0S_re').

        Args:
            bin_id (str | KinematicBin): The bin to get the matrix for.
            parameters (list[str] | None): Optional parameters to select, in the order
                returned. Defaults to None, returning all parameters.
            fill_fixed (bool): If True, requested parameters missing from the matrix
                (i.e. fixed parameters dropped in preprocessing) get zero rows and
                columns. Defaults to False.

        Returns:
            pd.DataFrame: float64 square matrix, with parameter names as index and
                columns.

        Raises:
            ValueError: If no covariance matrices are available.
            KeyError: If the bin is not found, or a requested parameter is missing and
                'fill_fixed' is False.
        """
        if self.covariance is None:
            raise ValueError("No covariance matrices are available in these results.")
        block = self._matrix_block(self.covariance, bin_id, "parameter")
        matrix = block.astype(np.float64)

        if parameters is None:
            return matrix
        missing = [p for p in parameters if p not in matrix.index]
        if missing and not fill_fixed:
            raise KeyError(
                f"The parameters {missing} are not in the covariance matrix of"
                f" '{bin_id}'. Fixed parameters are removed in preprocessing; use"
                " fill_fixed=True to give them zero (co)variance."
            )
        return matrix.reindex(index=parameters, columns=parameters, fill_value=0.0)

    def normalization_factors(
        self, kinematic_bins: list[KinematicBin] | None = None
    ) -> pd.DataFrame:
        """Diagonal normalization of each unique production coefficient per bin.

        For a unique coefficient V_i, this is
        N_i = sum_{reaction r, sum s} scale_r^2 * Re(NI^{r,s}_ii), summing over all the
        constrained copies of V_i, where scale_r is the 'parScale_<x>' parameter of
        the reaction (1 if absent). If the off-diagonal integrals between different
        amplitudes vanish, |V_i|^2 * N_i is the intensity of amplitude i. Whether that
        intensity is acceptance corrected depends on 'norm_int_type'.

        Args:
            kinematic_bins (list[KinematicBin] | None): Optional bins to select, in the
                order returned. Defaults to None, selecting all bins.

        Returns:
            pd.DataFrame: float64 factors indexed by 'bin_id', one column per unique
                coefficient name.

        Raises:
            ValueError: If no normalization integrals are available.
            KeyError: If a coefficient has no normalization integral in a bin, or
                reactions cannot be matched to their scale parameters.
        """
        if self.norm_int is None:
            raise ValueError(
                "No normalization integrals are available in these results."
            )
        names = self.coefficient_names
        fit = self._select_rows(self.fit, kinematic_bins)

        rows = []
        for _, fit_row in fit.iterrows():
            block = self._matrix_block(self.norm_int, fit_row["bin_id"], "amplitude")
            diag = np.real(np.diag(block.to_numpy()))
            position = {amp: i for i, amp in enumerate(block.index)}
            groups = group_by_unique(block.index, names, strict=True)
            scales = self._reaction_scales(fit_row, block.index)
            rows.append(
                {
                    name: sum(
                        scales[split_full_name(f)[0]] ** 2 * diag[position[f]]
                        for f in fulls
                    )
                    for name, fulls in groups.items()
                }
            )
        return pd.DataFrame(
            rows, index=pd.Index(fit["bin_id"], name="bin_id"), dtype=np.float64
        )

    @property
    def norm_int_type(self) -> NormIntType:
        """Whether 'norm_int' holds the accepted or generated (ampInt) integrals.

        Determined by reconstructing the total intensity in each bin,
        sum_{r,s} scale_r^2 sum_ij V_i V_j^* NI^{r,s}_ij, and comparing it to the
        'intensity' (accepted) and 'ac_intensity' (acceptance corrected) columns.

        Returns:
            Literal['accepted', 'generated', 'unknown']: 'unknown' if there are no
                integrals, or the reconstruction matches neither column in all bins.
        """
        if self._norm_int_type is None:
            self._norm_int_type = self._infer_norm_int_type()
        return self._norm_int_type

    def reconstructed_intensity(self) -> pd.Series:
        """Total intensity per bin, reconstructed from coefficients and 'norm_int'.

        Computes sum_{r,s} scale_r^2 sum_ij V_i V_j^* NI^{r,s}_ij over all reactions r
        and sums s, using the unique coefficient for each constrained copy.

        Returns:
            pd.Series: float64 reconstructed intensity indexed by 'bin_id'.

        Raises:
            ValueError: If no normalization integrals are available.
            KeyError: If reactions cannot be matched to their scale parameters.
        """
        if self.norm_int is None:
            raise ValueError(
                "No normalization integrals are available in these results."
            )
        names = self.coefficient_names
        coefficients = self.production_coefficients()
        totals = {}
        for _, fit_row in self.fit.iterrows():
            bin_id = fit_row["bin_id"]
            block = self._matrix_block(self.norm_int, bin_id, "amplitude")
            full_to_name = {
                f: name
                for name, fulls in group_by_unique(block.index, names).items()
                for f in fulls
            }
            scales = self._reaction_scales(fit_row, block.index)

            # group the full amplitudes by their (reaction, sum)
            sums: dict[tuple[str | None, str | None], list[str]] = {}
            for f in block.index:
                if f in full_to_name:
                    reaction, sum_name, _ = split_full_name(f)
                    sums.setdefault((reaction, sum_name), []).append(f)

            total = 0.0
            for (reaction, _), fulls in sums.items():
                v = np.array(
                    [coefficients.loc[bin_id, full_to_name[f]] for f in fulls],
                    dtype=np.complex128,
                )
                ni = block.loc[fulls, fulls].to_numpy(np.complex128)
                total += scales[reaction] ** 2 * np.real(v @ ni @ v.conj())
            totals[bin_id] = total
        return pd.Series(totals, name="intensity", dtype=np.float64).rename_axis(
            "bin_id"
        )

    def align_conjugation(
        self,
        references: tuple[str, str, int] | list[tuple[str, str, int]],
        ambiguity_sigma: float = 1.0,
    ) -> "Results":
        """Resolve the complex conjugation ambiguity with a fixed phase sign choice.

        Within a coherent sector (amplitudes with the same reflectivity and the same
        '<reaction>::<sum>::' prefix, if any), conjugating all production coefficients
        V -> V* leaves the likelihood unchanged, so each bin's fit lands on either
        solution at random. For each reference '(amp1, amp2, sign)', every row in
        which arg(V_amp1 V_amp2^*) does not have the requested sign is conjugated for
        the whole sector of 'amp1':

        - '<amp>_im' columns are negated in the fit, randomized and bootstrap frames
        (each row decided by its own reference phase)
        - phase difference columns between amplitudes of the sector are negated
        - covariance / correlation entries of the sector's '_im' parameters are
        sign-flipped, C_ij -> s_i s_j C_ij

        Intensities are unchanged.

        Args:
            references (tuple[str, str, int] | list[tuple[str, str, int]]): One
                '(amp1, amp2, sign)' per sector, with sign +1 or -1.
            ambiguity_sigma (float): Warn about bins whose fitted reference phase is
                within this many standard deviations of 0 or 180 degrees, where the
                choice is arbitrary. Defaults to 1.

        Returns:
            Results: a new, aligned Results instance.

        Raises:
            ValueError: For unknown amplitudes, amplitudes in different sectors, two
                references for the same sector, or a sign other than +1 / -1.
        """
        if isinstance(references, tuple):
            references = [references]
        names = self.coefficient_names

        # coefficients outside any reflectivity sector (e.g. an 'isotropic'
        # background) have no conjugation partner and are never touched
        sector_of: dict[str, tuple[str, str]] = {}
        for name in names:
            try:
                sector_of[name] = self.coherent_sector(name)
            except ValueError:
                continue

        sectors: list[tuple[str, str, int, list[str]]] = []
        seen_sectors = set()
        for amp1, amp2, sign in references:
            if sign not in (1, -1):
                raise ValueError(f"The sign must be +1 or -1, got {sign}")
            for amp in (amp1, amp2):
                if amp not in names:
                    raise ValueError(
                        f"'{amp}' is not a production coefficient. Available: {names}"
                    )
            sector = self.coherent_sector(amp1)
            if self.coherent_sector(amp2) != sector:
                raise ValueError(
                    f"'{amp1}' and '{amp2}' are in different coherent sectors"
                )
            if sector in seen_sectors:
                raise ValueError(f"Two references were given for the sector {sector}")
            seen_sectors.add(sector)
            members = [n for n, s in sector_of.items() if s == sector]
            sectors.append((amp1, amp2, sign, members))

        frames = {
            "fit": self.fit.copy(),
            "randomized": None if self.randomized is None else self.randomized.copy(),
            "bootstrap": None if self.bootstrap is None else self.bootstrap.copy(),
        }
        matrices = {
            "covariance": None if self.covariance is None else self.covariance.copy(),
            "correlation": (
                None if self.correlation is None else self.correlation.copy()
            ),
        }

        for amp1, amp2, sign, members in sectors:
            sector_phases = [
                p
                for p in self.phase_differences
                if all(a in members for a in p.split("_"))
            ]
            self._warn_ambiguous_reference(amp1, amp2, ambiguity_sigma)

            flipped_fit_bins: set[str] = set()
            for label, df in frames.items():
                if df is None:
                    continue
                flip = self._conjugation_mask(df, amp1, amp2, sign)
                for m in members:
                    df.loc[flip, f"{m}_im"] = -df.loc[flip, f"{m}_im"]
                for p in sector_phases:
                    if p in df.columns:
                        flipped = -df.loc[flip, p]
                        df.loc[flip, p] = flipped.where(flipped > -180.0, 180.0)
                if label == "fit":
                    flipped_fit_bins = set(df.loc[flip, "bin_id"])

            im_names = [f"{m}_im" for m in members]
            for label, matrix in matrices.items():
                if matrix is None or not flipped_fit_bins:
                    continue
                params = matrix["parameter"].astype(str)
                is_im = params.map(
                    lambda p: any(matches_unique(str(p), n) for n in im_names)
                )
                # the frame stacks all bins, so parameter names repeat
                im_cols = list(dict.fromkeys(c for c, f in zip(params, is_im) if f))
                in_bin = matrix["bin_id"].isin(flipped_fit_bins)
                # negate rows of the '_im' parameters, then their columns. Their
                # diagonal entries are negated twice, so keep their sign.
                rows = in_bin & is_im.to_numpy()
                other_cols = [c for c in params.unique() if c not in im_cols]
                matrix.loc[rows, other_cols] = -matrix.loc[rows, other_cols]
                matrix.loc[in_bin & ~is_im.to_numpy(), im_cols] = -matrix.loc[
                    in_bin & ~is_im.to_numpy(), im_cols
                ]

        return dataclasses.replace(
            self,
            fit=frames["fit"],
            randomized=frames["randomized"],
            bootstrap=frames["bootstrap"],
            covariance=matrices["covariance"],
            correlation=matrices["correlation"],
            conjugation_references=self.conjugation_references
            + tuple((a1, a2, s) for a1, a2, s, _ in sectors),
        )

    # ----------------------------------------------------------------------------------
    # Analysis
    # ----------------------------------------------------------------------------------
    @property
    def plot(self) -> FactoryPlotter:
        """Return a FactoryPlotter instance for plotting the results."""
        if self._factory_plotter is None:
            self._factory_plotter = FactoryPlotter(self)

        return self._factory_plotter

    def get_uncertainty(
        self, label: str, kinematic_bins: list[KinematicBin] | None = None
    ) -> pd.Series:
        """Get the uncertainty for a given label

        Args:
            label (str): column in fit result get calculate uncertainty for.
            kinematic_bins (list[KinematicBin] | None, optional): Optional list of
                kinematic bins to narrow down to. Defaults to None.

        Raises:
            KeyError: If 'label' is found in neither the bootstrap dataframe or as a
                '<label>_err' column in the fit dataframe.

        Returns:
            pd.Series: Standard deviations of bootstrap samples (if exists), or
                '<label>_err' column.
        """

        if self.bootstrap is not None and label in self.bootstrap.columns:
            grouped = self.bootstrap.groupby("bin_id", sort=False)[label]

            if label in self.phase_differences:
                uncertainty = grouped.apply(self._circular_std)
            else:
                uncertainty = grouped.std()

            if kinematic_bins is not None:
                bin_ids = [kinematic_bin.bin_id for kinematic_bin in kinematic_bins]
                uncertainty = uncertainty.reindex(bin_ids)
            return uncertainty
        else:
            if f"{label}_err" not in self.fit.columns:
                raise KeyError(
                    f"{label} could not be found in bootstrap data, and '{label}_err'"
                    " does not exist in the best fit result. Could not determine"
                    f" uncertainty of {label}"
                )

            if kinematic_bins is not None:
                bin_ids = [kinematic_bin.bin_id for kinematic_bin in kinematic_bins]
                selected = self.fit.set_index("bin_id", drop=False).loc[
                    bin_ids, f"{label}_err"
                ]
                return selected.reset_index(drop=True)
            else:
                return self.fit[f"{label}_err"]

    # ----------------------------------------------------------------------------------
    # Pass-Through
    # ----------------------------------------------------------------------------------
    @staticmethod
    def set_plot_style(style: str | Path) -> None:
        """Set the global matplotlib style used by all plotters."""
        from pwassist.plotting.base import BasePWAPlotter

        BasePWAPlotter.set_style(style)

    @staticmethod
    def get_plot_style_name() -> str:
        """Get the current global matplotlib style name used by all plotters."""
        from pwassist.plotting.base import BasePWAPlotter

        return BasePWAPlotter.get_style_name()

    @staticmethod
    def get_plot_style_path() -> Path:
        """Get the current global matplotlib style path used by all plotters."""
        from pwassist.plotting.base import BasePWAPlotter

        return BasePWAPlotter.get_style_path()

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
            [e.sum() for e in self.fit[list(refl_sums)].to_numpy()]
            if refl_sums
            else None
        )
        if total_reflectivity is not None:
            detected_events = self.fit["intensity"].to_numpy()
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

    def _resolve_kinematic_bins(
        self,
        t_bins: (
            tuple[float, float] | TBin | list[tuple[float, float]] | list[TBin] | None
        ) = None,
        energy_bins: (
            tuple[float, float]
            | EnergyBin
            | list[tuple[float, float]]
            | list[EnergyBin]
            | None
        ) = None,
        mass_bins: (
            tuple[float, float]
            | MassBin
            | list[tuple[float, float]]
            | list[MassBin]
            | None
        ) = None,
    ) -> list[KinematicBin]:
        """Resolve requested kinematic bins against the bins present in results.

        Args:
            t_bins (list[tuple[float, float]] | list[TBin] | None): List of requested t
                bins, either as (low, high) tuples or TBin instances.
            energy_bins (list[tuple[float, float]] | list[EnergyBin] | None): List of
                requested energy bins, either as (low, high) tuples or EnergyBin
                instances.
            mass_bins (list[tuple[float, float]] | list[MassBin] | None): List of
                requested mass bins, either as (low, high) tuples or MassBin instances.

        Returns:
            list[KinematicBin]: List of resolved kinematic bins that match the requested
                bins. If any of the requested bin types are None, then all kinematic
                bins of that type will be selected. Otherwise, only the specified bins
                will be selected.
        Raises:
            KeyError: If any of the requested bins are not found in the results, or if
                the requested bins are invalid (e.g. low >= high).
        """
        formatted_t_bins: list[TBin] = []
        if isinstance(t_bins, tuple):
            formatted_t_bins = [TBin.from_tuple(t_bins)]
        elif isinstance(t_bins, list) and all(isinstance(tb, tuple) for tb in t_bins):
            formatted_t_bins = [TBin.from_tuple(tb) for tb in t_bins]  # type: ignore
        elif isinstance(t_bins, TBin):
            formatted_t_bins = [t_bins]

        formatted_energy_bins: list[EnergyBin] = []
        if isinstance(energy_bins, tuple):
            formatted_energy_bins = [EnergyBin.from_tuple(energy_bins)]
        elif isinstance(energy_bins, list) and all(
            isinstance(eb, tuple) for eb in energy_bins
        ):
            formatted_energy_bins = [EnergyBin.from_tuple(eb) for eb in energy_bins]  # type: ignore
        elif isinstance(energy_bins, EnergyBin):
            formatted_energy_bins = [energy_bins]

        formatted_mass_bins: list[MassBin] = []
        if isinstance(mass_bins, tuple):
            formatted_mass_bins = [MassBin.from_tuple(mass_bins)]
        elif isinstance(mass_bins, list) and all(
            isinstance(mb, tuple) for mb in mass_bins
        ):
            formatted_mass_bins = [MassBin.from_tuple(mb) for mb in mass_bins]  # type: ignore
        elif isinstance(mass_bins, MassBin):
            formatted_mass_bins = [mass_bins]

        if t_bins is not None and any(
            t not in self.unique_t_bins for t in formatted_t_bins
        ):
            raise KeyError(
                f"Requested t bins {t_bins} not found in results. Available t bins:"
                f" {[str(b) for b in self.unique_t_bins]}"
            )
        if energy_bins is not None and any(
            e not in self.unique_energy_bins for e in formatted_energy_bins
        ):
            raise KeyError(
                f"Requested energy bins {energy_bins} not found in results. Available"
                f" energy bins: {[str(b) for b in self.unique_energy_bins]}"
            )
        if mass_bins is not None and any(
            m not in self.unique_mass_bins for m in formatted_mass_bins
        ):
            raise KeyError(
                f"Requested mass bins {mass_bins} not found in results. Available mass"
                f" bins: {[str(b) for b in self.unique_mass_bins]}"
            )

        return [
            kb
            for kb in self.kinematic_bins
            if (t_bins is None or kb.t_bin in formatted_t_bins)
            and (energy_bins is None or kb.energy_bin in formatted_energy_bins)
            and (mass_bins is None or kb.mass_bin in formatted_mass_bins)
        ]

    def _coefficient_frame(self, frame: CoefficientFrame) -> pd.DataFrame:
        """Return the fit, randomized, or bootstrap dataframe by name."""
        if frame not in ("fit", "randomized", "bootstrap"):
            raise ValueError(
                f"frame must be 'fit', 'randomized', or 'bootstrap', got '{frame}'"
            )
        df = getattr(self, frame)
        if df is None:
            raise ValueError(f"No '{frame}' results are available.")
        return df

    @staticmethod
    def _select_rows(
        df: pd.DataFrame, kinematic_bins: list[KinematicBin] | None
    ) -> pd.DataFrame:
        """Select rows of 'df' in the given bins, ordered as the bins are given."""
        if kinematic_bins is None:
            return df
        order = {kb.bin_id: i for i, kb in enumerate(kinematic_bins)}
        selected = df[df["bin_id"].isin(order)]
        missing = set(order) - set(selected["bin_id"])
        if missing:
            raise KeyError(f"The bins {sorted(missing)} are not in the results.")
        positions = selected["bin_id"].map(order).to_numpy()
        return selected.iloc[np.argsort(positions, kind="stable")]

    @staticmethod
    def _matrix_block(
        df: pd.DataFrame, bin_id: str | KinematicBin, key: str
    ) -> pd.DataFrame:
        """Return one bin's square matrix, indexed and labelled by the 'key' column."""
        bin_id = bin_id.bin_id if isinstance(bin_id, KinematicBin) else bin_id
        block = df[df["bin_id"] == bin_id]
        if block.empty:
            raise KeyError(f"No matrix found for bin '{bin_id}'.")
        names = block[key].astype(str).tolist()
        matrix = block[names].set_axis(pd.Index(names, name=key), axis=0)
        matrix.columns.name = None
        return matrix

    def _reaction_scales(
        self, fit_row: pd.Series, full_names: Iterable[str]
    ) -> dict[str | None, float]:
        """Scale factor of every reaction found in 'full_names' (None maps to 1)."""
        reactions = {split_full_name(f)[0] for f in full_names} - {None}
        columns = reaction_scale_columns(self.fit.columns, sorted(reactions))  # type: ignore
        scales: dict[str | None, float] = {
            r: (1.0 if col is None else float(fit_row[col]))
            for r, col in columns.items()
        }
        scales[None] = 1.0
        return scales

    def _infer_norm_int_type(self, rtol: float = 1e-3) -> NormIntType:
        """Implementation of the 'norm_int_type' property."""
        if self.norm_int is None:
            return "unknown"
        try:
            reconstructed = self.reconstructed_intensity()
        except (KeyError, ValueError) as e:
            warnings.warn(
                f"Could not reconstruct the intensity from the normalization"
                f" integrals: {e}",
                UserWarning,
            )
            return "unknown"

        fit = self.fit.set_index("bin_id")
        reconstructed = reconstructed.reindex(fit.index).to_numpy()

        def matches(column: str) -> bool:
            return column in fit.columns and bool(
                np.allclose(
                    reconstructed, fit[column].to_numpy(np.float64), rtol=rtol, atol=0
                )
            )

        # if the acceptance is 1, both match and the integrals are interchangeable
        if matches("ac_intensity"):
            return "generated"
        if matches("intensity"):
            return "accepted"
        return "unknown"

    def coherent_sector(self, name: str) -> tuple[str, str]:
        """The coherent sector of a production coefficient.

        Coefficients in the same sector interfere, while different sectors add
        incoherently. A sector is identified by the coefficient's '<reaction>::<sum>'
        prefix (empty when constrained) and its reflectivity.

        Args:
            name (str): A unique coefficient name, e.g. '1P+0p' or 'sum::1P+0p'.

        Returns:
            tuple[str, str]: (prefix, reflectivity), e.g. ('', 'p').

        Raises:
            ValueError: If the reflectivity cannot be parsed from the name (e.g. an
                'isotropic' background term).
        """
        prefix, _, amp = name.rpartition("::")
        try:
            reflectivity = self.parser.parse_amplitude(amp).e
        except ValueError as e:
            raise ValueError(
                f"Could not determine the reflectivity of '{name}': {e}"
            ) from e
        return prefix, reflectivity

    @staticmethod
    def _conjugation_mask(
        df: pd.DataFrame, amp1: str, amp2: str, sign: int
    ) -> np.ndarray:
        """Rows whose phase arg(V_amp1 V_amp2^*) has the opposite of 'sign'."""
        v1 = df[f"{amp1}_re"].to_numpy(np.float64) + 1j * df[f"{amp1}_im"].to_numpy(
            np.float64
        )
        v2 = df[f"{amp2}_re"].to_numpy(np.float64) + 1j * df[f"{amp2}_im"].to_numpy(
            np.float64
        )
        return sign * np.angle(v1 * np.conj(v2)) < 0

    def _warn_ambiguous_reference(self, amp1: str, amp2: str, n_sigma: float) -> None:
        """Warn about bins where the reference phase is near 0 or 180 degrees."""
        phase = self.find_phase(amp1, amp2)
        if phase is None or n_sigma <= 0:
            return
        bin_map = {kb.bin_id: kb for kb in self.kinematic_bins}
        try:
            sigma = self.get_uncertainty(
                phase, [bin_map[b] for b in self.fit["bin_id"]]
            ).to_numpy(np.float64)
        except KeyError:
            return

        v1 = self.production_coefficients()[amp1].to_numpy()
        v2 = self.production_coefficients()[amp2].to_numpy()
        abs_phase = np.abs(np.rad2deg(np.angle(v1 * np.conj(v2))))
        distance = np.minimum(abs_phase, 180.0 - abs_phase)
        ambiguous = self.fit["bin_id"][distance < n_sigma * sigma].tolist()
        if ambiguous:
            warnings.warn(
                f"The reference phase ({amp1}, {amp2}) is within {n_sigma:g} sigma of"
                f" 0 or 180 degrees in the bins {ambiguous}. The conjugation choice"
                " there is arbitrary.",
                UserWarning,
            )

    def _circular_std(self, angles: pd.Series) -> float:
        """Calculate the circular standard deviation of a series of angles

        Args:
            angles (pd.Series): Series of angles (in degrees). Preprocessing should
                ensure that angles are within [-180, 180] degrees.

        Raises:
            ValueError: if angles are not within [-180, 180] degrees.

        Returns:
            float: circular standard deviation in degrees. Returns NaN if the input
                series is empty or contains only NaN values.

        Note:
            This automatically corrects for the sign ambiguity in the phase differences
            by taking the absolute value before calculating.
        """
        angles = angles.dropna()
        if len(angles) == 0:
            return np.nan
        angles_rad = np.abs(np.deg2rad(angles))  # corrected for sign ambiguity

        if angles_rad.max() > np.pi:
            raise ValueError("Data must be within [-pi, pi]; check preprocessing.")
        return np.rad2deg(scipy.stats.circstd(angles_rad, low=0, high=np.pi))
