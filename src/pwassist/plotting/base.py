import importlib.resources
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats

from pwassist.io.binning import EnergyBin, KinematicBin, MassBin, TBin

if TYPE_CHECKING:
    from pwassist.core.result import Results

# Results dataframes that share the same fit-parameter naming convention (<parameter>
# and <parameter>_err columns). The 'fit', 'randomized', and 'bootstrap' frames all have
# this shape, of one row per fit. The 'correlation'/'covariance'/'norm_int' frames are
# matrix-shaped, with one row per parameter/amplitude, not following this convention.
_FIT_LIKE_FRAMES = frozenset({"fit", "randomized", "bootstrap"})


class BasePWAPlotter:
    """Base class all sub-plotters inherit from"""

    _STYLE_DIR = importlib.resources.files("pwassist") / "plotting" / "styles"
    _DEFAULT_STYLE = "default"

    _current_style: str | Path = _DEFAULT_STYLE

    def __init__(self, results: "Results"):
        self.results = results

    # ----------------------------------------------------------------------------------
    # Pass throughs
    # ----------------------------------------------------------------------------------
    @property
    def fit(self) -> pd.DataFrame:
        return self.results.fit

    @property
    def data(self) -> pd.DataFrame:
        return self.results.data

    @property
    def correlation(self) -> pd.DataFrame | None:
        return self.results.correlation

    @property
    def covariance(self) -> pd.DataFrame | None:
        return self.results.covariance

    @property
    def norm_int(self) -> pd.DataFrame | None:
        return self.results.norm_int

    @property
    def randomized(self) -> pd.DataFrame | None:
        return self.results.randomized

    @property
    def bootstrap(self) -> pd.DataFrame | None:
        return self.results.bootstrap

    @property
    def _mass_centers(self) -> list[float]:
        return self.results.get_mass_centers()

    @property
    def _mass_bin_width(self) -> float:
        return self.results.get_average_mass_bin_width()

    # ----------------------------------------------------------------------------------
    # Style Management
    # ----------------------------------------------------------------------------------
    @classmethod
    def available_styles(cls) -> list[str]:
        """List names of built-in styles available for use with set_style()."""
        return sorted(
            p.name.removesuffix(".mplstyle")
            for p in cls._STYLE_DIR.iterdir()
            if p.is_file() and p.name.endswith(".mplstyle")
        )

    @classmethod
    def get_style_name(cls) -> str:
        """Return name or path of the current style used by all plotters"""
        return str(cls._current_style)

    @classmethod
    def get_style_path(cls) -> Path:
        """Return the path to the current style used by all plotters"""
        if isinstance(cls._current_style, Path):
            return cls._current_style
        else:
            return Path(cls._resolve_style_path())

    @classmethod
    def set_style(cls, style: str | Path) -> None:
        """Set global matplotlib style used by all plotters

        Args:
            style (str | Path): Either a built-in style name (from available_styles())
                or a path to a custom .mplstyle file.

        Raises:
            ValueError: If the style is not a valid built-in style or a valid file path.
        """
        if isinstance(style, str) and style in cls.available_styles():
            cls._current_style = style
            return

        path = Path(style)
        if not path.is_file():
            raise ValueError(
                f"Style '{style}' not found. Use an existing file path or one of the"
                " available styles."
                f" Available styles: {cls.available_styles()}"
            )
        cls._current_style = path

    @classmethod
    def _resolve_style_path(cls) -> str:
        """Resolve the current style to a file path for use with plt.style.context()"""
        if isinstance(cls._current_style, Path):
            return str(cls._current_style)
        else:
            return str(cls._STYLE_DIR / f"{cls._current_style}.mplstyle")

    # ----------------------------------------------------------------------------------
    # Shared Helpers
    # ----------------------------------------------------------------------------------
    def get_bootstrap_uncertainty(
        self, label: str, kinematic_bins: list[KinematicBin] | None = None
    ) -> pd.Series:
        """Get the uncertainty for a given label calculated from the bootstrap data

        Args:
            label (str): label that should match a column in the bootstrap dataframe
            kinematic_bins (list[KinematicBin] | None, optional): Optional list of
                kinematic bins to narrow the calculation down to. Defaults to None.

        Raises:
            KeyError: If bootstrap fits are unavailable, or label is not in the
                bootstrap frame.

        Returns:
            pd.Series: Standard deviations of the bootstrap samples, indexed by
                kinematic bin.
        """

        if self.bootstrap is None:
            raise KeyError("Bootstrap results are not available in the results.")

        if label not in self.bootstrap.columns:
            raise KeyError(f"Label '{label}' not found in bootstrap results.")

        grouped = self.bootstrap.groupby("bin_id", sort=False)[label]

        if label in self.results.phase_differences:
            uncertainty = grouped.apply(self._circular_std)
        else:
            uncertainty = grouped.std()  # Standard deviation as error estimate

        if kinematic_bins is not None:
            bin_ids = [kinematic_bin.bin_id for kinematic_bin in kinematic_bins]
            uncertainty = uncertainty.reindex(bin_ids)
        return uncertainty

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

    def _resolve_kinematic_bins(
        self,
        t_bin: tuple[float, float] | TBin | None = None,
        energy_bin: tuple[float, float] | EnergyBin | None = None,
        mass_bin: tuple[float, float] | MassBin | None = None,
        indices: list[int] | None = None,
    ) -> list[KinematicBin]:
        """Resolve the sorted list of kinematic bins matching the given filters

        Args:
            t_bin (tuple[float, float] | TBin | None, optional): Optional t bin filter.
                Defaults to None.
            energy_bin (tuple[float, float] | EnergyBin | None, optional): Optional
                energy bin filter. Defaults to None.
            mass_bin (tuple[float, float] | MassBin | None, optional): Optional mass
                bin filter. Defaults to None.
            indices (list[int] | None, optional): Optional list of positions within the
                filtered, sorted list of kinematic bins to further select. Defaults to
                None (all bins included).

        Returns:
            list[KinematicBin]: Resolved, sorted (and possibly index-selected) kinematic
                bins.
        """
        results = self.results
        if any(b is not None for b in (t_bin, energy_bin, mass_bin)):
            results = results.filter_by_kinematic_bins(
                t_bins=t_bin, energy_bins=energy_bin, mass_bins=mass_bin
            )
        kinematic_bins = sorted(results.kinematic_bins)
        if indices is not None:
            kinematic_bins = [kinematic_bins[i] for i in indices]
        return kinematic_bins

    def _resolve_single_bin(
        self,
        t_bin: tuple[float, float] | TBin | None = None,
        energy_bin: tuple[float, float] | EnergyBin | None = None,
        mass_bin: tuple[float, float] | MassBin | None = None,
        indices: list[int] | None = None,
    ) -> KinematicBin:
        """Resolve exactly one kinematic bin from the given filters.

        Used by the singular BinPlotter, to select one TEM bin. If multiple bins are
        still present, optionally indices can resolve it to one.

        Args:
            t_bin (tuple[float, float] | TBin | None, optional): Fixes the t bin.
            Unnecessary if results span only one t bin. Defaults to None.
            energy_bin (tuple[float, float] | EnergyBin | None, optional): Fixes the
                energy bin. Unnecessary if results span only one energy bin. Defaults to
                None.
            mass_bin (tuple[float, float] | MassBin | None, optional): Fixes the mass
                bin. Unnecessary if results span only one mass bin. Defaults to None.
            indices (list[int] | None, optional): Optional list of positions within the
                filtered, sorted list of kinematic bins to narrow down to a singular
                bin. Defaults to None

        Returns:
            KinematicBin: The single resolved kinematic bin
        Raises:
            ValueError: If zero, or more than one, kinematic bin matches the filters.
        """
        candidates = self._resolve_kinematic_bins(t_bin, energy_bin, mass_bin, indices)
        if len(candidates) != 1:
            raise ValueError(
                f"Expected precisely one kinematic bin to be resolved, but"
                f" {len(candidates)} matched the given filters:"
                f" {[kb.bin_id for kb in candidates]}. Specify"
                f" t_bin/energy_bin/mass_bin (with further optional index if necessary)"
                f" to narrow down to a single bin."
            )
        return candidates[0]

    def _frame_columns(
        self,
        frame: Literal[
            "fit", "correlation", "covariance", "norm_int", "randomized", "bootstrap"
        ],
        columns: tuple[str, ...] | list[str] | None,
    ) -> tuple[pd.DataFrame, list[str]]:
        """Look up a results dataframe and resolve which columns to use.

        'fit' / 'randomized' / 'bootstrap' frames share the same '<col>'/'<col>_err'
        style column-name convention, so 'columns' gets their '_err' companions added
        automatically (when available). The matrix-shaped frames (e.g. 'correlation')
        use columns as-is.

        Args:
            frame (Literal['fit', 'correlation', 'covariance', 'norm_int',
                'randomized', 'bootstrap']): Which `results` dataframe to look up.
            columns (tuple[str, ...] | list[str] | None): The column names to include.
                If None, every column is used.

        Returns:
            tuple[pd.DataFrame, list[str]]: The requested dataframe itself, and the
                resolved list of column names to select from it.
        """
        requested_frame = getattr(self.results, frame, None)
        if requested_frame is None:
            raise KeyError(
                f"results.{frame} is not available. Make sure the results bundle"
                f" actually includes {frame} data."
            )

        if columns is None:
            frame_columns = [c for c in requested_frame.columns if c != "bin_id"]
        elif frame in _FIT_LIKE_FRAMES:
            frame_columns = list(columns) + [
                f"{col}_err"
                for col in columns
                if f"{col}_err" in requested_frame.columns
            ]
        else:
            frame_columns = list(columns)
        return requested_frame, frame_columns

    def _select_by_bin_id(
        self, frame: pd.DataFrame, bin_ids: list[str], columns: list[str]
    ) -> pd.DataFrame:
        """Select and order rows of results dataframe by resolved bin_ids

        'fit' and 'data' have one row per kinematic bin, but 'randomized' and
        'bootstrap' have many (one row per fit, many fits per bin). 'Correlation',
        'covariance', and 'norm_int' frames are matrices, with one row per
        fit/amplitude, and many rows per bin. This method selects by kinematic bin_id,
        keeping the original ordering of many rows per bin.

        Args:
            frame (pd.DataFrame): Dataframe to select from. Must have a "bin_id" column.
            bin_ids (list[str]): Resolved, ordered bin_ids to select and order by.
            columns (list[str]): Columns to keep, in addition to "bin_id", which is kept
                to keep rows identifiable.

        Returns:
            pd.DataFrame: Selected rows with 'bin_id' retained as the first column.

        Raises:
            KeyError: If "bin_id" is not a column in 'frame'
        """
        if "bin_id" not in frame.columns:
            raise KeyError(
                "Expected a 'bin_id' column to select kinematic bins by, but was not"
                f" found. Available columns: {list(frame.columns)}"
            )
        ordered_columns = ["bin_id"] + [c for c in columns if c != "bin_id"]
        selected = frame.set_index("bin_id", drop=False).loc[bin_ids, ordered_columns]
        return selected.reset_index(drop=True)

    def _replace_errors_with_bootstrap(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Replaces the 'frame's error columns by the bootstrap standard deviations

        If no bootstrap dataframe is available, it silently returns the original frame.
        This function is only really useful for the 'best' self.fit frame.

        Args:
            frame (pd.DataFrame): input dataframe

        Returns:
            pd.DataFrame: dataframe copy, with all '<param>_err' columns replaced by the
                standard deviation of the bootstrap samples for <param>.
        """
        if self.bootstrap is None:
            return frame

        updated_frame = frame.copy()
        bin_ids = frame["bin_id"]
        for error_column in (c for c in frame.columns if c.endswith("_err")):
            label = error_column.removesuffix("_err")
            uncertainty = self.get_bootstrap_uncertainty(label)
            updated_frame[error_column] = uncertainty.reindex(bin_ids).to_numpy()
        return updated_frame

    def _style(self):
        """Context manager to apply the current style for plotting"""
        return plt.style.context(self._resolve_style_path())
