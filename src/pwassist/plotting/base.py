import importlib.resources
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats

if TYPE_CHECKING:
    from pwassist.core.result import Results


class BasePWAPlotter:
    """Base class all sub-plotters inherit from"""

    _STYLE_DIR = importlib.resources.files("pwassist") / "plotting" / "styles"
    _DEFAULT_STYLE = Path("default")

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
    def get_bootstrap_error(self, label: str) -> pd.Series:
        """Get the bootstrap error for a given label from the fit dataframe."""

        if self.bootstrap is None:
            raise ValueError("Bootstrap results are not available in the results.")

        if label not in self.bootstrap.columns:
            raise KeyError(f"Label '{label}' not found in bootstrap results.")

        grouped = self.bootstrap.groupby("bin_id")[label]

        if label in self.results.phase_differences:
            return grouped.apply(self._circular_std)

        return grouped.std()  # Standard deviation as error estimate

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
        """
        angles = angles.dropna()
        if len(angles) == 0:
            return np.nan
        angles_rad = np.abs(np.deg2rad(angles))  # corrected for sign ambiguity

        if angles_rad.max() > np.pi:
            raise ValueError("Data must be within [-pi, pi]; check preprocessing.")
        return np.rad2deg(scipy.stats.circstd(angles_rad, low=0, high=np.pi))

    def _style(self):
        """Context manager to apply the current style for plotting"""
        return plt.style.context(self._resolve_style_path())
