import importlib.resources

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats

from pwassist.core.result import Results


class BasePWAPlotter:
    """Base class all sub-plotters inherit from"""

    _STYLE_PATH = str(
        importlib.resources.files("pwassist")
        / "plotting"
        / "styles"
        / "default.mplstyle"
    )

    def __init__(self, results: Results, channel: str = r"\pi^0\pi^-"):
        self.results = results
        self.channel = channel

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
    def _mass_centers(self) -> list[float]:
        return self.results.get_mass_centers()

    @property
    def _mass_bin_width(self) -> float:
        return self.results.get_average_mass_bin_width()

    # ----------------------------------------------------------------------------------
    # Shared Helpers
    # ----------------------------------------------------------------------------------
    def _get_pretty_label(self, label: str) -> str:
        # TODO: based off naming scheme, this will convert quantum number-based label
        # string into LaTeX style string for plotting. For example
        # "1S+0p" -> "1^{+}S_{0}^{+}"
        raise NotImplementedError("_get_pretty_label not yet implemented")

    def get_bootstrap_error(self, label: str) -> pd.Series:
        """Get the bootstrap error for a given label from the fit dataframe."""
        raise NotImplementedError("get_bootstrap_error not yet implemented")

        if self.bootstrap is None:
            raise ValueError("Bootstrap results are not available in the results.")

        if label not in self.boostrap.columns:
            raise KeyError(f"Label '{label}' not found in bootstrap results.")

        if label in self.phase_differences:
            return grouped.apply(self._circular_std)

        grouped = self.bootstrap.groupby("bin_id")[label]
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
        return plt.style.context(self._STYLE_PATH)
