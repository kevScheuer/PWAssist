import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pwassist.plotting.base import BasePWAPlotter


class BinPlotter(BasePWAPlotter):
    """Plotter for analyzing a single bin of data, e.g. mass, t, etc."""

    def corr_matrix(
        self, bin_idx: int, ax: matplotlib.axes.Axes | None = None
    ) -> matplotlib.axes.Axes:
        """Plot the correlation matrix of the fit parameters for a single bin.

        Args:
            bin_idx (int): The index of the bin to plot.
            ax (matplotlib.axes.Axes | None): Optional axes to plot on. If None, a new
                figure and axes will be created.
        Returns:
            matplotlib.axes.Axes: The axes object containing the correlation matrix plot
        """

        # TODO: determine bin_id of fit_result associated with correlation df bin_id
        # then get the correlation matrix for that bin_id from the fit_result

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        with self._style():
            # TODO: plot corr matrix
            pass

        return ax

    def randomized_summary(
        self,
        bin_idx: int,
        columns: list[str],
        likelihood_threshold: float = np.inf,
        ignore_failed_fits: bool = True,
        ignore_bad_error_matrix: bool = True,
        axs: np.ndarray | None = None,
    ) -> np.ndarray:
        """Plot a 2x2 summary plot of the randomized fit results in a bin

        This produces a figure with 4 subplots:
        - Upper left: histogram of the likelihood values for the randomized fits
        - Upper right: Weighted residuals of moments (to be implemented)
        - Bottom left: Weighted residuals of requested columns
        - Bottom right: Average absolute weighted residuals of requested columns
            vs moments. If moments are unavailable, then it will be vs the change in
            likelihood from the best fit.

        The likelihood difference is defined to be:
            delta_likelihood = likelihood_randomized - likelihood_best_fit

        Args:
            bin_idx (int): The index of the bin to plot.
            columns (list[str]): The columns of the randomized fit results to plot.
            likelihood_threshold (float, optional): Randomized fits with
                delta_likelihood greater than this value will be ignored. Defaults to np.inf.
            ignore_failed_fits (bool, optional): Ignores randomized fits that did not
                converge (lastMinuitCommandStatus != 0). Defaults to True.
            ignore_bad_error_matrix (bool, optional): Ignores randomized fits with
                bad error matrices (eMatrixStatus != 3). Defaults to True.

        Returns:
            np.ndarray: A 2x2 array of the axes objects containing the subplots.
        Raises:
            KeyError: If the randomized fit results are not available or if any of the
                requested columns are not found in the randomized fit results.
        """

        if self.results.randomized is None:
            raise KeyError("Randomized fit results are not available.")

        for col in columns:
            if col not in self.results.randomized.columns:
                raise KeyError(f"Column '{col}' not found in randomized fit results.")

        if axs is None:
            fig, axs = plt.subplots(2, 2)

        # TODO: implement the plotting of the 4 subplots as described in the docstring

        return axs  # type: ignore

    def pairplot(
        self,
        bin_indices: list[int],
        columns: list[str],
        correlation_threshold: float = 0.7,
    ) -> sns.PairGrid:
        """Create a comprehensive pairplot of bootstrap fit results

        Args:
            bin_indices (list[int]): The indices of the bins to include in the pairplot.
            columns (list[str]): The columns of the bootstrap fit results to include in
                the pairplot.
            correlation_threshold (float, optional): The threshold for highlighting
                plots with high correlation. Defaults to 0.7.

        Returns:
            sns.PairGrid: The seaborn PairGrid object containing the pairplot.

        Raises:
            KeyError: If the bootstrap fit results are not available or if any of the
                requested columns are not found in the bootstrap fit results.
        """

        if self.results.bootstrap is None:
            raise KeyError("Bootstrap fit results are not available.")

        for col in columns:
            if col not in self.results.bootstrap.columns:
                raise KeyError(f"Column '{col}' not found in bootstrap fit results.")

        # TODO: implement the pairplot creation using seaborn's PairGrid,
        # filtering by bin_indices and columns, and highlighting based on
        # correlation_threshold
        pg = sns.PairGrid(pd.DataFrame())

        return pg
