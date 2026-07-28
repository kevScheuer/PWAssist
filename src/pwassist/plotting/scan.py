import itertools
from typing import Any

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from uncertainties import ufloat, unumpy

from pwassist.plotting.base import BasePWAPlotter


class ScanPlotter(BasePWAPlotter):
    """Plotter for all results that scan across a range of bins, e.g. mass, t, etc."""

    def coherent_sum(
        self,
        sum_label: str,
        data_legend: str = "GlueX-I Data",
        indices: list[int] | None = None,
        ax: matplotlib.axes.Axes | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> matplotlib.axes.Axes:
        """Plot coherent sum group across the mass bins with data points.

        A sum group is defined by the quantum numbers that the sum groups and the
        amplitude naming scheme. For example, in the `JLme` scheme, one can plot all
        the available `JLe` coherent sums, or those that sum over the spin-projection
        `m`.

        Args:
            sum_label (str): The label of the coherent sum group to plot. See the
                `coherent_sums` member of the `Results` class for a list of available
                sum groups and the amplitudes that belong to each group.
            data_legend (str): The legend label for the data points. Defaults to
                "GlueX-I Data".
            indices (list[int] | None): Optional list of indices to select specific mass
                bins. If None, all bins will be plotted.
            ax (matplotlib.axes.Axes | None): Optional axes to plot on. If None, a new
                figure and axes will be created.
            kwargs (dict[str, Any] | None): Optional dictionary of keyword arguments
                to customize the plot appearance.

        Raises:
            KeyError: If the specified sum_label is not found in the coherent sums.
                Prints available sum labels.
        """
        if sum_label not in self.results.coherent_sums:
            raise KeyError(
                f"Sum label '{sum_label}' not found in coherent sums."
                f" Available sum labels: {list(self.results.coherent_sums.keys())}"
            )

        coherent_sums = self.results.coherent_sums[sum_label]
        fit_df, data_df = self._coherent_sum_dataframes(coherent_sums, indices)

        # default to Dark2 colormap, and cycle if more columns than colors
        colors = plt.get_cmap("Dark2").colors  # type: ignore
        colors = list(itertools.islice(itertools.cycle(colors), len(coherent_sums)))

        kwargs = {
            "marker": ["." for _ in range(len(coherent_sums))],
            "linestyle": ["" for _ in range(len(coherent_sums))],
            "alpha": [0.7 for _ in range(len(coherent_sums))],
            "colors": colors,
        }
        kwargs.update(kwargs or {})

        with self._style():
            fig, ax = (
                plt.subplots(layout="constrained")
                if ax is None
                else (ax.get_figure(), ax)
            )

            # plot the data points with error bars, using the appropriate events column
            # based on whether the results are acceptance-corrected or not
            if self.results.is_acc_corrected:
                data_points = unumpy.uarray(
                    data_df["ac_events"], data_df["ac_events_err"]
                )
            else:
                data_points = unumpy.uarray(data_df["events"], data_df["events_err"])

            ax.errorbar(
                x=data_df["m_center"],
                xerr=data_df["bin_width"] / 2.0,
                y=unumpy.nominal_values(data_points),
                yerr=unumpy.std_devs(data_points),
                label=data_legend,
                marker=".",
                linestyle="",
                color="black",
            )

            # plot each coherent sum with error bars
            for sum_idx, coh_sum in enumerate(coherent_sums):
                label = self.results.parser.sum_to_latex(sum_label, coh_sum)

                ax.errorbar(
                    x=data_df["m_center"],
                    xerr=data_df["bin_width"] / 2.0,
                    y=fit_df[coh_sum],
                    yerr=fit_df[f"{coh_sum}_err"],
                    label=label,
                    marker=kwargs["marker"][sum_idx],
                    linestyle=kwargs["linestyle"][sum_idx],
                    alpha=kwargs["alpha"][sum_idx],
                    color=kwargs["colors"][sum_idx],
                )

            ax.set_xlabel(r"Mass $(GeV)$")
            ax.set_ylabel(rf"Events / {data_df['bin_width'].mean():.3f} GeV")
            ax.set_ylim(bottom=0)
            ax.legend()

        return ax

    def _coherent_sum_dataframes(
        self, columns: tuple[str, ...], indices: list[int] | None = None
    ) -> tuple[pd.DataFrame | pd.Series, pd.DataFrame]:
        """Prepare the dataframes for the coherent_sum plot

        Args:
            columns (list[str]): The list of column names to include in the dataframes.
            indices (list[int] | None): Optional list of indices to select specific mass
                bins.
        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the fit dataframe and
                the data dataframe for the specified coherent sum group and indices.
        Todo:
            - this can potentially be generalized to other bin plotters
        """

        # bootstrap fits replace fit errors, if available
        fit_columns = list(columns)
        if self.results.bootstrap is not None:
            raise NotImplementedError(
                "Replacing column errors by bootstrap std() not yet supported"
            )
        else:
            fit_columns.extend([f"{col}_err" for col in fit_columns])

        fit_df = (
            self.results.fit.loc[indices, fit_columns]
            if indices is not None
            else self.results.fit[fit_columns]
        )

        data_columns = [
            "m_center",
            "m_low",
            "m_high",
            "events",
            "events_err",
            "ac_events",
            "ac_events_err",
        ]

        data_df = (
            self.results.data.loc[indices, data_columns]
            if indices is not None
            else self.results.data[data_columns]
        )
        data_df["bin_width"] = data_df["m_high"] - data_df["m_low"]

        return fit_df, data_df
