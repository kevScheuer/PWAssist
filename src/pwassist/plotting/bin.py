import warnings
from collections.abc import Sequence
from typing import Any, Literal

import matplotlib.axes
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.special import erf
from uncertainties import ufloat

from pwassist.io.binning import EnergyBin, KinematicBin, MassBin, TBin
from pwassist.plotting.base import BasePWAPlotter

# fit-parameter name suffixes that get rendered as a readable "(Re)"/"(Im)" tag
# rather than shown raw, e.g. "1S+0p_re" -> "$\Re(1S_{0}^{(+)})$"
_PARAMETER_PART_LABELS = {"re": "Re", "im": "Im"}


class BinPlotter(BasePWAPlotter):
    """Plotter for analyzing a single bin of data, e.g. mass, t, etc."""

    def corr_matrix(
        self,
        t_bin: tuple[float, float] | TBin | None = None,
        energy_bin: tuple[float, float] | EnergyBin | None = None,
        mass_bin: tuple[float, float] | MassBin | None = None,
        indices: list[int] | None = None,
        columns: list[str] | None = None,
        exclude_columns: list[str] | None = None,
        ax: matplotlib.axes.Axes | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> matplotlib.axes.Axes:
        """Plot the correlation matrix of the fit parameters for a single bin.

        This internally determines which parameters, specifically production
        coefficients, are unique. In the correlation dataframe the production
        coefficients are written in their 'full' amplitude form
        '<reaction>::<sum>::<amplitude>_<part>', but they are often times constrained
        across reactions and sums. To reduce the number of identical correlations being
        reported for constrained cases, only those coefficients who are unique will be
        plotted (assuming no explicit 'columns' were requested).

        Args:
            t_bin (tuple[float, float] | TBin | None): Fixes the t bin to plot from if
                the results span multiple t bins. If only 1 t bin is available,
                specification is unnecessary. Defaults to None.
            energy_bin (tuple[float,float] | EnergyBin | None): Fixes the beam energy
                bin to plot from if the results span multiple energy bins. If only 1
                energy bin is available, specification is unnecessary. Defaults to None.
            mass_bin (tuple[float,float] | EnergyBin | None): Fixes the mass bin to plot
                from if the results span multiple mass bins. If only 1 mass bin is
                available, specification is unnecessary. Defaults to None.
            indices (list[int] | None): Optional list of positions within the resolved
                kinematic bin to select specific bins. Defaults to None.
            columns (list[str] | None): Optional list of parameter columns to plot.
                Defaults to None, so that all unique parameters are plotted.
            exclude_columns (list[str] | None): Optional list of parameter columns to
                exclude, as sometimes it is easier to remove a few columns from many,
                than to explicitly request the many. Defaults to None, so that all
                unique parameters are plotted.
            ax (matplotlib.axes.Axes | None): Optional axes to plot on. If None, a new
                figure and axes will be created.
            kwargs (dict[str, Any] | None): Optional dictionary of keyword arguments
                to customize the plot appearance. Passed directly to 'seaborn.heatmap'
        Returns:
            matplotlib.axes.Axes: The axes object containing the correlation matrix plot
        """

        value_df, kinematic_bin = self._bin_dataframe(
            frame="correlation",
            t_bin=t_bin,
            energy_bin=energy_bin,
            mass_bin=mass_bin,
            indices=indices,
        )
        if "parameter" not in value_df.columns:
            raise KeyError(
                f"Expected a 'parameter' column in the correlation dataframe, but it"
                f" was not found. Available columns: {list(value_df.columns)}"
            )
        if not columns:
            parameters = self._filter_production_coefficients(
                value_df["parameter"].tolist()
            )
            parameters = [
                p
                for p in parameters
                if exclude_columns is None or p not in exclude_columns
            ]
        else:
            parameters = [
                p
                for p in columns
                if exclude_columns is None or p not in exclude_columns
            ]

        matrix = (
            value_df.set_index("parameter")
            .loc[parameters]
            .reindex(columns=parameters)
            .to_numpy()
        )
        labels = [
            self._parameter_label(
                p,
                self.results.are_reactions_constrained,
                self.results.are_sums_constrained,
            )
            for p in parameters
        ]

        default_kwargs = {
            "cmap": "coolwarm",
            "vmin": -1.0,
            "vmax": 1.0,
            "center": 0.0,
            "square": True,
            "annot": len(parameters) <= 15,
            "fmt": ".2f",
            "cbar_kws": {"label": "Correlation"},
            "xticklabels": labels,
            "yticklabels": labels,
        }
        default_kwargs.update(kwargs or {})
        kwargs = default_kwargs

        with self._style():
            fig, ax = (
                plt.subplots(layout="constrained")
                if ax is None
                else (ax.get_figure(), ax)
            )

            sns.heatmap(matrix, ax=ax, **kwargs)
            ax.set_title(self._bin_title(kinematic_bin), fontsize="small")
            ax.tick_params(axis="x", labelrotation=90)
            ax.tick_params(axis="y", labelrotation=0)

        return ax

    def production_coefficients(
        self,
        source: Literal["randomized", "bootstrap"] = "randomized",
        delta_lnL_threshold: float = np.inf,
        ignore_failed_fits: bool = True,
        ignore_bad_matrix: bool = True,
        columns: list[str] | None = None,
        t_bin: tuple[float, float] | TBin | None = None,
        energy_bin: tuple[float, float] | EnergyBin | None = None,
        mass_bin: tuple[float, float] | MassBin | None = None,
        indices: list[int] | None = None,
        axs: np.ndarray | None = None,
        kwargs: dict[str, dict[str, Any]] | None = None,
    ) -> np.ndarray:
        """Real and imaginary components of production coefficients for many fits

        Given a 'source' dataframe (randomized or bootstrap), each production
        coefficient is individually plotted with its real and imaginary components as
        a scatter plot, where all fits in a selected kinematic bin are plotted.

        Args:
            source (Literal['randomized', 'bootstrap'], optional): source dataframe to
                pull fits from. Defaults to "randomized".
            delta_lnL_threshold (float). Fits (i) with
                Δ(-2lnL_i - -2lnL_min) < threshold will be plotted. Defaults to np.inf,
                so all fits are included.
            ignore_failed_fits (bool): Does not plot any fits with
                lastMinuitCommandStatus != 0. Defaults to True.
            ignore_bad_matrix (bool): Does not plot any fits with eMatrixStatus != 3.
                Defaults to True.
            columns (list[str] | None, optional): select production coefficient columns
                to plot. Expects them to end with '_re' or '_im' parts. Defaults to
                None.
            t_bin (tuple[float, float] | TBin | None): Fixes the t bin to plot from if
                the results span multiple t bins. If only 1 t bin is available,
                specification is unnecessary. Defaults to None.
            energy_bin (tuple[float,float] | EnergyBin | None): Fixes the beam energy
                bin to plot from if the results span multiple energy bins. If only 1
                energy bin is available, specification is unnecessary. Defaults to None.
            mass_bin (tuple[float,float] | EnergyBin | None): Fixes the mass bin to plot
                from if the results span multiple mass bins. If only 1 mass bin is
                available, specification is unnecessary. Defaults to None.
            indices (list[int] | None): Optional list of positions within the resolved
                kinematic bin to select specific bins. Defaults to None.
            axs (np.ndarray | None, optional): Optional array of axes to plot on. Ensure
                that there are enough positions available for the number of production
                coefficients to plot. Defaults to None.
            kwargs (dict[str, dict[str], Any]] | None, optional): Optional dictionary of
                keyword arguments, for each production coefficient, to customize plot
                appearances. Recognizes 'amplitude' keys, which are the
                '<amplitude>_<part>' components in production coefficients e.g.
                '0S+1p_re'. The accompanying dict are the keyword arguments that will be
                used for that amplitude's plot. Defaults to None.

        Raises:
            KeyError: If source dataframe unrecognized
            ValueError: 'columns' does not have recognized production coefficient
                format.

        Returns:
            np.ndarray: square array of production coefficient real and imaginary parts.
        """

        if source not in ("randomized", "bootstrap"):
            raise KeyError(
                "Expected 'source' to either be the 'randomized' or 'bootstrap'"
                " dataframes."
            )

        value_df, kinematic_bin = self._bin_dataframe(
            frame=source,
            t_bin=t_bin,
            energy_bin=energy_bin,
            mass_bin=mass_bin,
            indices=indices,
        )
        delta_lnL = self._delta_lnL(value_df["likelihood"])

        # mask values according to function parameters
        mask = self._fit_quality_mask(
            value_df, delta_lnL_threshold, ignore_failed_fits, ignore_bad_matrix
        )

        value_df = value_df[mask]
        delta_lnL = delta_lnL[mask]

        if not columns:
            columns = [
                c for c in value_df.columns if c.endswith("_re") or c.endswith("_im")
            ]
        else:
            for c in columns:
                if not c.endswith("_re") and not c.endswith("_im"):
                    raise ValueError(
                        f"User provided columns '{c}' not a recognized production"
                        " coefficient."
                    )
        columns = sorted(columns)

        # pair all the re and im column names together (if present)
        pairs: set[tuple[str, str]] = set()
        for c in columns:
            base, part = c.split("_")
            if part == "re":
                re_col = c
                im_col = f"{base}_im" if f"{base}_im" in columns else ""
            elif part == "im":
                re_col = f"{base}_re" if f"{base}_re" in columns else ""
                im_col = c
            else:
                raise ValueError("unrecognized production coefficient component")

            pairs.add((re_col, im_col))

        # map amplitude names to complex series
        production_coeffs: dict[str, pd.Series[complex]] = {}
        for pair in pairs:
            base_amp_name = pair[0].split("_")[0] if pair[0] else pair[1].split("_")[0]
            re_series: pd.Series[float] = (
                value_df[pair[0]]
                if pair[0]
                else pd.Series(np.full(len(value_df[pair[1]]), 0.0, dtype=float))
            )
            im_series: pd.Series[float] = (
                value_df[pair[1]]
                if pair[1]
                else pd.Series(np.full(len(value_df[pair[0]]), 0.0, dtype=float))
            )
            production_coeffs[base_amp_name] = re_series + 1j * im_series

        # build colormap for delta(likelihoods)

        cmap = plt.get_cmap("cividis")
        norm = matplotlib.colors.Normalize(
            vmin=np.min(delta_lnL), vmax=np.max(delta_lnL)
        )
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        # ---Plot---
        with self._style():
            fig, axs = (
                plt.subplots(
                    int(np.ceil(np.sqrt(len(production_coeffs)))),
                    int(np.ceil(np.sqrt(len(production_coeffs)))),
                    layout="constrained",
                )
                if axs is None
                else (axs[0, 0].gcf(), axs)
            )
            assert axs is not None

            for i, (amp, series) in enumerate(production_coeffs.items()):
                ax = axs.flatten()[i]

                default_kwargs = {
                    "c": np.asarray(delta_lnL),
                    "cmap": cmap,
                    "norm": norm,
                    "s": 3,
                }
                if kwargs is not None and amp in kwargs:
                    default_kwargs.update(kwargs[amp] or {})

                complex_values = np.asarray(series, dtype=np.complex128)
                ax.scatter(complex_values.real, complex_values.imag, **default_kwargs)
                try:
                    label = self.results.parser.to_latex(amp)
                except ValueError:
                    label = amp
                ax.set_xlabel(rf"$\Re({label})$", loc="center")
                ax.set_ylabel(rf"$\Im({label})$", loc="center")

            # hide unused axes
            for ax in axs.flatten()[len(production_coeffs) :]:
                ax.set_visible(False)

            fig.suptitle(self._bin_title(kinematic_bin))
            fig.colorbar(
                sm,
                ax=axs,
                location="right",
                shrink=0.8,
                pad=0.02,
                label=r"$\Delta(-2\ln(\mathcal{L}))$",
            )

        return axs

    def likelihood_scatter(
        self,
        delta_lnL_threshold: float = np.inf,
        ignore_failed_fits: bool = True,
        ignore_bad_matrix: bool = True,
        columns: list[str] | None = None,
        t_bin: tuple[float, float] | TBin | None = None,
        energy_bin: tuple[float, float] | EnergyBin | None = None,
        mass_bin: tuple[float, float] | MassBin | None = None,
        indices: list[int] | None = None,
        axs: np.ndarray | None = None,
        kwargs: dict[str, dict[str, Any]] | None = None,
    ) -> np.ndarray:
        """Scatterplot of randomized fits compared to the nominal, colored by likelihood

        By default, this plots a grid of Δp, where p are the amplitudes and phase
        differences, and Δ is calculated as the difference between the randomized fit
        result and the nominal result from the 'fit' dataframe. Δp is on the y-axis,
        while the x-axis is the index associated with the sorted Δ-2lnL values. This
        allows one to view how the parameters converge (or don't) as the likelihood
        approaches the best solution.

        Args:
            delta_lnL_threshold (float). Fits (i) with
                Δ(-2lnL_i - -2lnL_min) < threshold will be plotted. Defaults to np.inf,
                so all fits are included.
            ignore_failed_fits (bool): Does not plot any fits with
                lastMinuitCommandStatus != 0. Defaults to True.
            ignore_bad_matrix (bool): Does not plot any fits with eMatrixStatus != 3.
                Defaults to True.
            columns (list[str] | None, optional): select results columns to plot.
                Defaults to None, meaning all amplitude intensities and phase
                differences will be plotted.
            t_bin (tuple[float, float] | TBin | None): Fixes the t bin to plot from if
                the results span multiple t bins. If only 1 t bin is available,
                specification is unnecessary. Defaults to None.
            energy_bin (tuple[float,float] | EnergyBin | None): Fixes the beam energy
                bin to plot from if the results span multiple energy bins. If only 1
                energy bin is available, specification is unnecessary. Defaults to None.
            mass_bin (tuple[float,float] | EnergyBin | None): Fixes the mass bin to plot
                from if the results span multiple mass bins. If only 1 mass bin is
                available, specification is unnecessary. Defaults to None.
            indices (list[int] | None): Optional list of positions within the resolved
                kinematic bin to select specific bins. Defaults to None.
            axs (np.ndarray | None, optional): Optional array of axes to plot on. Ensure
                that there are enough positions available for the number of production
                coefficients to plot. Defaults to None.
            kwargs (dict[str, dict[str], Any]] | None, optional): Optional dictionary of
                keyword arguments, for each column, to customize plot appearances. Keys
                are the column string, and the accompanying dicts are the keyword
                arguments that will be used for that amplitude's scatter plot. Defaults
                to None.

        Raises:
            KeyError: if the 'randomized' dataframe is unavailable, or non-existent
                columns are requested.

        Returns:
            np.ndarray: square array of scatter plots
        """

        if self.randomized is None:
            raise KeyError("No 'randomized' dataframe to plot results from")

        value_df, kinematic_bin = self._bin_dataframe(
            frame="randomized",
            t_bin=t_bin,
            energy_bin=energy_bin,
            mass_bin=mass_bin,
            indices=indices,
        )
        best_df, _ = self._bin_dataframe(  # 'best' fit values to compare against
            frame="fit",
            t_bin=t_bin,
            energy_bin=energy_bin,
            mass_bin=mass_bin,
            indices=indices,
        )
        delta_lnL = self._delta_lnL(value_df["likelihood"])

        # mask values according to function parameters
        mask = self._fit_quality_mask(
            value_df, delta_lnL_threshold, ignore_failed_fits, ignore_bad_matrix
        )

        value_df = value_df[mask]
        delta_lnL = delta_lnL[mask]

        # sort delta_lnL values for plots later
        sort_idx = delta_lnL.argsort()
        sorted_delta_lnL = delta_lnL[sort_idx]

        # build colormap
        cmap = plt.get_cmap("cividis")
        norm = matplotlib.colors.Normalize(
            vmin=np.min(delta_lnL), vmax=np.max(delta_lnL)
        )
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        if not columns:
            columns = list(self.results.amplitudes + self.results.phase_differences)
        else:
            missing_cols = [c for c in columns if c not in value_df.columns]
            if missing_cols:
                raise KeyError(
                    f"The columns '{missing_cols}' do not exist. Available columns:"
                    f" {value_df.columns}"
                )

        with self._style():
            fig, axs = (
                plt.subplots(
                    int(np.ceil(np.sqrt(len(columns)))),
                    int(np.ceil(np.sqrt(len(columns)))),
                    layout="constrained",
                )
                if axs is None
                else (axs[0, 0].gcf(), axs)
            )
            assert axs is not None

            for i, par in enumerate(columns):
                ax = axs.flatten()[i]

                rand_par_val = value_df[par].to_numpy()
                # match length of rand array
                best_par_val = np.full(len(rand_par_val), best_df[par])

                # calculate the Delta = best par - rand par
                if par in self.results.phase_differences:
                    vec_circ_res = np.vectorize(self._circular_residual)
                    delta_par = vec_circ_res(best_par_val, rand_par_val)
                else:
                    delta_par = best_par_val - rand_par_val

                # sort from low -> high delta_lnL
                delta_par = delta_par[sort_idx]

                # plot as function of the sorted indices, so that fits with similar
                # likelihoods aren't overlapping, and we can check that parameters
                # diverge as index increases (larger delta_lnL)
                default_kwargs = {
                    "c": np.asarray(delta_lnL),
                    "cmap": cmap,
                    "norm": norm,
                    "s": 3,
                }
                if kwargs is not None and par in kwargs:
                    default_kwargs.update(kwargs[par] or {})

                ax.scatter(
                    [j for j in range(len(delta_par))],
                    delta_par,
                    marker="o",
                    **default_kwargs,
                )

                try:
                    label = self.results.parser.to_latex(par)
                except ValueError:
                    label = par
                ax.set_ylabel(rf"$\Delta({label})$", loc="center")

            # hide unused axes
            for ax in axs.flatten()[len(columns) :]:
                ax.set_visible(False)

            fig.suptitle(self._bin_title(kinematic_bin))
            fig.colorbar(
                sm,
                ax=axs,
                location="right",
                shrink=0.8,
                pad=0.02,
                label=r"$\Delta(-2\ln(\mathcal{L}))$",
            )

        return axs

    def pairplot(
        self,
        columns: list[str],
        correlation_threshold: float = 0.7,
        sigmas: Sequence[float] = (1.0, 2.0, 3.0),
        fit_fractions: bool = True,
        show_uncertainty_bands: bool = True,
        normalize_axis_limits: bool = True,
        ignore_failed_fits: bool = True,
        ignore_bad_matrix: bool = True,
        t_bin: tuple[float, float] | TBin | None = None,
        energy_bin: tuple[float, float] | EnergyBin | None = None,
        mass_bin: tuple[float, float] | MassBin | None = None,
        indices: list[int] | None = None,
        kwargs: dict[str, dict[str, Any]] | None = None,
    ) -> sns.PairGrid:

        # parameter validation
        if self.results.bootstrap is None:
            raise KeyError("Bootstrap fit results are required to create a pairplot")

        columns = list(dict.fromkeys(columns))
        if len(columns) < 2:
            raise ValueError("A pairplot needs at least two distinct columns")
        missing_cols = [c for c in columns if c not in self.results.bootstrap.columns]
        if missing_cols:
            raise KeyError(
                f"The columns `{missing_cols}' are missing in the bootstrap results."
                f" Available columns: {list(self.results.bootstrap.columns)}"
            )
        if not 0.0 <= correlation_threshold <= 1.0:
            raise ValueError(
                f"correlation_threshold must be in [0, 1], got {correlation_threshold}"
            )
        if len(sigmas) == 0 or any(s <= 0 for s in sigmas):
            raise ValueError(f"sigmas must be positive numbers, got {sigmas}")
        unknown_keys = set(kwargs or {}) - {"grid", "diag", "lower", "upper"}
        if unknown_keys:
            raise ValueError(
                f"Unrecognized kwargs keys {sorted(unknown_keys)}. Recognized keys are"
                " 'grid', 'diag', 'lower', and 'upper'."
            )

        # get our dataframes of interest
        boot_df, kinematic_bin = self._bin_dataframe(
            frame="bootstrap",
            t_bin=t_bin,
            energy_bin=energy_bin,
            mass_bin=mass_bin,
            indices=indices,
        )
        best_df, _ = self._bin_dataframe(  # for 'best' fit values and MINUIT errors
            frame="fit",
            t_bin=t_bin,
            energy_bin=energy_bin,
            mass_bin=mass_bin,
            indices=indices,
        )
        boot_df = boot_df[
            self._fit_quality_mask(
                boot_df,
                ignore_failed_fits=ignore_failed_fits,
                ignore_bad_matrix=ignore_bad_matrix,
            )
        ]

        # convert amplitudes and coherent sums to fit fractions (if requested)
        intensity_col = "ac_intensity" if self.results.is_acc_corrected else "intensity"
        intensities = set(self.results.amplitudes).union(
            *self.results.coherent_sums.values()
        )
        fraction_columns = (
            [c for c in columns if c in intensities] if fit_fractions else []
        )
        if fraction_columns:
            for name, df in (("bootstrap", boot_df), ("fit", best_df)):
                if intensity_col not in df.columns:
                    raise KeyError(
                        f"Fit fractions need the total intensity '{intensity_col}', but"
                        f" it is not in the {name} results. Use fit_fractions=False to"
                        f" plot the intensities instead"
                    )
        plot_df = boot_df[columns].astype(float)
        for c in fraction_columns:
            plot_df[c] = plot_df[c] / boot_df[intensity_col].astype(float)

        # filter out nan/inf results, and error if too few results
        plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna()
        if len(plot_df) < 3:
            raise ValueError(
                f"Only {len(plot_df)} usable bootstrap fits remain in"
                f" {kinematic_bin.bin_id} after filtering, but at least 3 are needed."
            )

        # Drop any fixed parameters and warn the user
        constant_cols = [c for c in columns if plot_df[c].nunique() < 2]
        if constant_cols:
            warnings.warn(
                f"Dropping the columns {constant_cols} from the pairplot, since they"
                f" take a single value in every bootstrap fit (are fixed)",
                UserWarning,
            )
            columns = [c for c in columns if c not in constant_cols]
            plot_df = plot_df[columns]
        if len(columns) < 2:
            raise ValueError(
                "Fewer than two of the requested columns vary between bootstrap fits"
            )

        def _value_and_error(column: str) -> tuple[float, float]:
            # best fit values, and MINUIT errors, in the same units as the plot
            error_col = f"{column}_err"
            return (
                float(best_df[column].iloc[0]),
                float(best_df[error_col].iloc[0]) if error_col in best_df else np.nan,
            )

        best: dict[str, tuple[float, float]] = {}
        for c in columns:
            if c not in best_df.columns:
                continue
            value, error = _value_and_error(c)
            if c in fraction_columns:
                fraction = ufloat(value, error) / ufloat(
                    *_value_and_error(intensity_col)
                )  # type: ignore
                value, error = fraction.nominal_value, fraction.std_dev
            best[c] = (value, error)

        # label the coherent sums
        labels: dict[str, str] = {}
        for c in columns:
            sum_label = next(
                (k for k, v in self.results.coherent_sums.items() if c in v), None
            )
            try:
                labels[c] = rf"${self.results.parser.to_latex(c, sum_label)}$"
            except (ValueError, KeyError, IndexError):
                labels[c] = c

        color = "tab:blue"
        band_color = "tab:orange"
        default_kwargs: dict[str, dict[str, Any]] = {
            "grid": {"height": 2.2, "diag_sharey": False},
            "diag": {"color": color, "fill": True, "alpha": 0.5},
            "lower": {"color": color, "s": 8, "alpha": 0.5, "linewidth": 0},
            "upper": {"color": color, "linewidths": 1.2},
        }
        for panel, panel_kwargs in default_kwargs.items():
            panel_kwargs.update((kwargs or {}).get(panel) or {})
        kwargs = default_kwargs

        # now get to plottin'
        with self._style():
            pg = sns.PairGrid(plot_df, vars=columns, **kwargs["grid"])
            pg.map_diag(sns.kdeplot, **kwargs["diag"])
            pg.map_lower(sns.scatterplot, **kwargs["lower"])
            pg.map_upper(self._sigma_contours, sigmas=sigmas, **kwargs["upper"])

            corr = plot_df.corr()
            any_highlighted = False
            for row, y_col in enumerate(columns):
                for col, x_col in enumerate(columns):
                    ax = pg.axes[row, col]

                    if show_uncertainty_bands:
                        # every panel gets x-band, only off-diagonals get y-band
                        spans = [(x_col, ax.axvspan, ax.axvline)]
                        if row != col:
                            spans.append((y_col, ax.axhspan, ax.axhline))
                        for name, span, line in spans:
                            if name not in best:
                                continue
                            value, error = best[name]
                            if np.isfinite(error):
                                span(
                                    value - error,
                                    value + error,
                                    color=band_color,
                                    alpha=0.25,
                                    linewidth=0,
                                    zorder=0,
                                )
                            line(value, color=band_color, linewidth=1.0, zorder=2)

                    if row != col and abs(corr.iloc[row, col]) > correlation_threshold:  # type: ignore
                        any_highlighted = True
                        for spine in ax.spines.values():
                            spine.set_visible(True)
                            spine.set_edgecolor("black")
                            spine.set_linewidth(2.5)

            if normalize_axis_limits:
                for i, c in enumerate(columns):
                    if c in fraction_columns:
                        limits = (0.0, 1.0)
                    elif c in self.results.phase_differences:
                        limits = (-180.0, 180.0)
                    else:
                        continue
                    pg.axes[0, i].set_xlim(limits)  # x is shared along a column
                    pg.axes[i, 0].set_ylim(limits)  # y is shared along a row

            for i, c in enumerate(columns):
                pg.axes[-1, i].set_xlabel(labels[c], loc="center")
                pg.axes[i, 0].set_ylabel(labels[c], loc="center")

            handles: list[Line2D | Patch] = [
                Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="",
                    markersize=4,
                    color=kwargs["lower"].get("color", color),
                    alpha=0.6,
                    label=f"Bootstrap fits (N={len(plot_df)})",
                ),
                Line2D(
                    [],
                    [],
                    color=kwargs["upper"].get("color", color),
                    label=rf"${', '.join(f'{s:g}' for s in sigmas)}\sigma$ contours",
                ),
            ]
            if show_uncertainty_bands:
                handles.append(
                    Patch(
                        facecolor=band_color,
                        alpha=0.25,
                        label=r"best fit $\pm$ MINUIT",
                    )
                )
            if any_highlighted:
                handles.append(
                    Patch(
                        facecolor="None",
                        edgecolor="black",
                        linewidth=2.5,
                        label=rf"$|\rho| > {correlation_threshold:g}$",
                    )
                )
            pg.figure.legend(
                handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5)
            )
            pg.figure.suptitle(self._bin_title(kinematic_bin), y=1.02)
        return pg

    def bootstrap_convergence(
        self,
        min_samples: int = 10,
        columns: list[str] | None = None,
        exclude_columns: list[str] | None = None,
        t_bin: tuple[float, float] | TBin | None = None,
        energy_bin: tuple[float, float] | EnergyBin | None = None,
        mass_bin: tuple[float, float] | MassBin | None = None,
        indices: list[int] | None = None,
        axs: np.ndarray | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Plot all parameters stdevs / MINUIT error as a function bootstrap fit #

        This is a diagnostic plot to see if the bootstrap fit standard deviation (used
        as the uncertainty estimate on the fit parameters) has converged as a function
        of the number of bootstrap fits. If the stdev is still changing significantly
        as the number of fits increases, then more fits are needed, or the distribution
        should be examined for non-Gaussian behavior.

        Args:
            min_samples (int): Minimum number of bootstrap fits to start calculating
                the standard deviation. Defaults to 10.
            columns (list[str] | None): Optional list of parameter columns to plot.
                Defaults to None, which plots all columns that have a corresponding
                '<param>_err' column in the bootstrap dataframe (but not the
                '<param>_err' columns themselves).
            exclude_columns (list[str] | None): Optional list of parameter columns to
                exclude from the plot. Defaults to None, which will exclude all
                '<param>_err' columns.
            t_bin (tuple[float, float] | TBin | None): Fixes the t bin to plot from if
                the results span multiple t bins. If only 1 t bin is available,
                specification is unnecessary. Defaults to None.
            energy_bin (tuple[float,float] | EnergyBin | None): Fixes the beam energy
                bin to plot from if the results span multiple energy bins. If only 1
                energy bin is available, specification is unnecessary. Defaults to None.
            mass_bin (tuple[float,float] | EnergyBin | None): Fixes the mass bin to plot
                from if the results span multiple mass bins. If only 1 mass bin is
                available, specification is unnecessary. Defaults to None.
            indices (list[int] | None): Optional list of positions within the resolved
                kinematic bin to select specific bins. Defaults to None.
            ax (matplotlib.axes.Axes | None): Optional array of axes to plot on. Ensure
                that there are enough positions available for the number of parameters
                to plot. Defaults to None, creating a figure with hspace=0 and a shared
                x-axis.
            kwargs (dict[str, Any] | None): Optional dictionary of keyword arguments
                to customize the plot appearance. Passed directly to all subplots.
                Defaults to None.

        Raises:
            KeyError: if bootstrap fis are unavailable, or requested column is missing.

        Returns:
            matplotlib.axes.Axes: The axes object containing the plot.
        """

        if self.results.bootstrap is None:
            raise KeyError(
                "Bootstrap fit results are not available. Cannot plot convergence."
            )

        value_df, kinematic_bin = self._bin_dataframe(
            frame="bootstrap",
            t_bin=t_bin,
            energy_bin=energy_bin,
            mass_bin=mass_bin,
            indices=indices,
        )
        best_df, _ = self._bin_dataframe(  # 'best' fit values to get MINUIT error from
            frame="fit",
            t_bin=t_bin,
            energy_bin=energy_bin,
            mass_bin=mass_bin,
            indices=indices,
            replace_errors_with_bootstrap=False,
        )

        if exclude_columns is None:
            exclude_columns = [c for c in value_df.columns if c.endswith("_err")]

        # remove any columns that are entirely NaN or zero (fixed parameter)
        exclude_columns += [
            c.removesuffix("_err")
            for c in value_df.columns
            if not value_df[c].to_numpy().any()
        ]

        if columns is None:
            parameters = [
                c
                for c in value_df.columns
                if c not in exclude_columns and f"{c}_err" in value_df.columns
            ]
        else:
            missing_cols = [c for c in columns if c not in value_df.columns]
            if missing_cols:
                raise KeyError(
                    f"The columns '{missing_cols}' do not exist. Available columns:"
                    f" {value_df.columns}"
                )
            parameters = columns

        with self._style():
            if axs is None:
                fig, axs = plt.subplots(
                    len(parameters),
                    1,
                    sharex=True,
                    figsize=(10, int(1.5 * len(parameters))),
                )
                fig.subplots_adjust(hspace=0)
                fig.tight_layout()
                fig.subplots_adjust(top=0.96)
            else:
                fig, axs = axs[0, 0].gcf(), axs

            assert axs is not None

            for idx, ax in enumerate(axs.flatten()):
                par = parameters[idx]
                best_par_err = best_df[f"{par}_err"].iloc[0]
                # start at 10 bootstrap samples
                if par in self.results.phase_differences:
                    std_devs = (
                        value_df[par]
                        .expanding(min_periods=min_samples)
                        .apply(lambda x: self._circular_std(x), raw=False)
                    )
                else:
                    std_devs = value_df[par].expanding(min_periods=min_samples).std()

                default_kwargs = {"legend": False}
                default_kwargs.update(kwargs or {})

                (std_devs / best_par_err).plot(ax=ax, **default_kwargs)
                try:
                    if any([par in v for v in self.results.coherent_sums.values()]):
                        sum_label = next(
                            k for k, v in self.results.coherent_sums.items() if par in v
                        )
                        label = rf"${self.results.parser.to_latex(par, sum_label)}$"
                    else:
                        label = rf"${self.results.parser.to_latex(par)}$"
                except ValueError:
                    label = par

                ax.set_xlim(min_samples, len(value_df))
                ax.set_ylabel(
                    label,
                    rotation=45,
                    ha="right",
                )
                ax.set_xlabel("Number of Bootstrap Fits", loc="center")
                ax.ticklabel_format(axis="x", useOffset=False, style="plain")

            fig.suptitle(self._bin_title(kinematic_bin))

        return axs

    # ----------------------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------------------

    def _bin_dataframe(
        self,
        columns: tuple[str, ...] | list[str] | None = None,
        frame: Literal[
            "fit", "correlation", "covariance", "norm_int", "randomized", "bootstrap"
        ] = "fit",
        t_bin: tuple[float, float] | TBin | None = None,
        energy_bin: tuple[float, float] | EnergyBin | None = None,
        mass_bin: tuple[float, float] | MassBin | None = None,
        indices: list[int] | None = None,
        replace_errors_with_bootstrap: bool = True,
    ) -> tuple[pd.DataFrame, KinematicBin]:
        """Select rows from one results dataframe for a single, resolve kinematic bin.

        Single bin analog of 'ScanPlotter._scan_dataframes', where the same
        bin-resolution and frame selection process occurs, but is resolved down to
        one kinematic bin.

        Args:
            columns (tuple[str, ...] | list[str] | None, optional): Columns to include
                from 'frame'. For 'fit'/'randomized'/'bootstrap' frames, the '_err'
                companions are automatically added, if available. Defaults to None, so
                all columns are included.
            frame (Literal['fit', 'correlation', 'covariance', 'norm_int', 'randomized',
                'bootstrap']): Which 'results' dataframe to pull 'columns' from.
                Defaults to 'fit'.
            t_bin (tuple[float, float] | TBin | None): Optional fixed t bin.
                Unnecessary if results span only one t bin. Defaults to None.
            energy_bin (tuple[float, float] | EnergyBin | None): Optional fixed energy
                bin. Unnecessary if results span only one energy bin. Defaults to None.
            mass_bin (tuple[float, float] | MassBin | None): Optional fixed mass bin.
                Unnecessary if results span only one mass bin. Defaults to None.
            indices (list[int] | None, optional): Optional list of positions within the
                filtered, sorted list of kinematic bins to narrow down to a singular
                bin. Defaults to None
            replace_error_with_bootstrap (bool): If True, the 'fit' frame's '_err'
                columns will be replaced with the bootstrap stdevs, if available.
                Defaults to True.

        Returns:
            tuple[pd.DataFrame, KinematicBin]: The requested 'frame's rows for the
                resolved bin ("bin_id" retained, though should be a constant since only
                one bin should be returned), and the resolved KinematicBin itself.

        Raises:
            ValueError: If the given filters don't resolve to exactly one kinematic bin.
            KeyError: If the requested frame is not available on 'results'.
        """
        kinematic_bin = self._resolve_single_bin(t_bin, energy_bin, mass_bin, indices)
        requested_frame, frame_columns = self._frame_columns(frame, columns)
        value_df = self._select_by_bin_id(
            requested_frame, [kinematic_bin.bin_id], frame_columns
        )
        if frame == "fit" and replace_errors_with_bootstrap:
            value_df = self._replace_errors_with_bootstrap(value_df)
        return value_df, kinematic_bin

    def _filter_production_coefficients(self, parameters: list[str]) -> list[str]:
        """Removes production coefficients with duplicate information

        Production coefficients are named
        '<reaction>::<sum>::<amplitude>_re' or'<reaction>::<sum>::<amplitude>_im'. The
        production coefficients in the fit dataframe already tell us which parts are
        constrained e.g. columns named '<amplitude>_<part>' means reactions and sums
        are constrained, while '<reaction>::<sum>::<amplitude>_<part>' means neither are
        constrained and no duplicate information is present. This function removes those
        production coefficients that will provide repeated information

        If they are constrained across reactions or sums, then many will be duplicates
        and unnecessarily plotted. This removes the duplicate parameters using result's
        metadata to provide the minimum number of parameters needed.

        For example, if 'pi1::sumA::my_amp' and 'pi2::sumB::my_amp' exists, and both
        reactions and sums are constrained

        Args:
            parameters (list[str]): List of all AmpTools parameters from a fit,
                including production coefficients

        Returns:
            list[str]: the minimum set of production coefficients needed to fully
                describe the fit, and all other passed non-production coefficient
                parameters
        """

        full_amplitudes = []
        other_params = []
        for p in parameters:
            if "::" in p:
                full_amplitudes.append(p)
            else:
                other_params.append(p)

        unique_prod_coefficients = [
            c for c in self.results.fit.columns if "_re" in c or "_im" in c
        ]
        used_unique = []
        reduced_prod_coefficients = []
        for upc in unique_prod_coefficients:
            for fa in full_amplitudes:
                if upc in fa:
                    # the first full amplitude that contains the unique component is
                    # added. All other full amplitudes therefore are repeats.
                    reduced_prod_coefficients.append(fa)
                    used_unique.append(upc)
                    break
            if upc not in used_unique:
                raise KeyError(
                    f"The production coefficient {upc} has no corresponding 'full'"
                    " amplitude name"
                )

        return reduced_prod_coefficients + other_params

    def _parameter_label(
        self, parameter: str, drop_reaction_label: bool, drop_sum_label: bool
    ) -> str:
        """Render a raw fit-parameter as a readable label.

        Individual production coefficients are named
        '<reaction>::<sum>::<amplitude>_re' or'<reaction>::<sum>::<amplitude>_im'. This
        function will render the amplitude part in LaTeX via the results' parser and
        label the Re/Im part. The sum labels are dropped, as it is assumed
        the amplitude name carries the necessary information. The reaction label can be
        optionally kept, in the case that amplitudes are not constrained across
        reactions. Non-production coefficients are returned as-is.

        Args:
            parameter (str): Raw fit-parameter name, as it appears in the 'parameter'
                column of correlation/covariance dataframes.
            drop_reaction_label (bool): the '<reaction>::' string is dropped
            drop_sum_label (bool): the '<sum>::' string is dropped.

        Returns:
            str: A readable label for the parameter, or original string if the name
                does not match the expected '<amplitude>_<part>' format
        """
        if "::" not in parameter:
            return parameter

        reaction, sum, amp_name = parameter.split("::")
        reaction = "" if drop_reaction_label else f"{reaction}::"
        sum = "" if drop_sum_label else f"{sum}::"

        base, sep, part = amp_name.rpartition("_")
        if sep and part.lower() in _PARAMETER_PART_LABELS:
            try:
                amp_label = self.results.parser.to_latex(base)
            except (ValueError, KeyError):
                pass
            else:
                return (
                    rf"{reaction}{sum}"
                    rf"$\{_PARAMETER_PART_LABELS[part.lower()]}({amp_label})$"
                )
        return parameter

    def _bin_title(self, kinematic_bin: KinematicBin) -> str:
        """Build a descriptive title from a single bin's mass/t/energy range.

        Args:
            kinematic_bin (KinematicBin): The kinematic bin to describe.

        Returns:
            str: A LaTeX-formatted title with the bin's mass, t, and energy ranges.
        """
        m, t, e = (
            kinematic_bin.mass_bin,
            kinematic_bin.t_bin,
            kinematic_bin.energy_bin,
        )
        return (
            rf"${t.low:.3f} < -t < {t.high:.3f}\ GeV^2$,"
            rf" ${e.low:.2f} < E_{{\gamma}} < {e.high:.2f}\ GeV$,"
            rf" ${m.low:.3f} < M < {m.high:.3f}\ GeV$"
        )

    def _fit_quality_mask(
        self,
        df: pd.DataFrame,
        delta_lnL_threshold: float = np.inf,
        ignore_failed_fits: bool = True,
        ignore_bad_matrix: bool = True,
    ) -> np.ndarray:
        """Mask of fits that pass likelihood difference and other cuts

        Args:
            df (pd.DataFrame): Dataframe with one row per fit, from a single bin, so
                that likelihoods are comparable
            delta_lnL_threshold (float, optional): Keep fits with
                Δ(-2lnL_i - -2lnL_min) <= threshold. Defaults to np.inf, so no fits
                are removed.
            ignore_failed_fits (bool, optional): Remove fits with
                lastMinuitCommandStatus != 0 (abnormal termination). Defaults to True.
            ignore_bad_matrix (bool, optional): Remove fits with eMatrixStatus != 3
                (not 'full and accurate'). Defaults to True.

        Raises:
            KeyError: If a column needed for a requested cut is not in 'df'

        Returns:
            np.ndarray: boolean mask, True for fits to keep
        """

        needed = ["likelihood"]
        if ignore_failed_fits:
            needed.append("lastMinuitCommandStatus")
        if ignore_bad_matrix:
            needed.append("eMatrixStatus")
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise KeyError(
                f"Cannot apply fit quality cuts, since {missing} are missing from the"
                " dataframe. Set ignore_failed_fits/ignore_bad_matrix to False to skip"
                " the status cuts"
            )

        mask = self._delta_lnL(df["likelihood"]) <= delta_lnL_threshold
        if ignore_failed_fits:
            mask &= (df["lastMinuitCommandStatus"] == 0).to_numpy()
        if ignore_bad_matrix:
            mask &= (df["eMatrixStatus"] == 3).to_numpy()
        return mask

    def _sigma_contours(
        self,
        x: pd.Series | np.ndarray,
        y: pd.Series | np.ndarray,
        sigmas: Sequence[float] = (1.0, 2.0, 3.0),
        bw_method: str | float | None = None,
        grid_size: int = 128,
        color: Any = None,
        ax: matplotlib.axes.Axes | None = None,
        **kwargs: Any,
    ) -> None:

        ax = plt.gca() if ax is None else ax
        xy = np.vstack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])

        try:
            kde = scipy.stats.gaussian_kde(xy, bw_method=bw_method)
        except np.linalg.LinAlgError:
            warnings.warn(
                f"Skipping the 2D KDE contours of '{getattr(x, 'name', 'x')} vs"
                f" '{getattr(y, 'name', 'y')}': the samples are perfectly"
                f" (anti-)correlated so no KDE can be built",
                UserWarning,
            )
            return
        # density at each sample without its own kernel, K_H(0) = 1 / (2 pi sqrt(det H))
        n_samples = xy.shape[1]
        self_kernel = 1.0 / (2.0 * np.pi * np.sqrt(np.linalg.det(kde.covariance)))
        loo_density = (n_samples * kde(xy) - self_kernel) / (n_samples - 1)

        coverage = erf(np.asarray(sigmas, dtype=float) / np.sqrt(2.0))
        levels = np.unique(np.quantile(loo_density, 1.0 - coverage))
        levels = levels[levels > 0]
        if len(levels) == 0:
            return

        # make grid wide enough for outermost contour to close
        pad = 3.0 * np.sqrt(np.diag(kde.covariance))
        low, high = xy.min(axis=1) - pad, xy.max(axis=1) + pad
        grid_x, grid_y = np.meshgrid(
            np.linspace(low[0], high[0], grid_size),
            np.linspace(low[1], high[1], grid_size),
        )
        density = kde(np.vstack([grid_x.ravel(), grid_y.ravel()])).reshape(grid_x.shape)

        # levels ascend in density, so outermost contour is first
        rgb = matplotlib.colors.to_rgba("C0" if color is None else color)[:3]
        colors = [(*rgb, alpha) for alpha in np.linspace(0.4, 1.0, len(levels))]
        ax.contour(grid_x, grid_y, density, levels=levels, colors=colors, **kwargs)
