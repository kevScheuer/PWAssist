from typing import Any, Literal

import matplotlib.axes
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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
        mask = delta_lnL <= delta_lnL_threshold
        if ignore_failed_fits:
            mask &= value_df["lastMinuitCommandStatus"] == 0
        if ignore_bad_matrix:
            mask &= value_df["eMatrixStatus"] == 3

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
                ax.set_xlabel(rf"$\Re$({label})", loc="center")
                ax.set_ylabel(rf"$\Im$({label})", loc="center")

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
        if frame == "fit":
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
                return rf"{reaction}{sum}$\{_PARAMETER_PART_LABELS[part.lower()]}$({amp_label})"
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
            rf" ${m.low:.3f} < M < {m.high:.3f}\ GeV$,"
        )

    def _delta_lnL(
        self, likelihoods: list[float] | pd.Series | np.ndarray
    ) -> np.ndarray:
        """Compute Δ(-2lnL_i - -2lnL_min)

        Args:
            likelihoods (list[float] | pd.Series | np.ndarray): set of likelihoods from
                fit results

        Returns:
            np.ndarray: minimum likelihood subtracted from all elements.

        Note:
            This comparison is only valid for a set of likelihoods belonging to the same
                fit model and underlying data.
        """
        if isinstance(likelihoods, pd.Series):
            likelihoods = likelihoods.to_numpy()
        elif isinstance(likelihoods, list):
            likelihoods = np.array(likelihoods)
        return likelihoods - np.min(likelihoods)
