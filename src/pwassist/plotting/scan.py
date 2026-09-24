import itertools
from typing import Any, Literal

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from uncertainties import ufloat, unumpy

from pwassist.io.binning import EnergyBin, KinematicBin, MassBin, TBin
from pwassist.parser import SCHEMES, NamingScheme
from pwassist.plotting.base import BasePWAPlotter

# Different naming schemes use different reflectivity characters
_POSITIVE_REFLECTIVITY_CHARS = frozenset({"p", "+"})
_NEGATIVE_REFLECTIVITY_CHARS = frozenset({"n", "m", "-"})

# Ordering of orbital angular momentum letters
_L_ORDER = "SPDFGHIKLM"

# Shorthand kinematic variables and their axis labels
_KIN_VARIABLE_XLABELS: dict[str, str] = {
    "m": r"Mass $(GeV)$",
    "t": r"$-t$ $(GeV^2)$",
    "e": r"Beam Energy $(GeV)$",
}
_KIN_VARIABLE_YLABELS: dict[str, str] = {
    "m": r"Events / NUM $(GeV)$",  # replace NUM with bin width in functions
    "t": r"Events / NUM $(GeV^2)$",
    "e": r"Events / NUM $(GeV)$",
}


def _numeric_sort_key(value: str) -> float:
    """Numeric ordering for quantum-number character or string.

    Args:
        value (str): Quantum-number string to derive sort key from
    Returns:
        float: for sorting the string
    """
    if value in _L_ORDER:
        return float(_L_ORDER.index(value))

    sign_map = {"p": 1.0, "m": -1.0, "n": -1.0}

    if value and value[0] in sign_map and (value[1:] == "" or value[1:].isdigit()):
        magnitude = float(value[1:]) if value[1:] else 1.0
        return sign_map[value[0]] * magnitude

    try:
        return float(value)
    except ValueError:
        return 0.0


def _format_spin_projection(m: str) -> str:
    """Render spin-projection as a signed integer string

    Args:
        m (str): spin-projection e.g. "-1", "+0", "p2", "m", etc.
    Returns:
        str: signed integer e.g. "m1" -> "-1", or original if uninterpretable
    """
    sign_map = {"p": "+", "m": "-", "n": "-"}
    if m and m[0] in sign_map:
        digits = m[1:] or "0"
        return f"{sign_map[m[0]]}{digits}"
    return m


class ScanPlotter(BasePWAPlotter):
    """Plotter for all results that scan across a range of bins, e.g. mass, t, etc.

    Most methods will have an optional 'kin_variable' that specifies with kinematic
    quantity (mass, t, energy, or some other 'data' dataframe column) they want to
    scan across. This is paired with a 'stat' argument determining the bin centroid
    and edges, and optional t/energy/mass bin selections in case the results bundle
    spans multiple bins for the non-plotted scan variable.
    """

    def coherent_sum(
        self,
        sum_label: str,
        data_legend: str = "GlueX-I Data",
        fractional: bool = False,
        kin_variable: str = "m",
        stat: Literal["edges", "avg"] = "edges",
        t_bin: tuple[float, float] | TBin | None = None,
        energy_bin: tuple[float, float] | EnergyBin | None = None,
        mass_bin: tuple[float, float] | MassBin | None = None,
        indices: list[int] | None = None,
        ax: matplotlib.axes.Axes | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> matplotlib.axes.Axes:
        """Plot coherent sum group across the bins.

        A sum group is defined by the quantum numbers that the sum groups together and
        the amplitude naming scheme. For example, in the `JLme` scheme, one can plot all
        the available `JLe` coherent sums i.e. those that sum over the spin-projection
        `m`. If not plotting as fit fractions (fractional='false'), then data points are
        also plotted.

        Args:
            sum_label (str): The label of the coherent sum group to plot. See the
                `coherent_sums` member of the `Results` class for a list of available
                sum groups and the amplitudes that belong to each group.
            data_legend (str): The legend label for the data points. Defaults to
                "GlueX-I Data".
            fractional (bool): Whether to plot the sums as a fraction of the
                total intensity. Defaults to False.
            kin_variable (str): Shorthand ("m", "t", "e") or exact 'data' dataframe
                column name for the kinematic variable to plot against. Default to 'm'
                (mass).
            stat (Literal['edges', 'avg']): Whether x-value/error is from the bin center
                and (high-low)/2 'edges' (default) or from the actual bin average and
                rms of the underlying data.
            t_bin (tuple[float, float] | TBin | None): Fixes the t bin to plot from if
                the results span multiple t bins. If only 1 t bin is available,
                specification is unnecessary. Defaults to None.
            energy_bin (tuple[float,float] | EnergyBin | None): Fixes the beam energy
                bin to plot from if the results span multiple energy bins. If only 1
                energy bin is available, specification is unnecessary. Defaults to None.
            mass_bin (tuple[float,float] | MassBin | None): Fixes the mass bin to plot
                from if the results span multiple mass bins. If only 1 mass bin is
                available, specification is unnecessary. Defaults to None.
            indices (list[int] | None): Optional list of positions within the resolved
                kinematic bin to select specific bins. Defaults to None.
            ax (matplotlib.axes.Axes | None): Optional axes to plot on. If None, a new
                figure and axes will be created.
            kwargs (dict[str, Any] | None): Optional dictionary of keyword arguments
                to customize the plot appearance.

        Raises:
            KeyError: If the specified sum_label is not found in the coherent sums, or
                or if stat='avg' but the requested corresponding columns are not present
                in the 'data' dataframe.
            ValueError: If a bin is given for the dimension being scanned over, or
                if multiple bins are present on a non-scammed dimension, leaving an
                ambiguous plot range.
        """
        if sum_label not in self.results.coherent_sums:
            raise KeyError(
                f"Sum label '{sum_label}' not found in coherent sums."
                f" Available sum labels: {list(self.results.coherent_sums.keys())}"
            )
        coherent_sums = self.results.coherent_sums[sum_label]

        plot_columns = (
            list(coherent_sums) + ["intensity", "ac_intensity"]
            if fractional
            else coherent_sums
        )
        fit_df, data_df, x_label, y_label = self._scan_dataframes(
            columns=plot_columns,
            kin_variable=kin_variable,
            stat=stat,
            t_bin=t_bin,
            energy_bin=energy_bin,
            mass_bin=mass_bin,
            indices=indices,
        )

        # default to Dark2 colormap, and cycle if more columns than colors
        colors = plt.get_cmap("Dark2").colors  # type: ignore
        colors = list(itertools.islice(itertools.cycle(colors), len(coherent_sums)))
        default_kwargs = {
            "marker": ["." for _ in range(len(coherent_sums))],
            "linestyle": ["" for _ in range(len(coherent_sums))],
            "alpha": [0.7 for _ in range(len(coherent_sums))],
            "colors": colors,
        }
        default_kwargs.update(kwargs or {})
        kwargs = default_kwargs

        if fractional:
            y_label = (
                y_label.replace("Events", "Fit Fraction")
                if "Events" in y_label
                else "Fit Fraction"
            )

        with self._style():
            fig, ax = (
                plt.subplots(layout="constrained")
                if ax is None
                else (ax.get_figure(), ax)
            )

            if not fractional:
                # plot the data points with error bars, using the appropriate events
                # column based on whether the results are acceptance-corrected or not
                if self.results.is_acc_corrected:
                    data_points = unumpy.uarray(
                        data_df["ac_events"], data_df["ac_events_err"]
                    )
                else:
                    data_points = unumpy.uarray(
                        data_df["events"], data_df["events_err"]
                    )

                ax.errorbar(
                    x=data_df["x_center"],
                    xerr=data_df["x_err"],
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

                if fractional:
                    intensity = (
                        unumpy.uarray(
                            fit_df["ac_intensity"], fit_df["ac_intensity_err"]
                        )
                        if self.results.is_acc_corrected
                        else unumpy.uarray(fit_df["intensity"], fit_df["intensity_err"])
                    )
                    y = (
                        unumpy.uarray(fit_df[coh_sum], fit_df[f"{coh_sum}_err"])
                        / intensity
                    )
                    y_vals = unumpy.nominal_values(y)
                    y_errs = unumpy.std_devs(y)
                else:
                    y_vals = fit_df[coh_sum]
                    y_errs = fit_df[f"{coh_sum}_err"]

                ax.errorbar(
                    x=data_df["x_center"],
                    xerr=data_df["x_err"],
                    y=y_vals,
                    yerr=y_errs,
                    label=label,
                    marker=kwargs["marker"][sum_idx],
                    linestyle=kwargs["linestyle"][sum_idx],
                    alpha=kwargs["alpha"][sum_idx],
                    color=kwargs["colors"][sum_idx],
                )

            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_ylim(bottom=0)
            ax.legend()

        return ax

    def amplitudes(
        self,
        fractional: bool = False,
        sharey: bool = False,
        reflectivity: Literal["positive", "negative", "all"] = "all",
        kin_variable: str = "m",
        stat: Literal["edges", "avg"] = "edges",
        t_bin: tuple[float, float] | TBin | None = None,
        energy_bin: tuple[float, float] | EnergyBin | None = None,
        mass_bin: tuple[float, float] | MassBin | None = None,
        indices: list[int] | None = None,
        axs: np.ndarray | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Plot a grid of amplitudes, organized by spin (rows) and projection (columns)

        This plot will create a grid of all amplitudes plotted as a function of the
        selected kinematic variable, with the rows corresponding to the
        spin+parity+angular momenta combo (J^P L) and the columns corresponding to the
        spin-projection m. Any quantum numbers unavailable due to the naming scheme are
        dropped. Reflectivities are plotted together in the same plot, with the option
        to select only positive or negative reflectivities.

        Args:
            fractional (bool): Whether to plot the amplitudes as a fraction of the
                total intensity. Defaults to False.
            sharey (bool): Whether to share the y-axis across all amplitude plots.
                Defaults to False.
            reflectivity (Literal["positive", "negative", "all"]): Which reflectivity
                amplitudes to plot. Defaults to "all".
            kin_variable (str): Shorthand ("m", "t", "e") or exact 'data' dataframe
                column name for the kinematic variable to plot against.
                Defaults to "m" (mass).
            stat (Literal['edges', 'avg']): Whether x-value/error is from the bin center
                and (high-low)/2 'edges' (default) or from the actual bin average and
                rms of the underlying data.
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
            axs (np.ndarray | None): Optional array of axes to plot on. If None, a new
                figure and axes will be created. Note that one must be careful that the
                axes shape matches the expected shape.
            kwargs (dict[str, Any] | None): Optional dictionary of keyword arguments
                to customize the plot appearance. Recognized keys are "positive" and
                "negative" mapping to a dict of matplotlib 'errorbar' kwargs that
                override the defaults for the respective reflectivity amplitudes.
                e.g. {"positive": {"color": "red"}, "negative": {"color": "blue"}}.
                Defaults to None.

        Returns:
            np.ndarray: The array of axes objects containing the amplitude plots.
        Raises:
            ValueError: If the reflectivity argument is not one of "positive",
                "negative", or "all". If no naming scheme could be determined from the
                amplitudes, or if a bin is given for the dimension being scanned over,
                or if multiple bins are present along a non-scanned dimension, leaving
                an ambiguous plot range.
            IndexError: If the provided axes shape does not match the expected shape
                based on the number of amplitudes.
        """

        if reflectivity not in ["positive", "negative", "all"]:
            raise KeyError(
                f"Invalid reflectivity value: {reflectivity}. Must be one of "
                "'positive', 'negative', or 'all'."
            )

        amps = list(self.results.amplitudes)
        if not amps:
            raise KeyError("No amplitudes found in the results to plot.")
        # resolve naming scheme from the amplitudes (common across all, so select one)
        parser = self.results.parser
        scheme = parser.requested_scheme
        if scheme == NamingScheme.AUTO:
            scheme = parser.infer_naming_scheme(amps[0])
        if scheme == NamingScheme.AUTO:
            raise ValueError("Could not determine naming scheme from amplitudes.")
        scheme_def = SCHEMES[scheme]

        # Determine the row and column quantum numbers based on the naming scheme.
        row_quantum_numbers = tuple(
            qn for qn in scheme_def.single_amplitudes if qn not in ("m", "e")
        )
        row_group_key = "".join(row_quantum_numbers)

        parsed_amps = [(amp, parser.parse_amplitude(amp)) for amp in amps]

        def _keep(parsed) -> bool:
            if reflectivity == "all":
                return True
            if reflectivity == "positive":
                return parsed.e in _POSITIVE_REFLECTIVITY_CHARS
            return parsed.e in _NEGATIVE_REFLECTIVITY_CHARS

        parsed_amps = [(amp, parsed) for amp, parsed in parsed_amps if _keep(parsed)]
        if not parsed_amps:
            raise ValueError(
                f"No amplitudes found for reflectivity '{reflectivity}'. "
                f"Available amplitudes: {amps}"
            )

        row_groups: dict[tuple[str, ...], list[tuple[str, Any]]] = {}
        col_values: set[str] = set()
        for amp, parsed in parsed_amps:
            row_key = tuple(parsed.get(qn) for qn in row_quantum_numbers)
            row_groups.setdefault(row_key, []).append((amp, parsed))
            col_values.add(parsed.m)

        sorted_rows = sorted(
            row_groups, key=lambda rk: tuple(_numeric_sort_key(qn) for qn in rk)
        )
        sorted_cols = sorted(col_values, key=_numeric_sort_key)
        nrows, ncols = len(sorted_rows), len(sorted_cols)

        plot_columns = [amp for amp, _ in parsed_amps]
        if fractional:
            plot_columns.extend(["intensity", "ac_intensity"])
        fit_df, data_df, x_label, y_label = self._scan_dataframes(
            columns=plot_columns,
            kin_variable=kin_variable,
            stat=stat,
            t_bin=t_bin,
            energy_bin=energy_bin,
            mass_bin=mass_bin,
            indices=indices,
        )

        # default styling
        default_kwargs = {
            "positive": {
                "marker": ".",
                "linestyle": "",
                "alpha": 0.7,
                "color": "tab:red",
            },
            "negative": {
                "marker": ".",
                "linestyle": "",
                "alpha": 0.7,
                "color": "tab:blue",
            },
        }
        for refl in ("positive", "negative"):
            default_kwargs[refl].update(kwargs.get(refl, {}) if kwargs else {})
        kwargs = default_kwargs

        if fractional:
            y_label = (
                y_label.replace("Events", "Fit Fraction")
                if "Events" in y_label
                else "Fit Fraction"
            )

        max_value: float = 0.0  # for resetting y_lim later

        with self._style():
            if axs is None:
                fig, axs = plt.subplots(
                    nrows=nrows,
                    ncols=ncols,
                    sharey=sharey,
                    squeeze=False,
                    figsize=(4 * ncols, 3 * nrows),
                    layout="constrained",
                )
            else:
                axs = np.asarray(axs)
                try:
                    axs = axs.reshape(nrows, ncols)
                except ValueError:
                    raise IndexError(
                        f"Provided axes shape {axs.shape} does not match expected "
                        f"shape ({nrows}, {ncols}) based on the number of amplitudes."
                    )
            for row_idx, row_key in enumerate(sorted_rows):
                row_label = self.results.parser.sum_to_latex(
                    row_group_key, "".join(row_key)
                )
                for col_idx, col_value in enumerate(sorted_cols):
                    ax = axs[row_idx, col_idx]
                    amps_here = [
                        (amp, parsed)
                        for amp, parsed in row_groups[row_key]
                        if parsed.m == col_value
                    ]
                    if not amps_here:
                        ax.set_visible(False)
                        continue

                    # plot positive reflectivity first
                    amps_here.sort(
                        key=lambda item: item[1].e not in _POSITIVE_REFLECTIVITY_CHARS
                    )
                    for amp, parsed in amps_here:
                        refl_kind = (
                            "positive"
                            if parsed.e in _POSITIVE_REFLECTIVITY_CHARS
                            else "negative"
                        )

                        if fractional:
                            values = unumpy.uarray(fit_df[amp], fit_df[f"{amp}_err"])
                            intensity = (
                                unumpy.uarray(
                                    fit_df["ac_intensity"], fit_df["ac_intensity_err"]
                                )
                                if self.results.is_acc_corrected
                                else unumpy.uarray(
                                    fit_df["intensity"], fit_df["intensity_err"]
                                )
                            )

                            fraction = values / intensity

                            y = unumpy.nominal_values(fraction)
                            yerr = unumpy.std_devs(fraction)
                        else:
                            y = fit_df[amp].to_numpy()
                            yerr = fit_df[f"{amp}_err"].to_numpy()

                        max_value = max(max_value, y.max() + y.max() * 0.1)

                        ax.errorbar(
                            x=data_df["x_center"],
                            xerr=data_df["x_err"],
                            y=y,
                            yerr=yerr,
                            label=self.results.parser.to_latex(amp),
                            **kwargs[refl_kind],
                        )

                    ax.set_title(
                        rf"{row_label}, $m={_format_spin_projection(col_value)}$",
                        fontsize="small",
                    )
                    ax.set_ylim(bottom=0)
                    ax.legend(fontsize="x-small")

                    if row_idx == nrows - 1:
                        ax.set_xlabel(x_label)
                    if col_idx == 0:
                        ax.set_ylabel(y_label)

        # adjust y limits after plotting if using common axis
        if sharey:
            for ax in axs.flatten():
                ax.set_ylim(top=max_value)

        return axs

    def interference(
        self,
        amp1: str,
        amp2: str,
        indices: list[int] | None = None,
        amp1_kwargs: dict[str, Any] | None = None,
        amp2_kwargs: dict[str, Any] | None = None,
        amp_ax: matplotlib.axes.Axes | None = None,
        phase_ax: matplotlib.axes.Axes | None = None,
    ) -> np.ndarray:
        """Plot two amplitudes and their interference phase as a function of mass.

        Args:
            amp1 (str): The label of the first amplitude to plot.
            amp2 (str): The label of the second amplitude to plot.
            indices (list[int] | None): Optional list of indices to select specific mass
                bins. If None, all bins will be plotted.
            amp1_kwargs (dict[str, Any] | None): Optional dictionary of keyword
                arguments to customize the appearance of the first amplitude plot.
            amp2_kwargs (dict[str, Any] | None): Optional dictionary of keyword
                arguments to customize the appearance of the second amplitude plot.
            amp_ax (matplotlib.axes.Axes | None): Optional axes to plot the amplitudes
                on. If None, a new figure and axes will be created.
            phase_ax (matplotlib.axes.Axes | None): Optional axes to plot the phase
                difference on. If None, a new figure and axes will be created.
        Returns:
            np.ndarray: The array of axes objects containing the amplitude and phase
                difference plots.
        Raises:
            KeyError: If either phase is not in the fit results.
            ValueError: If only one of amp_ax or phase_ax is provided, but not both
        """

        if amp1 not in self.results.fit.columns:
            raise KeyError(
                f"Amplitude '{amp1}' not found in fit results. "
                f"Available amplitudes: {list(self.results.fit.columns)}"
            )
        if amp2 not in self.results.fit.columns:
            raise KeyError(
                f"Amplitude '{amp2}' not found in fit results. "
                f"Available amplitudes: {list(self.results.fit.columns)}"
            )

        if amp_ax is None and phase_ax is None:
            fig, axs = plt.subplots(
                nrows=2,
                ncols=1,
                sharex=True,
                gridspec_kw={"wspace": 0.0, "hspace": 0.07},
                height_ratios=[3, 1],
                layout="constrained",
            )
        elif amp_ax is None or phase_ax is None:
            raise ValueError(
                "Both amp_ax and phase_ax must be provided if one is specified."
            )
        else:
            axs = np.array([amp_ax, phase_ax])

        # TODO: obtain the relevant dataframes, and plot their amplitude and phase
        # differences on the appropriate axes. Use the provided kwargs for
        # customization.

        return axs

    def model_matrix(
        self,
        indices: list[int] | None = None,
        axs: np.ndarray | None = None,
    ) -> np.ndarray:
        """Plot the entire model behavior as a function of mass in a matrix of plots

        The matrix is organized where the amplitudes are plotted on the diagonal,
        grouped by reflectivity (similar to the amplitudes() method), and the
        off-diagonal plots show the interference between the amplitudes. The upper
        triangle is for the positive reflectivity amplitude interferences, and the
        lower triangle for the negative reflectivity amplitude interferences.

        Args:
            indices (list[int] | None): Optional list of indices to select specific mass
                bins. If None, all bins will be plotted.
            axs (np.ndarray | None): Optional array of axes to plot on. If None, a new
                figure and axes will be created. Note that one must be careful that the
                axes shape matches the expected shape.
        Returns:
            np.ndarray: The array of axes objects containing the amplitudes and
                interference plots.
        Raises:
            IndexError: If the provided axes shape does not match the expected shape
                based on the number of amplitudes and reflectivities.
        """

        if axs is None:
            # TODO: determine shape, reference amplitudes() method for guidance.
            nrows = 0
            ncols = 0
            fig, axs = plt.subplots(
                nrows,
                ncols,
                sharey=True,
                figsize=(4 * ncols, 3 * nrows),
                layout="constrained",
            )

        # TODO: make sure to plot errorbars as smooth fill_between due to small plots

        return axs  # type: ignore

    def convergence_rate(
        self,
        kin_variable: str = "m",
        stat: Literal["edges", "avg"] = "edges",
        t_bin: tuple[float, float] | TBin | None = None,
        energy_bin: tuple[float, float] | EnergyBin | None = None,
        mass_bin: tuple[float, float] | MassBin | None = None,
        indices: list[int] | None = None,
        ax: matplotlib.axes.Axes | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> matplotlib.axes.Axes:
        """Plot the convergence rate of the fit across bins.

        Requires randomized fits to be in the results. Plots the percentage of
        successful, failed, and converged-with-bad-error-matrix fits across the selected
        kinematic bins. This can be useful for diagnosing issues with the fit and
        understanding the stability of the fit across the mass range.

        Args:
            kin_variable (str): Shorthand ("m", "t", "e") or exact 'data' dataframe
                column name for the kinematic variable to plot against. Default to 'm'
                (mass).
            stat (Literal['edges', 'avg']): Whether x-value/error is from the bin center
                and (high-low)/2 'edges' (default) or from the actual bin average and
                rms of the underlying data.
            t_bin (tuple[float, float] | TBin | None): Fixes the t bin to plot from if
                the results span multiple t bins. If only 1 t bin is available,
                specification is unnecessary. Defaults to None.
            energy_bin (tuple[float,float] | EnergyBin | None): Fixes the beam energy
                bin to plot from if the results span multiple energy bins. If only 1
                energy bin is available, specification is unnecessary. Defaults to None.
            mass_bin (tuple[float,float] | MassBin | None): Fixes the mass bin to plot
                from if the results span multiple mass bins. If only 1 mass bin is
                available, specification is unnecessary. Defaults to None.
            indices (list[int] | None): Optional list of positions within the resolved
                kinematic bin to select specific bins. Defaults to None.
            ax (matplotlib.axes.Axes | None): Optional axes to plot on. If None, a new
                figure and axes will be created.
            kwargs (dict[str, Any] | None): Optional dictionary of keyword arguments
                to customize the plot appearance. Organized per fit status type as
                'successful', 'bad error matrix', 'failed', so given value lists should
                be in the same ordering e.g. "labels" : ["Success", "Bad Error",
                "Failed"].
        Returns:
            matplotlib.axes.Axes: The axes object containing the convergence rate plot.
        Raises:
            KeyError: If randomized fits are not available in the results.
        """

        if self.results.randomized is None:
            raise KeyError("Randomized fits are required to plot convergence rate.")

        fit_status_columns = ["eMatrixStatus", "lastMinuitCommandStatus"]

        df, data_df, x_label, _ = self._scan_dataframes(
            columns=fit_status_columns,
            frame="randomized",
            kin_variable=kin_variable,
            stat=stat,
            t_bin=t_bin,
            energy_bin=energy_bin,
            mass_bin=mass_bin,
            indices=indices,
        )

        successful_fits = (
            df.groupby("bin_id")
            .apply(
                (
                    lambda x: (
                        (x["eMatrixStatus"] == 3) & (x["lastMinuitCommandStatus"] == 0)
                    ).sum()
                )
            )
            .to_numpy()
        )
        bad_eMatrix_fits = (
            df.groupby("bin_id")
            .apply(
                (
                    lambda x: (
                        (x["eMatrixStatus"] != 3) & (x["lastMinuitCommandStatus"] == 0)
                    ).sum()
                )
            )
            .to_numpy()
        )
        failed_fits = (
            df.groupby("bin_id")
            .apply((lambda x: (x["lastMinuitCommandStatus"] != 0).sum()))
            .to_numpy()
        )

        max_length = max(len(successful_fits), len(bad_eMatrix_fits), len(failed_fits))
        if max_length == 0:
            raise ValueError(
                "No labelled fit statuses could be found, check validity of the"
                " randomized dataframe"
            )
        successful_fits = (
            np.zeros(max_length) if len(successful_fits) == 0 else successful_fits
        )
        bad_eMatrix_fits = (
            np.zeros(max_length) if len(bad_eMatrix_fits) == 0 else bad_eMatrix_fits
        )
        failed_fits = np.zeros(max_length) if len(failed_fits) == 0 else failed_fits

        default_kwargs = {
            "colors": ["tab:blue", "tab:orange", "tab:red"],
            "labels": ["Success", "Inaccurate Errors", "Failed"],
        }
        default_kwargs.update(kwargs or {})
        kwargs = default_kwargs

        with self._style():
            fig, ax = (
                plt.subplots(layout="constrained")
                if ax is None
                else (ax.get_figure(), ax)
            )

            bottom = np.zeros(len(successful_fits))
            for i, fit_status in enumerate(
                [successful_fits, bad_eMatrix_fits, failed_fits]
            ):
                ax.bar(
                    data_df["x_center"],
                    fit_status,
                    data_df["x_err"],
                    label=kwargs["labels"][i],
                    color=kwargs["colors"][i],
                    bottom=bottom,
                )
                bottom += fit_status

            ax.set_xlabel(x_label)
            ax.set_ylabel(r"# of fits")
            ax.set_ylim(bottom=0)
            ax.legend()

        return ax

    def ridgeline(
        self,
        columns: list[str],
        indices: list[int] | None = None,
    ) -> np.ndarray:
        """Create a ridgeline plot of the columns from the bootstrap distributions

        Args:
            columns (list[str]): _description_
            indices (list[int] | None, optional): _description_. Defaults to None.

        Returns:
            np.ndarray: _description_

        Raises:
            KeyError: If the specified columns are not found in the bootstrap dataframe,
                or if the bootstrap dataframe is not available in the results.
        """

        if self.results.bootstrap is None:
            raise KeyError(
                "Bootstrap distributions are required to create a ridgeline plot."
            )

        for col in columns:
            if col not in self.results.bootstrap.columns:
                raise KeyError(
                    f"Column '{col}' not found in bootstrap distributions. "
                    f"Available columns: {list(self.results.bootstrap.columns)}"
                )

        # TODO: replace with joypy.joyplot, ridgeplot, seaborn, or similar library
        fig, axs = plt.subplots(2, 2)

        return axs

    # ----------------------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------------------

    def _resolve_kin_variable(
        self,
        kin_variable: str,
        stat: Literal["edges", "avg"] = "edges",
    ) -> tuple[str | None, str | None, str | None, str | None, str, str]:
        """Resolve a shorthand or column name to 'data' dataframe column.

        Args:
            kin_variable (str): A known shorthand ("m", "t", "e") or exact name of a
                column in the 'data' dataframe to use as the x-axis values.
            stat (Literal['edges', 'avg']): Whether x-value/error is from the bin center
                and (high-low)/2 'edges' (default) or from the actual bin average and
                rms of the underlying data.
        Returns:
            tuple[str | None, str | None, str | None, str | None, str, str]: Central
                value column name (None if no dedicated "_center" column), the low-edge
                column name (if available), the high-edge column name (if available),
                and LaTeX formatted x,y axis label.

        Raises:
            KeyError: If kin_variable cannot be resolved, or for stat='edges' that
                there aren't enough columns to compute one, in the 'data' dataframe.
                Also if stat='avg' is requested but the corresponding
                <kin_variable>_<avg/rms> columns are not available.
            ValueError: If stat is not 'edges' or 'avg'
        """

        if stat not in ("edges", "avg"):
            raise ValueError(f"stat must be 'edges' or 'avg', got '{stat}'")

        columns = self.results.data.columns
        if kin_variable in _KIN_VARIABLE_XLABELS:
            x_label = _KIN_VARIABLE_XLABELS[kin_variable]
            y_label = _KIN_VARIABLE_YLABELS[kin_variable]

            low = f"{kin_variable}_low" if f"{kin_variable}_low" in columns else None
            high = f"{kin_variable}_high" if f"{kin_variable}_high" in columns else None
            bin_width = (
                self.results.data[high] - self.results.data[low]
                if (low is not None and high is not None)
                else None
            )
            if bin_width is not None and (
                (np.std(bin_width) / np.mean(bin_width)) < 0.01
            ):
                # if relative standard deviation of the bin width is within 1%, we can
                # reasonably assume that we have a constant bin width to label the
                # y-axis with
                y_label = y_label.replace("NUM", f"{bin_width.mean():.3f}")
            else:
                # bin width is either unknown or variable, so cut down y_label to just
                # "Events"
                y_label = y_label.split("/")[0]

            if stat == "avg":
                center, rms = f"{kin_variable}_avg", f"{kin_variable}_rms"
                if center not in columns or rms not in columns:
                    raise KeyError(
                        f"stat='avg' requires '{center}' and '{rms}' columns in the"
                        f" data dataframe for kin_variable='{kin_variable}', but at"
                        f" least one was not found. Available columns:"
                        f" {list(columns)}"
                    )
                return center, None, None, rms, x_label, y_label

            # stat == 'edges': prefers a dedicated "<kin_variable>_center" column,
            # but will otherwise compute midpoint from low/high columns. Falls back to
            # <kin_variable>_avg as a last resort.
            if f"{kin_variable}_center" in columns:
                center: str | None = f"{kin_variable}_center"
            elif low is not None and high is not None:
                center = None  # computed as (high-low)/2 in _scan_dataframes
            elif f"{kin_variable}_avg" in columns:
                center = f"{kin_variable}_avg"
            else:
                raise KeyError(
                    f"Could not resolve a center for kin_variable '{kin_variable}' with"
                    f" stat='edges': no '{kin_variable}'_center,"
                    f" '{kin_variable}_low/high pair, and no '{kin_variable}_avg'"
                    f" column were found. Available columns {list(columns)}"
                )
            return center, low, high, None, x_label, y_label

        elif kin_variable in columns:
            return kin_variable, None, None, None, kin_variable, kin_variable
        else:
            raise KeyError(
                f"kin_variable '{kin_variable}' cannot be resolved. Must be one of "
                f"{list(_KIN_VARIABLE_XLABELS.keys())} or a column in the 'data' "
                f"dataframe: {list(columns)}"
            )

    def _resolve_scan_bins(
        self,
        kin_variable: str,
        t_bin: tuple[float, float] | TBin | None,
        energy_bin: tuple[float, float] | EnergyBin | None,
        mass_bin: tuple[float, float] | MassBin | None,
    ) -> list[KinematicBin]:
        """Resolve the ordered list of kinematic bins to scan over.

        Results may span multiple bins e.g. 3 t bins x 2 energy bins x 10 mass bins. To
        plot as a function of one kinematic variable, the other two must be fixed to a
        single bin each. We only need to do this though if more than one unscanned bin
        is available. For example, if the results span 3 t bins and 2 energy bins, but
        only 1 mass bin, then we can plot as a function of t without needing to fix the
        mass bin, but would need to fix the energy bin.

        Args:
            kin_variable (str): The kinematic variable to scan over ("m", "t", "e").
                Any other value is treated as a column name in the 'data' dataframe, and
                will be used as the x-axis. The given t-bin/energy-bin/mass-bin will be
                used to select the appropriate bin to plot from.
            t_bin (tuple[float, float] | TBin | None): Optional fixed t bin.
            energy_bin (tuple[float, float] | EnergyBin | None): Optional fixed energy
                bin.
            mass_bin (tuple[float, float] | MassBin | None): Optional fixed mass bin.

        Returns:
            list[KinematicBin]: The ordered list of kinematic bins to scan over.

        Raises:
            ValueError: If a bin is given for the dimension being scanned over, or if
                multiple bins are present on a non-scanned dimension, leaving an
                ambiguous plot range.
        """
        if kin_variable == "m":
            if mass_bin is not None:
                raise ValueError(
                    "Cannot specify a mass bin when scanning over mass. "
                    "Please leave mass_bin as None."
                )
            return self.results.mass_kinematic_bins(t_bin=t_bin, energy_bin=energy_bin)
        elif kin_variable == "t":
            if t_bin is not None:
                raise ValueError(
                    "Cannot specify a t bin when scanning over t. "
                    "Please leave t_bin as None."
                )
            return self.results.t_kinematic_bins(
                mass_bin=mass_bin, energy_bin=energy_bin
            )
        elif kin_variable == "e":
            if energy_bin is not None:
                raise ValueError(
                    "Cannot specify an energy bin when scanning over energy. "
                    "Please leave energy_bin as None."
                )
            return self.results.energy_kinematic_bins(t_bin=t_bin, mass_bin=mass_bin)

        # arbitrary 'data' column case: no per-axis results method exists to check for
        # for ambiguity along an unknown axis, so this just applies whichever bins were
        # given as a plain filter
        return self._resolve_kinematic_bins(t_bin, energy_bin, mass_bin)

    def _scan_dataframes(
        self,
        columns: tuple[str, ...] | list[str] | None = None,
        frame: Literal[
            "fit", "correlation", "covariance", "norm_int", "randomized", "bootstrap"
        ] = "fit",
        kin_variable: str = "m",
        stat: Literal["edges", "avg"] = "edges",
        t_bin: tuple[float, float] | TBin | None = None,
        energy_bin: tuple[float, float] | EnergyBin | None = None,
        mass_bin: tuple[float, float] | MassBin | None = None,
        indices: list[int] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
        """Get a results dataframe and 'data' dataframe over a kinematic variable scan

        Every plot in this class needs two things: the set of kinematic bins to scan
        over, resolved and ordered along 'kin_variable' (see '_resolve_scan_bins'), and
        rows from one of the 'result' dataframes restricted to those bins. This does
        both, for any of the 'results.fit', '.correlation', ... . Fit, randomized, and
        bootstrap dataframes share the same per-quantity "<col>"/"<col>_err" convention,
        so those columns are automatically appended. Matrix-like results (norm-int,
        covariance, etc.) use 'columns' as-is.

        Unlike 'fit' or 'data' frames, the other frames have several rows per bin. These
        returned frames always keep a "bin_id" column so those rows stay identifiable.

        Args:
            columns (tuple[str, ...] | list[str] | None): The columns to include from
                'frame'. For 'fit'/'randomized'/'bootstrap' frames, the "_err"
                companions are automatically added, if available. If None (default) all
                columns are included.
            frame (Literal['fit', 'correlation', 'covariance', 'norm_int', 'randomized'
                'bootstrap']): Which 'results' dataframe to pull 'columns' from.
                Defaults to 'fit'.
            kin_variable (str): The kinematic variable to scan over ("m", "t", "e").
                Any other value is treated as a column name in the 'data' dataframe, and
                will be used as the x-axis. The given t-bin/energy-bin/mass-bin will be
                used to select the appropriate bin to plot from.
            stat (Literal['edges', 'avg']): Whether x-value/error is from the bin center
                and (high-low)/2 'edges' (default) or from the actual bin average and
                rms of the underlying data.
            t_bin (tuple[float, float] | TBin | None): Optional fixed t bin.
            energy_bin (tuple[float, float] | EnergyBin | None): Optional fixed energy
                bin.
            mass_bin (tuple[float, float] | MassBin | None): Optional fixed mass bin.
            indices (list[int] | None): Optional list of indices, within the resolved
                and sorted scan, to select specific bins. If None, all bins will be
                used.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, str, str]: The requested 'frame's rows
            (restricted to the resolve bins, 'bin_id' retained, grouped/ordered to match
            the scan), the 'data' dataframe (kinematic variable renamed to 'x_center'
            and 'x_err', 'bin_id' retained) and LaTeX formatted x-y axis labels for the
            kinematic variable.
        Raises:
            KeyError: If kin_variable cannot be resolved to a known shorthand or column
                name in the 'data' dataframe, or requested 'frame' is not available.
            ValueError: If a bin is given for the dimension being scanned over, or if
                multiple bins are present on a non-scanned dimension, leaving an
                ambiguous plot range.
        """
        center_col, low_col, high_col, rms_col, x_label, y_label = (
            self._resolve_kin_variable(kin_variable, stat)
        )

        kinematic_bins = self._resolve_scan_bins(
            kin_variable, t_bin, energy_bin, mass_bin
        )
        if indices is not None:
            kinematic_bins = [kinematic_bins[i] for i in indices]
        bin_ids = [kb.bin_id for kb in kinematic_bins]
        if not bin_ids:
            raise ValueError(
                "No kinematic bins found for the specified scan. Please check the "
                "provided t_bin, energy_bin, and mass_bin arguments."
            )

        requested_frame, frame_columns = self._frame_columns(frame, columns)
        value_df = self._select_by_bin_id(requested_frame, bin_ids, frame_columns)
        if frame == "fit":
            value_df = self._replace_errors_with_bootstrap(value_df)

        data_columns = [
            "events",
            "events_err",
            "ac_events",
            "ac_events_err",
        ]
        for extra_col in (center_col, low_col, high_col, rms_col):
            if extra_col is not None:
                data_columns.append(extra_col)
        data_df = self._select_by_bin_id(self.results.data, bin_ids, data_columns)
        if center_col is not None:
            data_df = data_df.rename(columns={center_col: "x_center"})
        else:
            # no dedicated center column, use the midpoint of the bin edges instead
            data_df["x_center"] = (data_df[low_col] + data_df[high_col]) / 2.0

        if rms_col is not None:
            data_df["x_err"] = data_df[rms_col]
        elif low_col is not None and high_col is not None:
            data_df["x_err"] = (data_df[high_col] - data_df[low_col]) / 2.0
        else:
            data_df["x_err"] = np.nan

        return value_df, data_df, x_label, y_label
