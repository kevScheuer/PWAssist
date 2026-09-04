"""Individual preprocessing steps that can act on fit results in a bin bundle"""

import warnings

import numpy as np
import pandas as pd

from pwassist.io.binning import BinBundle
from pwassist.io.catalog import Catalog, ResultsFile
from pwassist.parser import AmplitudeParser

FILE_TYPE_MAP: dict[str, type[ResultsFile]] = {
    cls.__name__: cls for cls in Catalog.RESULT_FILE_TYPES
}


def stamp_kinematic_bin_columns(bundle: BinBundle) -> None:
    """Attach kinematic bin information to each results file in the bundle"""
    kb = bundle.kinematic_bin

    # it may seem redundant to stamp the DataFile frame with the kinematic bin info
    # again, but this ensures there is a common set of columns across all results files,
    # allowing for easier group by operations and comparisons later on. Without it,
    # the non-rounded bin edges would cause mismatches
    for rf in FILE_TYPE_MAP.values():
        rf = bundle.get(rf)
        if rf is None:
            continue
        new_columns = {
            "t_bin": pd.Interval(kb.t_bin.low, kb.t_bin.high, closed="neither"),
            "mass_bin": pd.Interval(
                kb.mass_bin.low, kb.mass_bin.high, closed="neither"
            ),
            "energy_bin": pd.Interval(
                kb.energy_bin.low, kb.energy_bin.high, closed="neither"
            ),
            "bin_id": [kb.bin_id],
        }

        overwritten = [c for c in new_columns if c in rf.frame.columns]
        if overwritten:
            warnings.warn(
                f"[{bundle.bin_id}] Column(s) {overwritten} already exist in"
                f" {rf.__class__.__name__}.frame. Overwriting with kinematic bin"
                " values.",
                UserWarning,
            )

        stamped = pd.DataFrame(new_columns, index=rf.frame.index)
        rf.frame = pd.concat(
            [rf.frame.drop(columns=overwritten, errors="ignore"), stamped], axis=1
        )


def check_null_columns(bundle: BinBundle) -> None:
    """Check if null columns exist in any of the results files"""
    for label, rf in FILE_TYPE_MAP.items():
        if bundle.get(rf) is None:
            continue
        frame = bundle.get(rf).frame  # type: ignore claims frame could be None...
        null_cols = frame.columns[frame.isnull().any()].tolist()
        if null_cols:
            warnings.warn(
                f"[{bundle.bin_id}] {label} contains null values in columns:"
                f" {null_cols}.",
                UserWarning,
            )


def check_fit_status(bundle: BinBundle) -> None:
    """Flag final fits with bad Minuit or error matrix statuses"""
    fit = bundle.fit
    if (
        fit is None
        or "eMatrixStatus" not in fit.frame.columns
        or "lastMinuitCommandStatus" not in fit.frame.columns
    ):
        return
    df = fit.frame
    bad_matrix_rows = df.index[df["eMatrixStatus"] != 3].tolist()
    bad_statuses_rows = df.index[df["lastMinuitCommandStatus"] != 0].tolist()
    if bad_matrix_rows:
        warnings.warn(
            f"[{bundle.bin_id}] Fit contains bad error matrix statuses"
            f" (eMatrixStatus != 3) in rows: {bad_matrix_rows}."
            f" Covariance matrix may be unreliable.",
            UserWarning,
        )
    if bad_statuses_rows:
        warnings.warn(
            f"[{bundle.bin_id}] Fit contains bad Minuit statuses"
            f" (lastMinuitCommandStatus != 0) in rows: {bad_statuses_rows}."
            f" Fit may not have converged.",
            UserWarning,
        )


def check_error_columns(bundle: BinBundle) -> None:
    """Ensure '_err' columns are non-negative and finite

    Checks the final fits, randomized, and bootstrap results.
    """

    def warn_if_invalid(series: pd.Series, col_name: str, bin_id: str) -> None:
        if (series < 0).any():
            warnings.warn(
                f"[{bin_id}] Results contain negative values in error column"
                f" '{col_name}'.",
                UserWarning,
            )
        if not np.isfinite(series).all():
            warnings.warn(
                f"[{bin_id}] Results contain non-finite values in error column"
                f" '{col_name}'.",
                UserWarning,
            )

    fit = bundle.fit

    if fit is None:
        return

    for col in fit.frame.columns:
        if col.endswith("_err"):
            series = fit.frame[col]
            warn_if_invalid(series, col, bundle.bin_id)

    if bundle.randomized is not None:
        for col in bundle.randomized.frame.columns:
            if col.endswith("_err"):
                series = bundle.randomized.frame[col]
                warn_if_invalid(series, col, bundle.bin_id)

    if bundle.bootstrap is not None:
        for col in bundle.bootstrap.frame.columns:
            if col.endswith("_err"):
                series = bundle.bootstrap.frame[col]
                warn_if_invalid(series, col, bundle.bin_id)


def align_phase_column_names(bundle: BinBundle) -> None:
    """Make phase names consistent across fit, randomized, and bootstrap results"""
    fit = bundle.fit
    if fit is None:
        return

    parser = AmplitudeParser()
    phase_cols = parser.get_phase_differences(fit.frame.columns.to_list())
    reversed_phase_cols = [
        f"{a2}_{a1}" for a1, a2 in (col.split("_") for col in phase_cols)
    ]
    if bundle.randomized is not None:
        for col in reversed_phase_cols:
            if col in bundle.randomized.frame.columns:
                new_col = f"{col.split('_')[1]}_{col.split('_')[0]}"
                bundle.randomized.frame.rename(columns={col: new_col}, inplace=True)
    if bundle.bootstrap is not None:
        for col in reversed_phase_cols:
            if col in bundle.bootstrap.frame.columns:
                new_col = f"{col.split('_')[1]}_{col.split('_')[0]}"
                bundle.bootstrap.frame.rename(columns={col: new_col}, inplace=True)


def wrap_phase_columns(bundle: BinBundle) -> None:
    """Wrap phase columns (in radians) to the range (-180, 180] in degrees"""
    fit = bundle.fit
    if fit is None:
        return

    parser = AmplitudeParser()
    phase_cols = parser.get_phase_differences(fit.frame.columns.to_list())
    phase_err_cols = [
        f"{c}_err" for c in phase_cols if f"{c}_err" in fit.frame.columns
    ]  # We won't wrap them, but they need to be converted to degrees

    if not phase_cols:
        return

    for col in phase_cols:
        fit.frame[col] = np.rad2deg(
            np.angle(np.exp(1j * fit.frame[col]))
        )  # Wrap to (-pi, pi] using complex exponential
    for col in phase_err_cols:
        fit.frame[col] = np.rad2deg(fit.frame[col])  # Convert to degrees

    if bundle.randomized is not None:
        for col in phase_cols:
            bundle.randomized.frame[col] = np.rad2deg(
                np.angle(np.exp(1j * bundle.randomized.frame[col]))
            )
        for col in phase_err_cols:
            bundle.randomized.frame[col] = np.rad2deg(bundle.randomized.frame[col])

    if bundle.bootstrap is not None:
        for col in phase_cols:
            bundle.bootstrap.frame[col] = np.rad2deg(
                np.angle(np.exp(1j * bundle.bootstrap.frame[col]))
            )
        for col in phase_err_cols:
            bundle.bootstrap.frame[col] = np.rad2deg(bundle.bootstrap.frame[col])


def downcast_numeric_dtypes(bundle: BinBundle) -> None:
    """Downcast numeric columns to save memory"""
    for label, rf in FILE_TYPE_MAP.items():
        if bundle.get(rf) is None:
            continue
        df = bundle.get(rf).frame  # type: ignore
        for col in df.select_dtypes(include=["float64"]).columns:
            df[col] = pd.to_numeric(df[col], downcast="float")
        for col in df.select_dtypes(include=["int64"]).columns:
            df[col] = pd.to_numeric(df[col], downcast="integer")


def check_covariance_matrix(bundle: BinBundle) -> None:
    """Check if covariance matrix is positive semi-definite and symmetric"""
    cov = bundle.covariance
    if cov is None:
        return

    matrix = cov.frame.select_dtypes(include=[np.number])

    if cov.frame.shape[0] != matrix.shape[1]:
        warnings.warn(
            f"[{bundle.bin_id}] Covariance matrix is not square."
            f" Shape: {matrix.shape}",
            UserWarning,
        )
        return

    if not np.allclose(matrix.values, matrix.values.T):
        warnings.warn(
            f"[{bundle.bin_id}] Covariance matrix is not symmetric.",
            UserWarning,
        )

    eigenvalues = np.linalg.eigvalsh(matrix.values)
    if np.any(eigenvalues < 0):
        warnings.warn(
            f"[{bundle.bin_id}] Covariance matrix is not positive semi-definite.",
            UserWarning,
        )


def check_correlation_matrix(bundle: BinBundle) -> None:
    """Check if correlation matrix is symmetric and has values in [-1, 1]"""
    corr = bundle.correlation
    if corr is None:
        return

    matrix = corr.frame.select_dtypes(include=[np.number])

    if matrix.shape[0] != matrix.shape[1]:
        warnings.warn(
            f"[{bundle.bin_id}] Correlation matrix is not square."
            f" Shape: {matrix.shape}",
            UserWarning,
        )
        return

    if not np.allclose(matrix.values, matrix.values.T):
        warnings.warn(
            f"[{bundle.bin_id}] Correlation matrix is not symmetric.",
            UserWarning,
        )

    if not np.all((matrix.values >= -1) & (matrix.values <= 1)):
        warnings.warn(
            f"[{bundle.bin_id}] Correlation matrix has values outside [-1, 1].",
            UserWarning,
        )


def check_normalization_integral_matrix(bundle: BinBundle) -> None:
    """Check if the normInt is square and hermitian."""
    norm_int = bundle.norm_int
    if norm_int is None:
        return

    matrix = norm_int.frame.select_dtypes(include=[np.complexfloating])

    if matrix.shape[0] != matrix.shape[1]:
        warnings.warn(
            f"[{bundle.bin_id}] Normalization integral matrix is not square."
            f" Shape: {matrix.shape}",
            UserWarning,
        )
        return

    if not np.allclose(matrix.values, matrix.values.conj().T):
        warnings.warn(
            f"[{bundle.bin_id}] Normalization integral matrix is not Hermitian.",
            UserWarning,
        )
