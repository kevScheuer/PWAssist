"""Individual preprocessing steps that can act on fit results in a bin bundle"""

import warnings

import numpy as np
import pandas as pd

from pwassist.io.binning import BinBundle


def check_null_columns(bundle: BinBundle) -> None:
    """Check if null columns exist in the fit or data files of a bundle"""
    for label, rf in (("fit", bundle.fit), ("data", bundle.data)):
        if rf is None:
            continue
        null_cols = rf.frame.columns[rf.frame.isnull().any()].tolist()
        if null_cols:
            warnings.warn(
                f"[{bundle.bin_id}] {label} file contains null values in columns: {null_cols}",
                UserWarning,
            )


def check_fit_status(bundle: BinBundle) -> None:
    """Flag fits with bad Minuit or error matrix statuses"""
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
    """Ensure '_err' columns are non-negative and finite"""
    fit = bundle.fit

    if fit is None:
        return

    err_cols = [c for c in fit.frame.columns if c.endswith("_err")]
    for col in err_cols:
        series = fit.frame[col]
        if (series < 0).any():
            warnings.warn(
                f"[{bundle.bin_id}] Fit contains negative values in error column"
                f" '{col}'.",
                UserWarning,
            )
        if not np.isfinite(series).all():
            warnings.warn(
                f"[{bundle.bin_id}] Fit contains non-finite values in error column"
                f" '{col}'.",
                UserWarning,
            )


def wrap_phase_columns(bundle: BinBundle) -> None:
    """Wrap phase columns (in radians) to the range (-180, 180] in degrees

    Todo:
        - Right now no phase difference identification utility is implemented, as this
            depends on configuring some form of naming scheme convention for the
            amplitude names. So currently we just check if the column name is in the
            JLme_JLme format by counting the characters after splitting on "_"
    """
    fit = bundle.fit
    if fit is None:
        return

    phase_cols = [
        c
        for c in fit.frame.columns
        if (
            "_" in c and len(c.split("_")[0]) == 4 and len(c.split("_")[1]) == 4
        )  # JLme
        or (
            "_" in c and len(c.split("_")[0]) == 5 and len(c.split("_")[1]) == 5
        )  # eJPmL
    ]
    phase_err_cols = [
        f"{c}_err" for c in phase_cols if f"{c}_err" in fit.frame.columns
    ]  # We won't wrap them, but they need to be converted to degrees

    if not phase_cols:
        warnings.warn(
            f"[{bundle.bin_id}] No phase columns found in fit file to wrap.",
            UserWarning,
        )
        return

    for col in phase_cols:
        fit.frame[col] = np.rad2deg(
            np.angle(np.exp(1j * fit.frame[col]))
        )  # Wrap to (-pi, pi] using complex exponential
    for col in phase_err_cols:
        fit.frame[col] = np.rad2deg(fit.frame[col])  # Convert to degrees


def downcast_numeric_dtypes(bundle: BinBundle) -> None:
    """Downcast numeric columns to save memory"""
    for label, rf in (("fit", bundle.fit), ("data", bundle.data)):
        if rf is None:
            continue
        df = rf.frame
        for col in df.select_dtypes(include=["float64"]).columns:
            df[col] = pd.to_numeric(df[col], downcast="float")
        for col in df.select_dtypes(include=["int64"]).columns:
            df[col] = pd.to_numeric(df[col], downcast="integer")
        if "file" in df.columns:
            df["file"] = df["file"].astype("category")


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
