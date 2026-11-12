"""Individual preprocessing steps that can act on fit results in a bin bundle"""

import warnings

import numpy as np
import pandas as pd

from pwassist.core.parameters import coefficient_columns, group_by_unique
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

    Checks the final fits, randomized, and bootstrap results. Any null entries are
    ignored, as they are already flagged by check_null_columns.
    """

    def negative_cols(frame: pd.DataFrame) -> list[str]:
        return [
            c
            for c in frame.columns
            if c.endswith("_err") and (frame[c][frame[c].notnull()] < 0).any()
        ]

    def non_finite_cols(frame: pd.DataFrame) -> list[str]:
        return [
            c
            for c in frame.columns
            if c.endswith("_err")
            and not np.isfinite(frame[c][frame[c].notnull()]).all()
        ]

    def warn(
        neg_cols: list[str], non_finite_cols: list[str], bin_id: str, label: str
    ) -> None:
        if neg_cols:
            warnings.warn(
                f"[{bin_id}] {label} contain negative values in error columns:"
                f" {neg_cols}.",
                UserWarning,
            )
        if non_finite_cols:
            warnings.warn(
                f"[{bin_id}] {label} contain non-finite values in error columns:"
                f" {non_finite_cols}.",
                UserWarning,
            )

    fit = bundle.fit

    if fit is None:
        return

    warn(
        negative_cols(fit.frame),
        non_finite_cols(fit.frame),
        bundle.bin_id,
        "Fit results",
    )

    if bundle.randomized is not None:
        warn(
            negative_cols(bundle.randomized.frame),
            non_finite_cols(bundle.randomized.frame),
            bundle.bin_id,
            "Randomized results",
        )

    if bundle.bootstrap is not None:
        warn(
            negative_cols(bundle.bootstrap.frame),
            non_finite_cols(bundle.bootstrap.frame),
            bundle.bin_id,
            "Bootstrap results",
        )


def align_phase_column_names(bundle: BinBundle) -> None:
    """Make phase names consistent across fit, randomized, and bootstrap results

    The order of the names does matter, as it indicates how the phase difference was
    requested from AmpTools. PD(amp1, amp2) == -1.0 * PD(amp2, amp2), and so we must
    incur a sign flip if we flip the names.
    """
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
                bundle.randomized.frame[col] = bundle.randomized.frame[col] * -1.0
                bundle.randomized.frame.rename(columns={col: new_col}, inplace=True)
    if bundle.bootstrap is not None:
        for col in reversed_phase_cols:
            if col in bundle.bootstrap.frame.columns:
                new_col = f"{col.split('_')[1]}_{col.split('_')[0]}"
                bundle.bootstrap.frame[col] = bundle.bootstrap.frame[col] * -1.0
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


# Matrix-valued files keep full precision, since they are inverted / decomposed
# downstream (e.g. covariance matrices in mass-dependent fits)
_FULL_PRECISION_FILE_TYPES: frozenset[str] = frozenset(
    {"CovarianceFile", "CorrelationFile", "NormIntFile"}
)


def downcast_numeric_dtypes(bundle: BinBundle) -> None:
    """Downcast numeric columns to save memory

    Covariance, correlation, and normalization integral matrices are left in full
    precision.
    """
    for label, rf in FILE_TYPE_MAP.items():
        if label in _FULL_PRECISION_FILE_TYPES or bundle.get(rf) is None:
            continue
        df = bundle.get(rf).frame  # type: ignore
        for col in df.select_dtypes(include=["float64"]).columns:
            df[col] = pd.to_numeric(df[col], downcast="float")
        for col in df.select_dtypes(include=["int64"]).columns:
            df[col] = pd.to_numeric(df[col], downcast="integer")


def reduce_covariance_matrices(bundle: BinBundle) -> None:
    """Reduce covariance/correlation matrices to the unique, free fit parameters

    AmpTools writes one row/column per '<reaction>::<sum>::<amplitude>_<part>', even
    when production coefficients are constrained across reactions and sums, and it
    keeps fixed parameters (zero variance). This leaves a rank-deficient matrix that
    cannot be inverted. This step:

    1. Keeps only the first copy of each constrained production coefficient, and
    renames it to the unique name used in the fit dataframe (e.g. 'm1p0S_re').
    Dropped copies are checked to be identical to the kept one.
    2. Drops fixed parameters, identified by a zero diagonal entry.

    Non-production parameters (e.g. 'parScale_<x>') keep their name. The step is
    idempotent: already-reduced matrices are left untouched.
    """
    fit = bundle.fit
    if fit is None:
        return
    unique_coeffs = coefficient_columns(fit.frame.columns)

    for label, rf in (
        ("Covariance", bundle.covariance),
        ("Correlation", bundle.correlation),
    ):
        if rf is None:
            continue
        rf.frame = _reduce_matrix_frame(rf.frame, unique_coeffs, label, bundle.bin_id)


def _reduce_matrix_frame(
    frame: pd.DataFrame, unique_coeffs: list[str], label: str, bin_id: str
) -> pd.DataFrame:
    """Implementation of 'reduce_covariance_matrices' for a single matrix frame."""
    if "parameter" not in frame.columns:
        warnings.warn(
            f"[{bin_id}] {label} matrix has no 'parameter' column, so it could not be"
            " reduced.",
            UserWarning,
        )
        return frame

    params = frame["parameter"].astype(str).tolist()
    missing = [p for p in params if p not in frame.columns]
    if missing:
        warnings.warn(
            f"[{bin_id}] {label} matrix is missing the columns {missing[:5]}"
            f"{'...' if len(missing) > 5 else ''}, so it could not be reduced.",
            UserWarning,
        )
        return frame

    matrix = frame[params].to_numpy(dtype=np.float64)
    row_of = {p: i for i, p in enumerate(params)}

    # 1. de-duplicate constrained copies of production coefficients
    groups = group_by_unique([p for p in params if "::" in p], unique_coeffs)
    rename: dict[str, str] = {}
    dropped: set[str] = set()
    mismatched: list[str] = []
    for unique, fulls in groups.items():
        if not fulls:
            continue
        kept, *copies = fulls
        rename[kept] = unique
        dropped.update(copies)
        kept_row = matrix[row_of[kept]]
        scale = np.max(np.abs(kept_row)) if kept_row.size else 0.0
        for copy in copies:
            if not np.allclose(
                matrix[row_of[copy]], kept_row, rtol=1e-6, atol=1e-9 * scale
            ):
                mismatched.append(copy)
    if mismatched:
        warnings.warn(
            f"[{bin_id}] {label} rows for {mismatched[:5]}"
            f"{'...' if len(mismatched) > 5 else ''} differ from the first copy of"
            " the same production coefficient, though they share a name in the fit"
            " results. Only the first copy is kept.",
            UserWarning,
        )

    # 2. drop fixed parameters
    fixed = {p for i, p in enumerate(params) if matrix[i, i] == 0.0}

    keep = [p for p in params if p not in dropped and p not in fixed]
    if len(keep) == len(params):
        return frame  # nothing to reduce

    keep_rows = [row_of[p] for p in keep]
    keep_set = set(keep)
    columns = [c for c in frame.columns if c not in row_of or c in keep_set]
    reduced = frame.iloc[keep_rows][columns].reset_index(drop=True)
    reduced[keep] = reduced[keep].astype(np.float64)
    reduced["parameter"] = [rename.get(p, p) for p in keep]
    return reduced.rename(columns=rename)


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

    def _psd_tolerance(eigenvalues: np.ndarray) -> float:
        rtol = eigenvalues.size * np.finfo(float).eps
        return rtol * np.max(np.abs(eigenvalues))

    values = matrix.values
    sym = (values + values.T) / 2.0  # remove rounding asymmetry
    eigenvalues = np.linalg.eigvalsh(sym)

    tol = _psd_tolerance(eigenvalues)
    if np.any(eigenvalues < -tol):
        warnings.warn(
            f"[{bundle.bin_id}] Covariance matrix is not positive semi-definite:"
            f" min eigenvalue {eigenvalues.min():.3e}"
            f" (tolerance -{tol:.3e}, max eigenvalue {eigenvalues.max():.3e})",
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
