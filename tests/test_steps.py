import pathlib

import numpy as np
import pandas as pd
import pytest

import pwassist.preprocessing.steps as steps
from pwassist.io.binning import BinBundle, KinematicBin, MassBin
from pwassist.io.catalog import (
    BootstrapFile,
    CorrelationFile,
    CovarianceFile,
    DataFile,
    FitFile,
    NormIntFile,
    RandomizedFile,
)


def make_bundle(
    path: pathlib.Path,
    fit: pd.DataFrame | None = None,
    data: pd.DataFrame | None = None,
    correlation: pd.DataFrame | None = None,
    covariance: pd.DataFrame | None = None,
    norm_int: pd.DataFrame | None = None,
    randomized: pd.DataFrame | None = None,
    bootstrap: pd.DataFrame | None = None,
    bin_id="T=0.1,0.2-E=8.2,8.8-M=1.0,1.1",
) -> BinBundle:
    bundle = BinBundle(
        kinematic_bin=KinematicBin.from_bin_id(bin_id), bin_id=bin_id, paths={}
    )
    for name, cls, frame in (
        ("FitFile", FitFile, fit),
        ("DataFile", DataFile, data),
        ("CorrelationFile", CorrelationFile, correlation),
        ("CovarianceFile", CovarianceFile, covariance),
        ("NormIntFile", NormIntFile, norm_int),
        ("RandomizedFile", RandomizedFile, randomized),
        ("BootstrapFile", BootstrapFile, bootstrap),
    ):
        if frame is not None:
            dummy_path = pathlib.Path(path / f"{name}.csv")
            bundle.paths[name] = dummy_path
            bundle._loaded[name] = cls(path=dummy_path, frame=frame)
    return bundle


@pytest.fixture
def fit_only_bundle(tmp_path: pathlib.Path) -> BinBundle:
    """A fixture that provides a BinBundle with only a fit file."""
    return make_bundle(
        path=tmp_path,
        fit=pd.DataFrame(
            {
                "file": str(tmp_path / "FitFile.csv"),
                "likelihood": [-1234.5],
                "eMatrixStatus": [3],
                "lastMinuitCommandStatus": [0],
                "intensity": [10.0],
            }
        ),
    )


class TestCheckNullColumns:

    def test_good_bundle(self, fit_only_bundle, recwarn):
        """Test that a bundle with no null columns passes without warnings."""
        # Should not raise any warnings
        steps.check_null_columns(fit_only_bundle)
        assert len(recwarn) == 0

    def test_empty_bundle(self, recwarn):
        """Test that an empty bundle passes with no warnings."""
        empty_bundle = make_bundle(path=pathlib.Path("."), fit=None, data=None)
        steps.check_null_columns(empty_bundle)
        assert len(recwarn) == 0

    def test_bundle_with_nulls(self, tmp_path, recwarn):
        """Test that a bundle with null columns raises a warning."""
        fit_with_nulls = pd.DataFrame(
            {
                "likelihood": [-1234.5, -2345.6],
                "eMatrixStatus": [3, 3],
                "lastMinuitCommandStatus": [0, 0],
                "intensity": [10.0, None],  # Null value in intensity column
            }
        )
        bundle_with_nulls = make_bundle(path=tmp_path, fit=fit_with_nulls, data=None)
        steps.check_null_columns(bundle_with_nulls)
        assert len(recwarn) == 1
        warning = recwarn.pop()
        assert issubclass(warning.category, UserWarning)
        assert str(warning.message) == (
            "[T=0.1,0.2-E=8.2,8.8-M=1.0,1.1] FitFile contains null values in columns:"
            " ['intensity']."
        )


class TestFitStatus:

    def test_good_fit_status(self, fit_only_bundle, recwarn):
        """Test that a bundle with good fit statuses passes without warnings."""
        steps.check_fit_status(fit_only_bundle)
        assert len(recwarn) == 0

    def test_bad_error_matrix_status(self, tmp_path, recwarn):
        """Test that a bundle with bad error matrix statuses raises a warning."""
        fit_with_bad_matrix = pd.DataFrame(
            {
                "likelihood": [-1234.5, -2345.6],
                "eMatrixStatus": [3, 2],  # Bad status in second row
                "lastMinuitCommandStatus": [0, 0],
                "intensity": [10.0, 20.0],
            }
        )
        bundle_with_bad_matrix = make_bundle(
            path=tmp_path, fit=fit_with_bad_matrix, data=None
        )
        steps.check_fit_status(bundle_with_bad_matrix)
        assert len(recwarn) == 1
        warning = recwarn.pop()
        assert issubclass(warning.category, UserWarning)
        assert str(warning.message) == (
            "[T=0.1,0.2-E=8.2,8.8-M=1.0,1.1] Fit contains bad error matrix statuses"
            " (eMatrixStatus != 3) in rows: [1]."
            " Covariance matrix may be unreliable."
        )

    def test_bad_minuit_status(self, tmp_path, recwarn):
        """Test that a bundle with bad Minuit statuses raises a warning."""
        fit_with_bad_minuit = pd.DataFrame(
            {
                "likelihood": [-1234.5, -2345.6],
                "eMatrixStatus": [3, 3],
                "lastMinuitCommandStatus": [0, 1],  # Bad status in second row
                "intensity": [10.0, 20.0],
            }
        )
        bundle_with_bad_minuit = make_bundle(
            path=tmp_path, fit=fit_with_bad_minuit, data=None
        )
        steps.check_fit_status(bundle_with_bad_minuit)
        assert len(recwarn) == 1
        warning = recwarn.pop()
        assert issubclass(warning.category, UserWarning)
        assert str(warning.message) == (
            "[T=0.1,0.2-E=8.2,8.8-M=1.0,1.1] Fit contains bad Minuit statuses"
            " (lastMinuitCommandStatus != 0) in rows: [1]."
            " Fit may not have converged."
        )

    def test_bad_both_statuses(self, tmp_path, recwarn):
        """Test that a bundle with both bad statuses raises two warnings."""
        fit_with_bad_both = pd.DataFrame(
            {
                "likelihood": [-1234.5, -2345.6],
                "eMatrixStatus": [3, 2],  # Bad status in second row
                "lastMinuitCommandStatus": [0, 1],  # Bad status in second row
                "intensity": [10.0, 20.0],
            }
        )
        bundle_with_bad_both = make_bundle(
            path=tmp_path, fit=fit_with_bad_both, data=None
        )
        steps.check_fit_status(bundle_with_bad_both)
        assert len(recwarn) == 2
        warning1 = recwarn.pop()
        warning2 = recwarn.pop()
        assert issubclass(warning1.category, UserWarning)
        assert issubclass(warning2.category, UserWarning)
        assert str(warning1.message) == (
            "[T=0.1,0.2-E=8.2,8.8-M=1.0,1.1] Fit contains bad error matrix statuses"
            " (eMatrixStatus != 3) in rows: [1]."
            " Covariance matrix may be unreliable."
        )
        assert str(warning2.message) == (
            "[T=0.1,0.2-E=8.2,8.8-M=1.0,1.1] Fit contains bad Minuit statuses"
            " (lastMinuitCommandStatus != 0) in rows: [1]."
            " Fit may not have converged."
        )

    def test_no_fit_file(self, tmp_path, recwarn):
        """Test that a bundle with no fit file passes without warnings."""
        bundle_no_fit = make_bundle(path=tmp_path, fit=None, data=None)
        steps.check_fit_status(bundle_no_fit)
        assert len(recwarn) == 0

    def test_missing_columns(self, tmp_path, recwarn):
        """Test that a bundle with missing columns passes without warnings."""
        fit_missing_columns = pd.DataFrame(
            {
                "likelihood": [-1234.5, -2345.6],
                "intensity": [10.0, 20.0],
            }
        )
        bundle_missing_columns = make_bundle(
            path=tmp_path, fit=fit_missing_columns, data=None
        )
        steps.check_fit_status(bundle_missing_columns)
        assert len(recwarn) == 0


class TestErrorColumns:

    def test_good_error_columns(self, fit_only_bundle, recwarn):
        """Test that a bundle with good error columns passes without warnings."""
        steps.check_error_columns(fit_only_bundle)
        assert len(recwarn) == 0

    def test_negative_error_column(self, tmp_path, recwarn):
        """Test that a bundle with negative values in error columns raises a warning."""
        fit_with_negative_err = pd.DataFrame(
            {
                "likelihood": [-1234.5, -2345.6],
                "eMatrixStatus": [3, 3],
                "lastMinuitCommandStatus": [0, 0],
                "intensity": [10.0, 20.0],
                "intensity_err": [1.0, -2.0],  # Negative value in second row
            }
        )
        bundle_with_negative_err = make_bundle(
            path=tmp_path, fit=fit_with_negative_err, data=None
        )
        steps.check_error_columns(bundle_with_negative_err)
        assert len(recwarn) == 1
        warning = recwarn.pop()
        assert issubclass(warning.category, UserWarning)
        assert str(warning.message) == (
            "[T=0.1,0.2-E=8.2,8.8-M=1.0,1.1] Fit results contain negative values in"
            " error columns: ['intensity_err']."
        )

    def test_non_finite_error_column(self, tmp_path, recwarn):
        """Test that a bundle with non-finite values in error cols raises a warning."""
        fit_with_non_finite_err = pd.DataFrame(
            {
                "likelihood": [-1234.5, -2345.6],
                "eMatrixStatus": [3, 3],
                "lastMinuitCommandStatus": [0, 0],
                "intensity": [10.0, 20.0],
                "intensity_err": [1.0, float("inf")],  # Non-finite value in second row
            }
        )
        bundle_with_non_finite_err = make_bundle(
            path=tmp_path, fit=fit_with_non_finite_err, data=None
        )
        steps.check_error_columns(bundle_with_non_finite_err)
        assert len(recwarn) == 1
        warning1 = recwarn.pop()
        assert issubclass(warning1.category, UserWarning)
        assert str(warning1.message) == (
            "[T=0.1,0.2-E=8.2,8.8-M=1.0,1.1] Fit results contain non-finite values in"
            " error columns: ['intensity_err']."
        )

    def test_no_fit_file(self, tmp_path, recwarn):
        """Test that a bundle with no fit file passes without warnings."""
        bundle_no_fit = make_bundle(path=tmp_path, fit=None, data=None)
        steps.check_error_columns(bundle_no_fit)
        assert len(recwarn) == 0

    def test_no_error_columns(self, tmp_path, recwarn):
        """Test that a bundle with no error columns passes without warnings."""
        fit_no_err_cols = pd.DataFrame(
            {
                "likelihood": [-1234.5, -2345.6],
                "eMatrixStatus": [3, 3],
                "lastMinuitCommandStatus": [0, 0],
                "intensity": [10.0, 20.0],
            }
        )
        bundle_no_err_cols = make_bundle(path=tmp_path, fit=fit_no_err_cols, data=None)
        steps.check_error_columns(bundle_no_err_cols)
        assert len(recwarn) == 0


class TestAlignPhaseColumnNames:
    # TODO: create a fit frame with phases, then a randomized frame with some flipped
    # phases, and test that the randomized frame is aligned to the fit frame after
    # running the align_phase_column_names step.
    pass


class TestWrapPhaseColumns:
    def test_phase_wrapping(self, tmp_path, recwarn):
        """Test that phase columns are wrapped to [-pi, pi]."""
        fit_with_phase = pd.DataFrame(
            {
                "likelihood": [-1234.5, -2345.6],
                "eMatrixStatus": [3, 3],
                "lastMinuitCommandStatus": [0, 0],
                "intensity": [10.0, 20.0],
                "1S0+_re": [1.0, 2.0],  # re/im parts garbage vals, just for parsing
                "1S0+_im": [3.5, -4.0],
                "1S1+_re": [0.5, 1.0],
                "1S1+_im": [4.9, 10.2],
                "1S0+_1S1+": [3.5, -4.0],  # Phase values outside [-pi, pi]
                "1S0+_1S1+_err": [4.9, 10.2],  # Corresponding error column
            }
        )
        bundle_with_phase = make_bundle(path=tmp_path, fit=fit_with_phase, data=None)
        steps.wrap_phase_columns(bundle_with_phase)
        wrapped_phases = bundle_with_phase.fit.frame["1S0+_1S1+"].values

        # ensure all wrapped phases are within [-pi, pi]
        assert all(-180.0 <= p <= 180.0 for p in wrapped_phases)

        # ensure that the wrapped values are correct
        degree_values = np.rad2deg(np.angle(np.exp(1j * np.array([3.5, -4.0]))))
        assert wrapped_phases == pytest.approx(degree_values, rel=1e-6)

        # ensure that the error columns are in degrees, correct, but not wrapped
        wrapped_errors = bundle_with_phase.fit.frame["1S0+_1S1+_err"].values
        degree_errors = np.rad2deg(np.array([4.9, 10.2]))
        assert wrapped_errors == pytest.approx(degree_errors, rel=1e-6)

    def test_no_phase_columns(self, fit_only_bundle, recwarn):
        """Test that a bundle with no phase columns passes without warnings."""
        steps.wrap_phase_columns(fit_only_bundle)
        assert len(recwarn) == 0

    def test_no_fit_file(self, tmp_path, recwarn):
        """Test that a bundle with no fit file passes without warnings."""
        bundle_no_fit = make_bundle(path=tmp_path, fit=None, data=None)
        steps.wrap_phase_columns(bundle_no_fit)
        assert len(recwarn) == 0


class TestDowncastNumericDTypes:

    def test_float64_to_float32(self, fit_only_bundle):
        """Test that float64 columns are downcast to float32."""
        # Add a float64 column to the fit frame
        values = [10.0]
        fit_only_bundle.fit.frame["float64_col"] = pd.Series(values, dtype="float64")
        steps.downcast_numeric_dtypes(fit_only_bundle)
        assert fit_only_bundle.fit.frame["float64_col"].dtype == "float32"
        assert fit_only_bundle.fit.frame["float64_col"].values == pytest.approx(
            values, rel=1e-6
        )

    def test_int64_to_int32(self, fit_only_bundle):
        """Test that int64 columns are downcast to lowest possible integer type"""
        # Add an int64 column to the fit frame
        value_int8 = [10]
        fit_only_bundle.fit.frame["int8_col"] = pd.Series(value_int8, dtype="int8")
        steps.downcast_numeric_dtypes(fit_only_bundle)
        assert fit_only_bundle.fit.frame["int8_col"].dtype == "int8"
        assert fit_only_bundle.fit.frame["int8_col"].values.tolist() == value_int8

        value_int16 = [1000]
        fit_only_bundle.fit.frame["int16_col"] = pd.Series(value_int16, dtype="int16")
        steps.downcast_numeric_dtypes(fit_only_bundle)
        assert fit_only_bundle.fit.frame["int16_col"].dtype == "int16"
        assert fit_only_bundle.fit.frame["int16_col"].values.tolist() == value_int16

        value_int32 = [100000]
        fit_only_bundle.fit.frame["int32_col"] = pd.Series(value_int32, dtype="int32")
        steps.downcast_numeric_dtypes(fit_only_bundle)
        assert fit_only_bundle.fit.frame["int32_col"].dtype == "int32"
        assert fit_only_bundle.fit.frame["int32_col"].values.tolist() == value_int32


class TestCheckCovarianceMatrix:

    def test_good_covariance_matrix(self, tmp_path, recwarn):
        cov_matrix = pd.DataFrame(
            {"file": ["file1", "file1"], "param1": [2, 6], "param2": [6, 18]},
        )
        good_cov_bundle = make_bundle(
            path=tmp_path, covariance=cov_matrix, fit=None, data=None
        )
        steps.check_covariance_matrix(good_cov_bundle)
        assert len(recwarn) == 0

    def test_non_square_covariance_matrix(self, tmp_path, recwarn):
        non_square_cov_matrix = pd.DataFrame(
            {
                "file": ["file1", "file1"],
                "param1": [2, 6],
                "param2": [6, 18],
                "param3": [18, 54],
            },
        )

        non_square_cov_bundle = make_bundle(
            path=tmp_path, covariance=non_square_cov_matrix, fit=None, data=None
        )
        steps.check_covariance_matrix(non_square_cov_bundle)
        assert len(recwarn) == 1
        warning = recwarn.pop()
        assert issubclass(warning.category, UserWarning)
        assert str(warning.message) == (
            "[T=0.1,0.2-E=8.2,8.8-M=1.0,1.1] Covariance matrix is not square."
            " Shape: (2, 3)"
        )

    def test_non_symmetric_covariance_matrix(self, tmp_path, recwarn):
        non_symmetric_cov_matrix = pd.DataFrame(
            {"file": ["file1", "file1"], "param1": [2, 6], "param2": [5.7, 18]},
        )

        non_symmetric_cov_bundle = make_bundle(
            path=tmp_path, covariance=non_symmetric_cov_matrix, fit=None, data=None
        )
        steps.check_covariance_matrix(non_symmetric_cov_bundle)
        assert len(recwarn) == 1
        warning = recwarn.pop()
        assert issubclass(warning.category, UserWarning)
        assert str(warning.message) == (
            "[T=0.1,0.2-E=8.2,8.8-M=1.0,1.1] Covariance matrix is not symmetric."
        )

    def test_non_positive_semi_definite_covariance_matrix(self, tmp_path, recwarn):
        non_psd_cov_matrix = pd.DataFrame(
            {"file": ["file1", "file1"], "param1": [0.1, 0.2], "param2": [0.2, -0.3]},
        )

        non_psd_cov_bundle = make_bundle(
            path=tmp_path, covariance=non_psd_cov_matrix, fit=None, data=None
        )
        steps.check_covariance_matrix(non_psd_cov_bundle)
        assert len(recwarn) == 1
        warning = recwarn.pop()
        assert issubclass(warning.category, UserWarning)
        assert (
            "[T=0.1,0.2-E=8.2,8.8-M=1.0,1.1] Covariance matrix is not positive"
            " semi-definite"
        ) in str(warning.message)

    def test_no_covariance_matrix(self, tmp_path, recwarn):
        """Test that a bundle with no covariance matrix passes without warnings."""
        bundle_no_cov = make_bundle(path=tmp_path, covariance=None, fit=None, data=None)
        steps.check_covariance_matrix(bundle_no_cov)
        assert len(recwarn) == 0


class TestCheckCorrelationMatrix:

    def test_good_correlation_matrix(self, tmp_path, recwarn):
        corr_matrix = pd.DataFrame(
            {"file": ["file1", "file1"], "param1": [1, 0.5], "param2": [0.5, 1]},
        )
        good_corr_bundle = make_bundle(
            path=tmp_path, correlation=corr_matrix, fit=None, data=None
        )
        steps.check_correlation_matrix(good_corr_bundle)
        assert len(recwarn) == 0

    def test_non_square_correlation_matrix(self, tmp_path, recwarn):
        non_square_corr_matrix = pd.DataFrame(
            {
                "file": ["file1", "file1"],
                "param1": [1, 0.5],
                "param2": [0.5, 1],
                "param3": [0.2, 0.3],
            },
        )

        non_square_corr_bundle = make_bundle(
            path=tmp_path, correlation=non_square_corr_matrix, fit=None, data=None
        )
        steps.check_correlation_matrix(non_square_corr_bundle)
        assert len(recwarn) == 1
        warning = recwarn.pop()
        assert issubclass(warning.category, UserWarning)
        assert str(warning.message) == (
            "[T=0.1,0.2-E=8.2,8.8-M=1.0,1.1] Correlation matrix is not square."
            " Shape: (2, 3)"
        )

    def test_non_symmetric_correlation_matrix(self, tmp_path, recwarn):
        non_symmetric_corr_matrix = pd.DataFrame(
            {"file": ["file1", "file1"], "param1": [1, 0.5], "param2": [0.6, 1]},
        )

        non_symmetric_corr_bundle = make_bundle(
            path=tmp_path, correlation=non_symmetric_corr_matrix, fit=None, data=None
        )
        steps.check_correlation_matrix(non_symmetric_corr_bundle)
        assert len(recwarn) == 1
        warning = recwarn.pop()
        assert issubclass(warning.category, UserWarning)
        assert str(warning.message) == (
            "[T=0.1,0.2-E=8.2,8.8-M=1.0,1.1] Correlation matrix is not symmetric."
        )

    def test_correlation_matrix_out_of_bounds(self, tmp_path, recwarn):
        out_of_bounds_corr_matrix = pd.DataFrame(
            {"file": ["file1", "file1"], "param1": [1, 1.5], "param2": [1.5, 1]},
        )

        out_of_bounds_corr_bundle = make_bundle(
            path=tmp_path, correlation=out_of_bounds_corr_matrix, fit=None, data=None
        )
        steps.check_correlation_matrix(out_of_bounds_corr_bundle)
        assert len(recwarn) == 1
        warning = recwarn.pop()
        assert issubclass(warning.category, UserWarning)
        assert str(warning.message) == (
            "[T=0.1,0.2-E=8.2,8.8-M=1.0,1.1] Correlation matrix has values outside"
            " [-1, 1] (largest |rho| - 1 = 5.000e-01)."
        )

    def test_correlation_matrix_rounding_tolerated(self, tmp_path, recwarn):
        """float64 rounding just above |rho| = 1 should not warn."""
        rounded_corr_matrix = pd.DataFrame(
            {
                "file": ["file1", "file1"],
                "param1": [1.0000000000000002, -0.5],
                "param2": [-0.5, 0.9999999999999998],
            },
        )
        bundle = make_bundle(
            path=tmp_path, correlation=rounded_corr_matrix, fit=None, data=None
        )
        steps.check_correlation_matrix(bundle)
        assert len(recwarn) == 0

    def test_no_correlation_matrix(self, tmp_path, recwarn):
        """Test that a bundle with no correlation matrix passes without warnings."""
        bundle_no_corr = make_bundle(
            path=tmp_path, correlation=None, fit=None, data=None
        )
        steps.check_correlation_matrix(bundle_no_corr)
        assert len(recwarn) == 0


def make_constrained_covariance(perturb_copy: bool = False) -> pd.DataFrame:
    """Covariance of 2 reactions sharing coefficients 'a' (anchor) and 'b'.

    Unique free parameters: a_re, b_re, b_im, parScale_r2. Fixed: a_im, parScale_r1.
    """
    free = ["a_re", "b_re", "b_im", "parScale_r2"]
    rng = np.random.default_rng(1)
    x = rng.normal(size=(4, 4))
    base = x @ x.T + 4 * np.eye(4)

    params = [
        "r1::s::a_re",
        "r1::s::a_im",
        "r1::s::b_re",
        "r1::s::b_im",
        "r2::s::a_re",
        "r2::s::a_im",
        "r2::s::b_re",
        "r2::s::b_im",
        "parScale_r1",
        "parScale_r2",
    ]
    unique_of = {
        p: p.split("::")[-1] if "::" in p else p for p in params
    }  # full -> unique
    matrix = np.zeros((len(params), len(params)))
    for i, pi in enumerate(params):
        for j, pj in enumerate(params):
            ui, uj = unique_of[pi], unique_of[pj]
            if ui in free and uj in free:
                matrix[i, j] = base[free.index(ui), free.index(uj)]
    if perturb_copy:
        matrix[6, 6] *= 1.5  # r2::s::b_re no longer identical to its r1 copy

    frame = pd.DataFrame(matrix, columns=params)
    frame.insert(0, "parameter", params)
    frame.insert(0, "file", "cov.csv")
    frame["bin_id"] = "T=0.1,0.2-E=8.2,8.8-M=1.0,1.1"
    return frame


@pytest.fixture
def constrained_fit() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a_re": [1.0],
            "a_im": [0.0],
            "b_re": [0.5],
            "b_im": [0.2],
            "parScale_r1": [1.0],
            "parScale_r2": [0.9],
        }
    )


class TestDowncastExemption:

    def test_matrices_keep_float64(self, tmp_path, constrained_fit):
        cov = make_constrained_covariance()
        bundle = make_bundle(
            path=tmp_path, fit=constrained_fit, covariance=cov, correlation=cov.copy()
        )
        assert bundle.covariance is not None and bundle.correlation is not None
        steps.downcast_numeric_dtypes(bundle)
        assert bundle.fit.frame["a_re"].dtype == np.float32
        assert (bundle.covariance.frame["r1::s::a_re"].dtype) == np.float64
        assert (bundle.correlation.frame["r1::s::a_re"].dtype) == np.float64


class TestReduceCovarianceMatrices:

    def test_reduces_to_unique_free_parameters(
        self, tmp_path, constrained_fit, recwarn
    ):
        bundle = make_bundle(
            path=tmp_path, fit=constrained_fit, covariance=make_constrained_covariance()
        )
        steps.reduce_covariance_matrices(bundle)
        assert len(recwarn) == 0
        assert bundle.covariance is not None
        frame = bundle.covariance.frame
        expected = ["a_re", "b_re", "b_im", "parScale_r2"]
        assert frame["parameter"].tolist() == expected
        assert [
            c for c in frame.columns if c not in ("file", "parameter", "bin_id")
        ] == (expected)
        matrix = frame[expected].to_numpy()
        assert matrix.dtype == np.float64
        assert np.linalg.matrix_rank(matrix) == len(expected)
        assert np.all(np.linalg.eigvalsh(matrix) > 0)

    def test_idempotent(self, tmp_path, constrained_fit):
        bundle = make_bundle(
            path=tmp_path, fit=constrained_fit, covariance=make_constrained_covariance()
        )
        assert bundle.covariance is not None
        steps.reduce_covariance_matrices(bundle)
        once = bundle.covariance.frame.copy()
        steps.reduce_covariance_matrices(bundle)
        pd.testing.assert_frame_equal(once, bundle.covariance.frame)

    def test_warns_on_differing_copies(self, tmp_path, constrained_fit):
        bundle = make_bundle(
            path=tmp_path,
            fit=constrained_fit,
            covariance=make_constrained_covariance(perturb_copy=True),
        )
        assert bundle.covariance is not None
        with pytest.warns(UserWarning, match="differ from the first copy"):
            steps.reduce_covariance_matrices(bundle)
        assert "r2::s::b_re" not in bundle.covariance.frame.columns

    def test_correlation_is_reduced(self, tmp_path, constrained_fit):
        cov = make_constrained_covariance()
        params = cov["parameter"].tolist()
        matrix = cov[params].to_numpy()
        sigma = np.sqrt(np.diag(matrix))
        with np.errstate(divide="ignore", invalid="ignore"):
            corr_values = np.nan_to_num(matrix / np.outer(sigma, sigma))
        corr = cov.copy()
        corr[params] = corr_values

        bundle = make_bundle(path=tmp_path, fit=constrained_fit, correlation=corr)
        assert bundle.correlation is not None
        steps.reduce_covariance_matrices(bundle)
        frame = bundle.correlation.frame
        assert frame["parameter"].tolist() == ["a_re", "b_re", "b_im", "parScale_r2"]
        np.testing.assert_allclose(np.diag(frame[frame["parameter"]].to_numpy()), 1.0)

    def test_no_fit_or_matrices(self, tmp_path, recwarn):
        steps.reduce_covariance_matrices(make_bundle(path=tmp_path))
        assert len(recwarn) == 0
