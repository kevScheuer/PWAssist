import pathlib

import numpy as np
import pandas as pd
import pytest

import pwassist.preprocessing.steps as steps
from pwassist.io.binning import BinBundle, MassBin
from pwassist.io.catalog import (
    CorrelationFile,
    CovarianceFile,
    DataFile,
    FitFile,
    NormIntFile,
)


def make_bundle(
    path: pathlib.Path,
    fit: pd.DataFrame | None = None,
    data: pd.DataFrame | None = None,
    correlation: pd.DataFrame | None = None,
    covariance: pd.DataFrame | None = None,
    norm_int: pd.DataFrame | None = None,
    bin_id="mass_1.0-1.1",
) -> BinBundle:
    bundle = BinBundle(mass_bin=MassBin.from_bin_id(bin_id), bin_id=bin_id, paths={})
    for name, cls, frame in (
        ("FitFile", FitFile, fit),
        ("DataFile", DataFile, data),
        ("CorrelationFile", CorrelationFile, correlation),
        ("CovarianceFile", CovarianceFile, covariance),
        ("NormIntFile", NormIntFile, norm_int),
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
            "[mass_1.0-1.1] FitFile contains null values in columns: ['intensity']."
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
            "[mass_1.0-1.1] Fit contains bad error matrix statuses"
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
            "[mass_1.0-1.1] Fit contains bad Minuit statuses"
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
            "[mass_1.0-1.1] Fit contains bad error matrix statuses"
            " (eMatrixStatus != 3) in rows: [1]."
            " Covariance matrix may be unreliable."
        )
        assert str(warning2.message) == (
            "[mass_1.0-1.1] Fit contains bad Minuit statuses"
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
            "[mass_1.0-1.1] Fit contains negative values in error column"
            " 'intensity_err'."
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
                "parameter": [0.5, 1.0],
                "parameter_err": [0.1, float("nan")],  # Non-finite value in second row
            }
        )
        bundle_with_non_finite_err = make_bundle(
            path=tmp_path, fit=fit_with_non_finite_err, data=None
        )
        steps.check_error_columns(bundle_with_non_finite_err)
        assert len(recwarn) == 2
        warning1 = recwarn.pop()
        warning2 = recwarn.pop()
        assert issubclass(warning1.category, UserWarning)
        assert str(warning1.message) == (
            "[mass_1.0-1.1] Fit contains non-finite values in error column"
            " 'intensity_err'."
        )
        assert issubclass(warning2.category, UserWarning)
        assert str(warning2.message) == (
            "[mass_1.0-1.1] Fit contains non-finite values in error column"
            " 'parameter_err'."
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

    def test_file_categorization(self, fit_only_bundle):
        """Test that the file column is made into a category dtype."""
        steps.downcast_numeric_dtypes(fit_only_bundle)
        assert fit_only_bundle.fit.frame["file"].dtype.name == "category"


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
            "[mass_1.0-1.1] Covariance matrix is not square." " Shape: (2, 3)"
        )

    def test_non_symmetric_covariance_matrix(self, tmp_path, recwarn):
        non_symmetric_cov_matrix = pd.DataFrame(
            {"file": ["file1", "file1"], "param1": [2, 6], "param2": [6.3, 18]},
        )

        non_symmetric_cov_bundle = make_bundle(
            path=tmp_path, covariance=non_symmetric_cov_matrix, fit=None, data=None
        )
        steps.check_covariance_matrix(non_symmetric_cov_bundle)
        assert len(recwarn) == 1
        warning = recwarn.pop()
        assert issubclass(warning.category, UserWarning)
        assert str(warning.message) == (
            "[mass_1.0-1.1] Covariance matrix is not symmetric."
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
        assert str(warning.message) == (
            "[mass_1.0-1.1] Covariance matrix is not positive semi-definite."
        )

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
            "[mass_1.0-1.1] Correlation matrix is not square." " Shape: (2, 3)"
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
            "[mass_1.0-1.1] Correlation matrix is not symmetric."
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
            "[mass_1.0-1.1] Correlation matrix has values outside [-1, 1]."
        )

    def test_no_correlation_matrix(self, tmp_path, recwarn):
        """Test that a bundle with no correlation matrix passes without warnings."""
        bundle_no_corr = make_bundle(
            path=tmp_path, correlation=None, fit=None, data=None
        )
        steps.check_correlation_matrix(bundle_no_corr)
        assert len(recwarn) == 0
