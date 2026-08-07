import pathlib

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
                "likelihood": [-1234.5],
                "eMatrixStatus": [3],
                "lastMinuitCommandStatus": [0],
                "intensity": [10.0],
            }
        ),
    )


class TestCheckNullColumns:

    fit_with_nulls = pd.DataFrame(
        {
            "likelihood": [-1234.5, -2345.6],
            "eMatrixStatus": [3, 3],
            "lastMinuitCommandStatus": [0, 0],
            "intensity": [10.0, None],  # Null value in intensity column
        }
    )

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
        bundle_with_nulls = make_bundle(
            path=tmp_path, fit=self.fit_with_nulls, data=None
        )
        steps.check_null_columns(bundle_with_nulls)
        assert len(recwarn) == 1
        warning = recwarn.pop()
        assert issubclass(warning.category, UserWarning)
        assert str(warning.message) == (
            "[mass_1.0-1.1] FitFile contains null values in columns: ['intensity']."
        )
