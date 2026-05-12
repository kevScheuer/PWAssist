from pathlib import Path

import pytest

from pwassist.io import Catalog


@pytest.fixture
def sample_bins(tmp_path: Path):
    bin_dir = tmp_path / "mass_1.0-1.1"
    bin_dir.mkdir()
    (bin_dir / "fit.csv").write_text(
        "likelihood,eMatrixStatus,intensity,parameter\n-1234.5,0,10.0,p1\n"
    )
    (bin_dir / "data.csv").write_text(
        "events,efficiency,m_low,m_high\n1000,0.05,1.00,1.10\n"
    )
    (bin_dir / "correlation.csv").write_text("file,parameter,p1,\nfit.csv,p1,1.0\n")
    (bin_dir / "covariance.csv").write_text("file,parameter,p1,\nfit.csv,p1,10.0\n")
    (bin_dir / "normint.csv").write_text(
        "file,amplitude,amp1,amp2\nfit.csv,amp1,8+0j,2-1j\n"
    )
    return tmp_path


class TestIdentifyFileType:

    def test_identify_fit_file(self, sample_bins):
        catalog = Catalog(sample_bins)
        fit_file = sample_bins / "mass_1.0-1.1" / "fit.csv"
        assert catalog.identify_file_type(fit_file).__name__ == "FitFile"

    def test_identify_data_file(self, sample_bins):
        catalog = Catalog(sample_bins)
        data_file = sample_bins / "mass_1.0-1.1" / "data.csv"
        assert catalog.identify_file_type(data_file).__name__ == "DataFile"

    def test_identify_correlation_file(self, sample_bins):
        catalog = Catalog(sample_bins)
        corr_file = sample_bins / "mass_1.0-1.1" / "correlation.csv"
        assert catalog.identify_file_type(corr_file).__name__ == "CorrelationFile"

    def test_identify_covariance_file(self, sample_bins):
        catalog = Catalog(sample_bins)
        cov_file = sample_bins / "mass_1.0-1.1" / "covariance.csv"
        assert catalog.identify_file_type(cov_file).__name__ == "CovarianceFile"

    def test_identify_normint_file(self, sample_bins):
        catalog = Catalog(sample_bins)
        normint_file = sample_bins / "mass_1.0-1.1" / "normint.csv"
        assert catalog.identify_file_type(normint_file).__name__ == "NormIntFile"

    def test_identify_unknown_file(self, sample_bins):
        catalog = Catalog(sample_bins)
        unknown_file = sample_bins / "mass_1.0-1.1" / "unknown.csv"
        unknown_file.write_text("some,random,columns\n1,2,3\n")
        with pytest.raises(ValueError):
            catalog.identify_file_type(unknown_file)

    # TODO: add some more tests for corr / cov matrices, since those depend on values
    # not just columns. Maybe NaNs, strings, or other things in numeric columns?

    # TODO: add a test for distinguishing corr / cov when cov matrix has values in
    # [-1, 1] (which is technically possible but unlikely in practice).
    # I might have to instead add a check in the original method. Maybe look for
    # all 1's on the diagonal of the corr matrix?


# TODO: Make a test class for the scan method
