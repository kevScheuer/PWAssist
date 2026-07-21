from pathlib import Path

import pandas as pd
import pytest

from pwassist.io import Catalog


@pytest.fixture
def sample_bins(tmp_path: Path):

    # Create a sample mass bin with all required and optional files to test the catalog
    # scanning
    bin_dir = tmp_path / "mass_1.0-1.1"
    bin_dir.mkdir()
    (bin_dir / "fit.csv").write_text(
        "likelihood,eMatrixStatus,intensity,parameter\n-1234.5,0,10.0,p1\n"
    )
    (bin_dir / "data.csv").write_text(
        "events,efficiency,m_low,m_high\n1000,0.05,1.00,1.10\n"
    )
    (bin_dir / "correlation.csv").write_text(
        "file,parameter,p1,p2\n" "fit.csv,p1,1.0,0.3\n" "fit.csv,p2,0.3,1.0\n"
    )
    (bin_dir / "covariance.csv").write_text("file,parameter,p1,\nfit.csv,p1,10.0\n")
    (bin_dir / "normint.csv").write_text(
        "file,amplitude,amp1,amp2\nfit.csv,amp1,8+0j,2-1j\n"
    )

    # Create a second bin that only contains the basic required files (fit and data) to
    # test optional file handling
    bin_dir2 = tmp_path / "mass_1.1-1.2"
    bin_dir2.mkdir()
    (bin_dir2 / "fit.csv").write_text(
        "likelihood,eMatrixStatus,intensity,parameter\n-2345.6,0,20.0,p1\n"
    )
    (bin_dir2 / "data.csv").write_text(
        "events,efficiency,m_low,m_high\n2000,0.10,1.10,1.20\n"
    )
    return tmp_path


@pytest.fixture
def bin_missing_required_files(tmp_path: Path):
    # Create a mass bin that is missing required files to test error handling
    bin_dir = tmp_path / "mass_1.2-1.3"
    bin_dir.mkdir()
    (bin_dir / "fit.csv").write_text(
        "likelihood,eMatrixStatus,intensity,parameter\n-3456.7,0,30.0,p1\n"
    )
    # Missing data.csv file
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

    def test_scan_requires_files(self, bin_missing_required_files):
        catalog = Catalog(bin_missing_required_files)
        with pytest.raises(FileNotFoundError):
            catalog.scan()

    def test_scan(self, sample_bins):
        catalog = Catalog(sample_bins)
        manifest = catalog.scan()
        assert isinstance(manifest, pd.DataFrame)

        assert len(manifest) == 7

        first_bin_id = "mass_1.0-1.1"
        second_bin_id = "mass_1.1-1.2"

        for index, row in manifest.iterrows():
            # ensure that file types match the expected files in the sample_bins fixture
            file_type_to_name = {
                "FitFile": "fit.csv",
                "DataFile": "data.csv",
                "CorrelationFile": "correlation.csv",
                "CovarianceFile": "covariance.csv",
                "NormIntFile": "normint.csv",
            }
            if row["bin_id"] == first_bin_id:
                file_type = row["file_type"]
                expected_file_name = file_type_to_name[file_type]
                assert row["file_path"] == str(
                    sample_bins / first_bin_id / expected_file_name
                )

            # only the first bin has optional files, so we can check that the second bin
            # only has required files
            elif row["bin_id"] == second_bin_id:
                assert row["file_type"] in ["FitFile", "DataFile"]
                assert row["file_path"] == str(
                    sample_bins / second_bin_id / file_type_to_name[row["file_type"]]
                )

            else:
                pytest.fail(f"Unexpected bin_id in manifest: {row['bin_id']}")
