"""Tests for the binning module of pwassist/io/binning.py"""

from pathlib import Path

import pandas as pd
import pytest

from pwassist.io.binning import BinBundle, BinCollection, MassBin


class TestMassBin:
    """Unit tests for the MassBin class."""

    def test_from_bin_id_valid(self):
        """Test creating a MassBin from valid bin ID."""
        bin_id = "mass_1.0-1.1"
        mass_bin = MassBin.from_bin_id(bin_id)
        assert mass_bin.low == 1.0
        assert mass_bin.high == 1.1
        assert mass_bin.center == pytest.approx(1.05)
        assert mass_bin.width == pytest.approx(0.1)
        assert str(mass_bin) == "1.0-1.1"

    def test_from_bin_id_missing_high(self):
        """Test creating a MassBin from an invalid bin ID."""
        with pytest.raises(ValueError):
            MassBin.from_bin_id("mass_1.0-")  # Missing high value

    def test_from_bin_id_missing_low(self):
        with pytest.raises(ValueError):
            MassBin.from_bin_id("mass_-1.1")  # Missing low value

    def test_from_bin_id_invalid_format(self):
        with pytest.raises(ValueError):
            MassBin.from_bin_id("mass_1.0-1.1-1.2")  # Too many values

    def test_lt_comparison(self):
        """Test comparison of MassBin."""
        bin1 = MassBin(1.0, 1.1)
        bin2 = MassBin(1.1, 1.2)
        assert bin1 < bin2


@pytest.fixture
def sample_manifest(tmp_path: Path):
    """Fixture for a sample manifest DataFrame from a catalog scan."""

    # Create temp fit and data CSV files for two mass bins
    fit1 = tmp_path / "mass_1.0-1.1" / "fit.csv"
    fit1.parent.mkdir(parents=True, exist_ok=True)
    fit1.write_text("likelihood,eMatrixStatus,intensity,parameter\n-1234.5,0,10.0,p1\n")
    data1 = tmp_path / "mass_1.0-1.1" / "data.csv"
    data1.write_text("events,efficiency,m_low,m_high\n1000,0.05,1.00,1.10\n")

    fit2 = tmp_path / "mass_1.1-1.2" / "fit.csv"
    fit2.parent.mkdir(parents=True, exist_ok=True)
    fit2.write_text("likelihood,eMatrixStatus,intensity,parameter\n-2345.6,0,20.0,p1\n")
    data2 = tmp_path / "mass_1.1-1.2" / "data.csv"
    data2.write_text("events,efficiency,m_low,m_high\n2000,0.10,1.10,1.20\n")

    data = {
        "bin_id": ["mass_1.0-1.1", "mass_1.0-1.1", "mass_1.1-1.2", "mass_1.1-1.2"],
        "file_type": ["FitFile", "DataFile", "FitFile", "DataFile"],
        "file_path": [fit1, data1, fit2, data2],
    }
    return pd.DataFrame(data)


class TestBinCollection:
    """Unit tests for the BinCollection class."""

    def test_from_manifest(self, sample_manifest):
        """Test creating a BinCollection from a manifest DataFrame."""
        collection = BinCollection(sample_manifest)
        assert len(collection) == 2  # Two unique mass bins

        # check that we can get bundles components for each mass bin
        first_bin = MassBin.from_bin_id("mass_1.0-1.1")
        second_bin = MassBin.from_bin_id("mass_1.1-1.2")
        assert collection[first_bin].fit.frame["likelihood"].iloc[0] == -1234.5
        assert collection[first_bin].data.frame["events"].iloc[0] == 1000
        assert collection[second_bin].fit.frame["likelihood"].iloc[0] == -2345.6
        assert collection[second_bin].data.frame["events"].iloc[0] == 2000
