"""Tests for the binning module of pwassist/io/binning.py"""

from pathlib import Path

import pandas as pd
import pytest

from pwassist.io.binning import BinBundle, BinCollection, MassBin
from pwassist.io.catalog import Catalog, DataFile, FitFile


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
def sample_catalog(tmp_path: Path) -> Catalog:
    """Fixture for a sample Catalog instance."""

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

    catalog = Catalog(tmp_path)
    catalog.scan()  # Populate the catalog with the test files

    return catalog


@pytest.fixture
def sample_manifest(sample_catalog: Catalog) -> pd.DataFrame:
    """Fixture for a sample manifest DataFrame from a catalog scan."""
    return sample_catalog.manifest


class TestBinBundle:
    """Unit tests for the BinBundle class."""

    def test_bin_bundle_creation(self, sample_manifest):
        """Test creating a BinBundle with valid inputs."""
        bundles = [v for k, v in BinCollection(sample_manifest)]
        assert len(bundles) == 2
        assert bundles[0].mass_bin.low == 1.0
        assert bundles[0].mass_bin.high == 1.1
        assert bundles[0].bin_id == "mass_1.0-1.1"
        assert bundles[0].paths["FitFile"].name == "fit.csv"
        assert bundles[0].paths["DataFile"].name == "data.csv"
        assert bundles[0]._loaded == {}  # Not loaded yet

    def test_loading_results_files(self, sample_manifest):
        """Test loading ResultsFile instances from a BinBundle."""
        collection = BinCollection(sample_manifest)
        for mass_bin, bundle in collection:
            assert bundle.get("FitFile") is not None
            assert bundle.get(DataFile) is not None
            assert bundle.get("CorrelationFile") is None  # Not present

            with pytest.raises(TypeError):
                bundle.get(123)  # Invalid type # type: ignore
            # TODO: Run get again on FitFile but assert from_path is not called again
            # (path ResultsFile.from_path and check call count)

    def test_unload_results_files(self, sample_manifest):
        """Test unloading ResultsFile instances from a BinBundle."""
        collection = BinCollection(sample_manifest)

        # for first bundle, unload individually by class and string
        bundle = list(collection)[0][1]

        # Load FitFile and DataFile
        fit_file = bundle.get("FitFile")
        data_file = bundle.get(DataFile)
        assert fit_file is not None
        assert data_file is not None
        assert len(bundle._loaded) == 2

        # Unload FitFile by class type
        bundle.unload(FitFile)
        assert "FitFile" not in bundle._loaded
        assert "DataFile" in bundle._loaded

        # Unload DataFile by string name
        bundle.unload("DataFile")
        assert "DataFile" not in bundle._loaded

        # Unload all (should be empty now)
        bundle.unload()
        assert len(bundle._loaded) == 0

        # For second bundle, test unloading all at once
        bundle2 = list(collection)[1][1]
        # Load FitFile and DataFile
        bundle2.get("FitFile")
        bundle2.get(DataFile)
        assert len(bundle2._loaded) == 2

        bundle2.unload()  # Unload all
        assert len(bundle2._loaded) == 0

        with pytest.raises(TypeError):
            bundle2.unload(123)  # Invalid type # type: ignore


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

        # test that __iter__ method returns the correct mass bins and bundles
        for mass_bin, bundle in collection:
            if mass_bin == first_bin:
                assert bundle.bin_id == "mass_1.0-1.1"
            elif mass_bin == second_bin:
                assert bundle.bin_id == "mass_1.1-1.2"

            assert bundle.fit is not None
            assert bundle.data is not None

    def test_from_catalog(self, sample_catalog: Catalog):
        """Test creating a BinCollection from a Catalog instance."""
        collection = BinCollection.from_catalog(sample_catalog)
        assert len(collection) == 2  # Two unique mass bins

        # perform a few similar checks as in test_from_manifest
        first_bin = MassBin.from_bin_id("mass_1.0-1.1")
        second_bin = MassBin.from_bin_id("mass_1.1-1.2")
        assert collection[first_bin].fit.frame["likelihood"].iloc[0] == -1234.5
        assert collection[first_bin].data.frame["events"].iloc[0] == 1000
        assert collection[second_bin].fit.frame["likelihood"].iloc[0] == -2345.6
        assert collection[second_bin].data.frame["events"].iloc[0] == 2000
