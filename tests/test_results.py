import dataclasses
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

from pwassist.core.result import Results
from pwassist.io.binning import BinCollection
from pwassist.io.catalog import Catalog
from pwassist.preprocessing.preprocessor import Preprocessor

EXAMPLE_DATA = pathlib.Path(__file__).resolve().parents[1] / "examples" / "data"

# The example results (eJPmL naming): 18 unique coefficients, the anchors m1mpP and
# p1mpP are fixed real, and the first parScale is fixed.
N_FREE_PARAMETERS = 18 * 2 - 2 + 3
REFERENCES = [("m1p0S", "m1pmS", -1), ("p1p0S", "p1pmS", -1)]


@pytest.fixture(scope="module")
def results() -> Results:
    catalog = Catalog(EXAMPLE_DATA)
    catalog.scan()
    preprocessor = Preprocessor()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        processed = [
            preprocessor.run(b) for _, b in BinCollection.from_catalog(catalog)
        ]
    return Results.from_processed_bins(processed, final_state_parity=1)


@pytest.fixture(scope="module")
def aligned(results: Results) -> Results:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ambiguous reference phase warnings
        return results.align_conjugation(REFERENCES)


def phase(coefficients: pd.DataFrame, amp1: str, amp2: str) -> np.ndarray:
    values = coefficients[amp1].to_numpy() * np.conj(coefficients[amp2].to_numpy())
    return np.rad2deg(np.angle(values))


class TestPreprocessedMatrices:

    def test_no_preprocessing_warnings(self, results):
        assert all(not r.warnings for r in results.reports)

    def test_covariance_is_reduced_and_full_rank(self, results):
        for bin_id in results.fit["bin_id"]:
            matrix = results.covariance_matrix(bin_id)
            assert matrix.shape == (N_FREE_PARAMETERS, N_FREE_PARAMETERS)
            assert matrix.to_numpy().dtype == np.float64
            assert np.linalg.matrix_rank(matrix.to_numpy()) == N_FREE_PARAMETERS
            assert "m1mpP_im" not in matrix.index  # fixed anchor

    def test_covariance_matches_coefficient_errors(self, results):
        matrix = results.covariance_matrix(results.fit["bin_id"].iloc[0])
        assert set(results.coefficient_names) == {
            p[:-3] for p in matrix.index if p.endswith(("_re", "_im"))
        }

    def test_covariance_fill_fixed(self, results):
        bin_id = results.fit["bin_id"].iloc[0]
        with pytest.raises(KeyError):
            results.covariance_matrix(bin_id, ["m1mpP_re", "m1mpP_im"])
        matrix = results.covariance_matrix(
            bin_id, ["m1mpP_re", "m1mpP_im"], fill_fixed=True
        )
        assert matrix.loc["m1mpP_im"].eq(0).all()
        assert matrix.loc["m1mpP_re", "m1mpP_re"] > 0


class TestCoefficientsAndNormalization:

    def test_production_coefficients(self, results):
        coefficients = results.production_coefficients()
        assert list(coefficients.index) == results.fit["bin_id"].tolist()
        assert list(coefficients.columns) == results.coefficient_names
        assert coefficients.dtypes.eq(np.complex128).all()

    def test_production_coefficient_bin_order(self, results):
        bins = sorted(results.kinematic_bins, reverse=True)
        coefficients = results.production_coefficients(bins)
        assert list(coefficients.index) == [kb.bin_id for kb in bins]

    def test_stored_phases_are_coefficient_phases(self, results):
        coefficients = results.production_coefficients()
        np.testing.assert_allclose(
            phase(coefficients, "m1p0S", "m1pmS"),
            results.fit["m1p0S_m1pmS"].to_numpy(np.float64),
            atol=1e-3,
        )

    def test_norm_int_type_is_accepted(self, results):
        # the example norm_int.csv holds the accepted integrals
        assert results.norm_int_type == "accepted"
        np.testing.assert_allclose(
            results.reconstructed_intensity().to_numpy(),
            results.fit["intensity"].to_numpy(np.float64),
            rtol=1e-5,
        )

    def test_normalization_factors(self, results):
        factors = results.normalization_factors()
        assert list(factors.columns) == results.coefficient_names
        assert (factors.to_numpy() > 0).all()

        # with accepted integrals, the diagonal terms sum to the accepted intensity
        # up to the (small) interference between amplitudes in the acceptance
        coefficients = results.production_coefficients()
        diagonal_sum = (np.abs(coefficients) ** 2 * factors).sum(axis=1).to_numpy()
        np.testing.assert_allclose(
            diagonal_sum, results.fit["intensity"].to_numpy(np.float64), rtol=0.2
        )


class TestAlignConjugation:

    def test_reference_sign_is_enforced(self, aligned):
        coefficients = aligned.production_coefficients()
        for amp1, amp2, sign in REFERENCES:
            assert (sign * phase(coefficients, amp1, amp2) > 0).all()
        assert (aligned.fit["m1p0S_m1pmS"] < 0).all()
        assert aligned.conjugation_references == tuple(REFERENCES)

    def test_intensities_and_magnitudes_unchanged(self, results, aligned):
        amplitudes = list(results.amplitudes)
        pd.testing.assert_frame_equal(results.fit[amplitudes], aligned.fit[amplitudes])
        np.testing.assert_allclose(
            np.abs(results.production_coefficients()),
            np.abs(aligned.production_coefficients()),
        )

    def test_conjugation_is_per_sector(self, results, aligned):
        before = results.production_coefficients()
        after = aligned.production_coefficients()
        for bin_id in before.index:
            for sector in ("m", "p"):
                names = [n for n in results.coefficient_names if n.startswith(sector)]
                b, a = before.loc[bin_id, names], after.loc[bin_id, names]
                assert np.allclose(a, b) or np.allclose(a, np.conj(b))

    def test_covariance_transforms_with_jacobian(self, results, aligned):
        before = results.production_coefficients()
        after = aligned.production_coefficients()
        for bin_id in before.index:
            c0 = results.covariance_matrix(bin_id)
            c1 = aligned.covariance_matrix(bin_id)
            conjugated = {
                n
                for n in results.coefficient_names
                if not np.isclose(before.loc[bin_id, n], after.loc[bin_id, n])
            }
            s = np.array(
                [
                    -1.0 if p.endswith("_im") and p[:-3] in conjugated else 1.0
                    for p in c0.index
                ]
            )
            np.testing.assert_allclose(np.outer(s, s) * c0.to_numpy(), c1.to_numpy())

    def test_round_trip(self, aligned):
        flipped = [(a1, a2, -s) for a1, a2, s in REFERENCES]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            back = aligned.align_conjugation(flipped).align_conjugation(REFERENCES)
        pd.testing.assert_frame_equal(back.fit, aligned.fit)
        pd.testing.assert_frame_equal(back.covariance, aligned.covariance)

    @pytest.mark.parametrize(
        "references",
        [
            ("m1p0S", "p1pmS", 1),  # different sectors
            ("m1p0S", "m1pmS", 2),  # bad sign
            ("m1p0S", "nope", 1),  # unknown amplitude
            [("m1p0S", "m1pmS", 1), ("m1p0S", "m1ppS", 1)],  # same sector twice
        ],
    )
    def test_invalid_references(self, results, references):
        with pytest.raises(ValueError):
            results.align_conjugation(references)

    def test_ignores_coefficients_without_sector(self, results):
        """An 'isotropic' background has no reflectivity and is left untouched."""
        fit = results.fit.copy()
        fit["isotropic_re"] = 1.5
        fit["isotropic_im"] = -0.5
        with_background = dataclasses.replace(results, fit=fit)
        assert "isotropic" in with_background.coefficient_names

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            aligned = with_background.align_conjugation(REFERENCES)
        assert (aligned.fit["isotropic_im"] == -0.5).all()
        with pytest.raises(ValueError, match="reflectivity"):
            with_background.align_conjugation(("isotropic", "m1pmS", 1))

    def test_save_load_keeps_references(self, aligned, tmp_path):
        path = tmp_path / "results.pkl"
        aligned.save(path)
        assert Results.load(path).conjugation_references == tuple(REFERENCES)
