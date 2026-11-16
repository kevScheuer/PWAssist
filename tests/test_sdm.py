import dataclasses
import pathlib
import warnings

import numpy as np
import pytest

from pwassist.core.result import Results
from pwassist.core.sdm import (
    SDMBlock,
    SpinDensityMatrix,
    element_jacobian,
    element_labels,
    from_elements,
    to_elements,
)
from pwassist.io.binning import BinCollection
from pwassist.io.catalog import Catalog
from pwassist.preprocessing.preprocessor import Preprocessor

EXAMPLE_DATA = pathlib.Path(__file__).resolve().parents[1] / "examples" / "data"
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
        return Results.from_processed_bins(
            processed, final_state_parity=1
        ).align_conjugation(REFERENCES)


@pytest.fixture(scope="module")
def sdms(results) -> dict[str, SpinDensityMatrix]:
    return results.spin_density_matrices()


def random_amplitudes(rng, n_ranks, n):
    return rng.normal(size=(n_ranks, n)) + 1j * rng.normal(size=(n_ranks, n))


class TestElements:

    def test_labels(self):
        assert element_labels(["a", "b", "c"]) == (
            "Re[a,a]",
            "Re[b,b]",
            "Re[c,c]",
            "Re[a,b]",
            "Im[a,b]",
            "Re[a,c]",
            "Im[a,c]",
            "Re[b,c]",
            "Im[b,c]",
        )

    def test_round_trip(self):
        t = random_amplitudes(np.random.default_rng(1), 2, 4)
        rho = np.einsum("ra,rc->ac", t, t.conj())
        elements = to_elements(rho)
        assert elements.shape == (16,)
        np.testing.assert_allclose(from_elements(elements), rho)
        np.testing.assert_allclose(elements[:4], np.real(np.diag(rho)))
        np.testing.assert_allclose(elements[4:6], [rho[0, 1].real, rho[0, 1].imag])

    def test_from_elements_requires_square(self):
        with pytest.raises(ValueError, match="square"):
            from_elements(np.zeros(5))

    @pytest.mark.parametrize("n_ranks", [1, 2])
    def test_jacobian_matches_finite_differences(self, n_ranks):
        rng = np.random.default_rng(2)
        t = random_amplitudes(rng, n_ranks, 3)

        def elements(amplitudes):
            return to_elements(np.einsum("ra,rc->ac", amplitudes, amplitudes.conj()))

        d_re, d_im = element_jacobian(t)
        step = 1e-6
        for k in range(t.size):
            for part, analytic in ((1.0, d_re), (1j, d_im)):
                shifted = t.copy().ravel()
                shifted[k] += part * step
                numeric = (elements(shifted.reshape(t.shape)) - elements(t)) / step
                np.testing.assert_allclose(analytic[:, k], numeric, atol=1e-5)


class TestBlock:

    def test_rank_one(self):
        block = SDMBlock.rank_one("p", ["a", "b"])
        assert block.rank == 1 and block.n_waves == 2
        assert block.coefficients == ("a", "b")
        assert len(block.labels) == 4

    def test_invalid_layout(self):
        with pytest.raises(ValueError, match="one coefficient"):
            SDMBlock("p", ("a", "b"), (("a",),))
        with pytest.raises(ValueError, match="at least one rank"):
            SDMBlock("p", ("a",), ())


class TestResultsSDM:

    def test_default_blocks(self, results):
        blocks = results.sdm_blocks()
        assert set(blocks) == {"m", "p"}
        assert all(b.rank == 1 for b in blocks.values())
        assert set(blocks["m"].waves) == {
            n for n in results.coefficient_names if n.startswith("m")
        }

    def test_rank_one_properties(self, sdms):
        for sdm in sdms.values():
            rho = sdm.matrices
            np.testing.assert_allclose(rho, np.conj(np.swapaxes(rho, 1, 2)))
            diagonal = np.real(np.diagonal(rho, axis1=1, axis2=2))
            # full coherence: |rho_ac|^2 = rho_aa rho_cc
            np.testing.assert_allclose(
                np.abs(rho) ** 2, diagonal[:, :, None] * diagonal[:, None, :]
            )

    def test_diagonal_is_normalized_intensity(self, results, sdms):
        coefficients = results.production_coefficients()
        norm = results.normalization_factors()
        sdm = sdms["m"]
        for wave in sdm.waves:
            expected = np.abs(coefficients[wave]) ** 2 * norm[wave]
            np.testing.assert_allclose(sdm.intensity(wave), expected.to_numpy())

    def test_phase_matches_stored(self, results, sdms):
        sdm = sdms["m"]
        stored = results.fit.set_index("bin_id").loc[list(sdm.bin_ids), "m1p0S_m1pmS"]
        np.testing.assert_allclose(
            sdm.phase("m1p0S", "m1pmS"), stored.to_numpy(np.float64), atol=1e-3
        )

    def test_covariance_rank(self, sdms):
        for sdm in sdms.values():
            n = sdm.block.n_waves
            assert sdm.covariance.shape == (len(sdm.bin_ids), n * n, n * n)
            for c in sdm.covariance:
                scale = np.sqrt(np.outer(np.diag(c), np.diag(c)))
                eigenvalues = np.linalg.eigvalsh(c / scale)
                # a rank-1 SDM of n waves has 2n - 1 real degrees of freedom
                assert np.sum(eigenvalues > 1e-9 * eigenvalues.max()) == 2 * n - 1

    def test_covariance_matches_finite_differences(self, results, sdms):
        """J C J^T with J from differentiating the SDM in the fit parameters."""
        sdm = sdms["p"]
        bin_id = sdm.bin_ids[0]
        fit_covariance = results.covariance_matrix(bin_id)
        coefficients = sdm.block.coefficients
        parameters = [
            p
            for p in fit_covariance.index
            if p.startswith("parScale_") or p[:-3] in coefficients
        ]
        row = results.fit["bin_id"] == bin_id

        def elements(fit):
            shifted = dataclasses.replace(results, fit=fit)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                values = shifted.spin_density_matrices(
                    [kb for kb in results.kinematic_bins if kb.bin_id == bin_id],
                    blocks=[sdm.block],
                    covariance=False,
                )
            return values["p"].elements[0]

        base_fit = results.fit.copy()
        for p in parameters:
            base_fit[p] = base_fit[p].astype(np.float64)
        jacobian = np.empty((len(sdm.labels), len(parameters)))
        for k, p in enumerate(parameters):
            step = 1e-6 * max(1.0, abs(base_fit.loc[row, p].iloc[0]))
            plus, minus = base_fit.copy(), base_fit.copy()
            plus.loc[row, p] += step
            minus.loc[row, p] -= step
            jacobian[:, k] = (elements(plus) - elements(minus)) / (2 * step)
        c = fit_covariance.loc[parameters, parameters].to_numpy()
        expected = jacobian @ c @ jacobian.T
        np.testing.assert_allclose(
            sdm.covariance[0], expected, rtol=1e-4, atol=1e-8 * np.abs(expected).max()
        )

    def test_scales_change_covariance(self, results, sdms):
        without = results.spin_density_matrices(propagate_scales=False)
        assert not np.allclose(without["p"].covariance, sdms["p"].covariance)

    def test_select_is_sub_block(self, sdms):
        sdm = sdms["m"]
        waves = list(sdm.waves[2:0:-1])
        sub = sdm.select(waves)
        assert sub.waves == tuple(waves)
        i, j = sdm.waves.index(waves[0]), sdm.waves.index(waves[1])
        np.testing.assert_allclose(sub.matrices[:, 0, 1], sdm.matrices[:, i, j])
        label = f"Re[{waves[0]},{waves[0]}]"
        np.testing.assert_allclose(
            sub.errors[:, sub.labels.index(label)],
            sdm.errors[:, sdm.labels.index(label)],
        )
        with pytest.raises(KeyError):
            sdm.select(["nope"])

    def test_to_frame(self, sdms):
        frame = sdms["p"].to_frame()
        assert frame.columns[0] == "bin_id"
        assert "Re[p1mpP,p1mpP]_err" in frame.columns
        assert len(frame) == len(sdms["p"].bin_ids)

    def test_rank_two_block_is_sum_of_ranks(self, results, sdms):
        m, p = results.sdm_blocks()["m"], results.sdm_blocks()["p"]
        block = SDMBlock(
            "both", ("x", "y"), ((m.waves[0], m.waves[1]), (p.waves[0], None))
        )
        sdm = results.spin_density_matrices(blocks=[block])["both"]
        rho_m = sdms["m"].select(list(m.waves[:2])).matrices
        rho_p = sdms["p"].matrices[:, 0, 0]
        expected = rho_m.copy()
        expected[:, 0, 0] += rho_p
        np.testing.assert_allclose(sdm.matrices, expected)
        assert sdm.covariance.shape == (len(sdm.bin_ids), 4, 4)

    def test_sample_frames(self, results):
        if results.bootstrap is None and results.randomized is None:
            pytest.skip("example results have no sample fits")
        frame = "bootstrap" if results.bootstrap is not None else "randomized"
        samples = results.spin_density_matrices(frame=frame, covariance=False)
        assert samples["m"].covariance is None
        assert len(samples["m"].bin_ids) == len(getattr(results, frame))
        with pytest.raises(ValueError, match="fit"):
            results.spin_density_matrices(frame=frame)

    def test_warns_without_conjugation_choice(self, results):
        unaligned = dataclasses.replace(results, conjugation_references=())
        with pytest.warns(UserWarning, match="conjugation"):
            unaligned.spin_density_matrices(covariance=False)

    def test_unknown_coefficient(self, results):
        with pytest.raises(KeyError, match="unknown"):
            results.spin_density_matrices(blocks=[SDMBlock.rank_one("x", ["nope"])])
