"""Spin-density matrices (SDMs) of the partial waves in each kinematic bin.

Formalism
---------
This follows the conventions of the COMPASS mass-dependent analyses, see

- F. Haas, PhD thesis, Technische Universitaet Muenchen (2014), the chapters on the
  mass-independent fit model and on the spin-density matrix used as input to the
  mass-dependent fit,
- S. Wallner, PhD thesis, Technische Universitaet Muenchen, the chapters on the
  partial-wave decomposition, the spin-density matrix, and its use (with uncertainties)
  in the resonance-model fit,
- COMPASS, Phys. Rev. D 98, 092003 (2018), Secs. II and IV.

In a kinematic bin, the intensity distribution is an incoherent sum over 'blocks' b
(here: the reflectivities, which do not interfere) and 'ranks' r (incoherent
configurations within a block, e.g. recoil helicity configurations), each a coherent sum
over the partial waves a,

    I(tau) = sum_b sum_r | sum_a T^{b,r}_a psi_a(tau) |^2,

with transition amplitudes T and decay amplitudes psi. All observable information in
the bin is then contained in the spin-density matrix of each block,

    rho^b_{ac} = sum_r T^{b,r}_a (T^{b,r}_c)^*,

which is hermitian, positive semi-definite, and has rank <= the number of ranks R.
With the decay amplitudes normalized to their (acceptance corrected) integrals, as done
here through T_a = V_a sqrt(N_a) (see 'Results.normalization_factors'),

- rho_aa is the acceptance corrected number of events in wave a (the intensity),
- arg rho_ac is the phase difference of waves a and c,
- |rho_ac|^2 <= rho_aa rho_cc, with equality ("full coherence") for rank 1.

Why the SDM: rho is invariant under any unitary mixing T^b -> U T^b of the ranks,
including the global phase of a block. It therefore does not depend on the choices that
make the production coefficients unique in the PWA fit (which wave is fixed real, its
sign, the order of the ranks), and rank > 1 needs no special treatment. The
mass-dependent fit compares a model SDM to these matrices.

What the SDM does NOT remove: for two pseudoscalars produced with a linearly polarized
beam, the Zlm decay functions within a reflectivity sum are real, so complex conjugation
T -> T^* of a whole block leaves the intensity unchanged, i.e. rho^b and (rho^b)^* fit
the data equally well. Resolve this before building the SDMs with
'Results.align_conjugation'.

Element vector
--------------
A block with n waves has n^2 real SDM elements. They are ordered as

    [rho_00, rho_11, ..., rho_(n-1)(n-1),
     Re rho_01, Im rho_01, Re rho_02, Im rho_02, ..., Re rho_(n-2)(n-1), Im ...],

and labelled 'Re[a,a]' (intensities), 'Re[a,c]' and 'Im[a,c]' (a before c).
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------------
# Layout
# ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SDMBlock:
    """Which production coefficients make up one block of the spin-density matrix.

    Attributes:
        name (str): Block label, e.g. the reflectivity 'p' or '<prefix>::p'.
        waves (tuple[str, ...]): Labels of the block's waves (rows/columns of rho).
        ranks (tuple[tuple[str | None, ...], ...]): For each rank r, the production
            coefficient that gives T^r_a for every wave a, or None if wave a does not
            appear in rank r (its T^r_a is then 0). A rank-1 block has one entry.
    """

    name: str
    waves: tuple[str, ...]
    ranks: tuple[tuple[str | None, ...], ...]

    def __post_init__(self) -> None:
        if not self.ranks:
            raise ValueError(f"Block '{self.name}' needs at least one rank.")
        for coefficients in self.ranks:
            if len(coefficients) != len(self.waves):
                raise ValueError(
                    f"Each rank of block '{self.name}' must list one coefficient (or"
                    f" None) per wave {self.waves}, got {coefficients}."
                )

    @classmethod
    def rank_one(cls, name: str, coefficients: Sequence[str]) -> "SDMBlock":
        """A rank-1 block whose waves are the coefficients themselves."""
        waves = tuple(coefficients)
        return cls(name=name, waves=waves, ranks=(waves,))

    @property
    def n_waves(self) -> int:
        return len(self.waves)

    @property
    def rank(self) -> int:
        return len(self.ranks)

    @property
    def coefficients(self) -> tuple[str, ...]:
        """All production coefficients used by the block, without duplicates."""
        names = [c for rank in self.ranks for c in rank if c is not None]
        return tuple(dict.fromkeys(names))

    @property
    def labels(self) -> tuple[str, ...]:
        """Labels of the real SDM elements, in element-vector order."""
        return element_labels(self.waves)


# ------------------------------------------------------------------------------------
# Element vectors
# ------------------------------------------------------------------------------------


def element_labels(waves: Sequence[str]) -> tuple[str, ...]:
    """Labels of the n^2 real SDM elements of 'waves', in element-vector order."""
    labels = [f"Re[{a},{a}]" for a in waves]
    for i, j in _upper_pairs(len(waves)):
        labels += [f"Re[{waves[i]},{waves[j]}]", f"Im[{waves[i]},{waves[j]}]"]
    return tuple(labels)


def to_elements(rho: np.ndarray) -> np.ndarray:
    """Real element vectors of hermitian matrices.

    Args:
        rho (np.ndarray): Complex matrices, shape (..., n, n).

    Returns:
        np.ndarray: Elements, shape (..., n^2), see the module docstring for the order.
    """
    n = rho.shape[-1]
    i, j = np.array(_upper_pairs(n), dtype=np.intp).reshape(-1, 2).T
    upper = rho[..., i, j]
    off_diagonal = np.stack([np.real(upper), np.imag(upper)], axis=-1)
    off_diagonal = off_diagonal.reshape(rho.shape[:-2] + (-1,))
    diagonal = np.real(np.diagonal(rho, axis1=-2, axis2=-1))
    return np.concatenate([diagonal, off_diagonal], axis=-1)


def from_elements(elements: np.ndarray) -> np.ndarray:
    """Hermitian matrices from real element vectors (inverse of 'to_elements')."""
    n = int(round(np.sqrt(elements.shape[-1])))
    if n * n != elements.shape[-1]:
        raise ValueError(f"{elements.shape[-1]} elements do not form a square matrix.")
    rho = np.zeros(elements.shape[:-1] + (n, n), dtype=np.complex128)
    rho[..., np.arange(n), np.arange(n)] = elements[..., :n]
    for k, (i, j) in enumerate(_upper_pairs(n)):
        value = elements[..., n + 2 * k] + 1j * elements[..., n + 2 * k + 1]
        rho[..., i, j] = value
        rho[..., j, i] = np.conj(value)
    return rho


def element_jacobian(amplitudes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Derivatives of the SDM elements with respect to the transition amplitudes.

    With rho_ac = sum_r T^r_a (T^r_c)^*, the derivatives with respect to the real and
    imaginary parts of one amplitude T^r_k are

        d rho_ac / d Re T^r_k = delta_ak (T^r_c)^* + delta_ck T^r_a,
        d rho_ac / d Im T^r_k = i delta_ak (T^r_c)^* - i delta_ck T^r_a,

    of which each element takes the real (diagonal, Re) or imaginary (Im) part.

    Args:
        amplitudes (np.ndarray): Complex amplitudes T^r_a, shape (R, n).

    Returns:
        tuple[np.ndarray, np.ndarray]: d elements / d Re T and d elements / d Im T,
            each of shape (n^2, R * n), with the amplitudes flattened rank by rank.
    """
    n_ranks, n = amplitudes.shape
    d_re = np.zeros((n, n, n_ranks, n), dtype=np.complex128)
    d_im = np.zeros((n, n, n_ranks, n), dtype=np.complex128)
    for r in range(n_ranks):
        t = amplitudes[r]
        for a in range(n):
            for c in range(n):
                d_re[a, c, r, a] += np.conj(t[c])
                d_re[a, c, r, c] += t[a]
                d_im[a, c, r, a] += 1j * np.conj(t[c])
                d_im[a, c, r, c] -= 1j * t[a]

    def flatten(d: np.ndarray) -> np.ndarray:
        # the elements are linear in rho, so their derivatives follow 'to_elements'
        return to_elements(np.moveaxis(d.reshape(n, n, -1), -1, 0)).T

    return flatten(d_re), flatten(d_im)


def _upper_pairs(n: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


# ------------------------------------------------------------------------------------
# Container
# ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SpinDensityMatrix:
    """Spin-density matrices of one block, with element covariances.

    Attributes:
        block (SDMBlock): The block layout.
        bin_ids (tuple[str, ...]): Kinematic bin of each row. Rows from randomized or
            bootstrap fits repeat their bin.
        matrices (np.ndarray): Complex matrices rho, shape (n_rows, n, n).
        covariance (np.ndarray | None): Covariance of the element vectors, propagated
            from the PWA fit covariance, shape (n_rows, n^2, n^2). For rank R < n it
            is singular: rho has only 2 R n - R^2 free real parameters.
    """

    block: SDMBlock
    bin_ids: tuple[str, ...]
    matrices: np.ndarray
    covariance: np.ndarray | None = None

    @property
    def waves(self) -> tuple[str, ...]:
        return self.block.waves

    @property
    def labels(self) -> tuple[str, ...]:
        return self.block.labels

    @property
    def elements(self) -> np.ndarray:
        """Real element vectors, shape (n_rows, n^2)."""
        return to_elements(self.matrices)

    @property
    def errors(self) -> np.ndarray:
        """Standard deviations of the elements, shape (n_rows, n^2)."""
        if self.covariance is None:
            raise ValueError("No covariance was computed for these matrices.")
        return np.sqrt(np.clip(np.diagonal(self.covariance, axis1=1, axis2=2), 0, None))

    def intensity(self, wave: str) -> np.ndarray:
        """rho_aa of 'wave', shape (n_rows,)."""
        i = self.waves.index(wave)
        return np.real(self.matrices[:, i, i])

    def phase(self, wave1: str, wave2: str) -> np.ndarray:
        """Phase difference arg rho_{wave1,wave2} in degrees, shape (n_rows,)."""
        i, j = self.waves.index(wave1), self.waves.index(wave2)
        return np.rad2deg(np.angle(self.matrices[:, i, j]))

    def to_frame(self) -> pd.DataFrame:
        """Elements (and '<label>_err' errors, if available) with a 'bin_id' column."""
        frame = pd.DataFrame(self.elements, columns=list(self.labels))
        if self.covariance is not None:
            errors = pd.DataFrame(
                self.errors, columns=[f"{label}_err" for label in self.labels]
            )
            frame = pd.concat([frame, errors], axis=1)
        frame.insert(0, "bin_id", list(self.bin_ids))
        return frame

    def select(self, waves: Sequence[str]) -> "SpinDensityMatrix":
        """The sub-matrix of some waves, with the matching covariance block.

        Dropping waves from an SDM is exact: the remaining elements and their
        covariance are unchanged (no reference wave is needed).
        """
        unknown = [w for w in waves if w not in self.waves]
        if unknown:
            raise KeyError(f"{unknown} are not waves of block '{self.block.name}'.")
        index = [self.waves.index(w) for w in waves]
        block = SDMBlock(
            name=self.block.name,
            waves=tuple(waves),
            ranks=tuple(tuple(rank[i] for i in index) for rank in self.block.ranks),
        )
        # the new elements are a signed selection of the old ones: if the order of
        # two waves is swapped, Re[a,c] = Re[c,a] and Im[a,c] = -Im[c,a]
        position = {label: k for k, label in enumerate(self.labels)}
        selection = np.zeros((len(block.labels), len(self.labels)))
        for k, label in enumerate(block.labels):
            if label in position:
                selection[k, position[label]] = 1.0
            else:
                kind, pair = label[:2], label[3:-1]
                a, c = pair.split(",")
                swapped = f"{kind}[{c},{a}]"
                selection[k, position[swapped]] = 1.0 if kind == "Re" else -1.0
        covariance = (
            None
            if self.covariance is None
            else np.einsum("ij,bjk,lk->bil", selection, self.covariance, selection)
        )
        return SpinDensityMatrix(
            block=block,
            bin_ids=self.bin_ids,
            matrices=self.matrices[:, index][:, :, index],
            covariance=covariance,
        )
