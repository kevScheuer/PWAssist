import itertools
import warnings
from dataclasses import dataclass, replace
from enum import Enum
from typing import Callable


class NamingScheme(Enum):
    AUTO = "auto"
    JLME = "JLme"  # e.g. 1P-1p
    EJPML = "eJPmL"  # e.g. p1p0S
    LME = "Lme"  # e.g. S0+


@dataclass(frozen=True, slots=True)
class ParsedAmplitude:
    """Quantum numbers extracted from the amplitude name"""

    amp_name: str
    e: str = ""
    J: str = ""
    P: str = ""
    m: str = ""
    L: str = ""

    def get(self, quantum_number: str) -> str:
        """Get the value of a quantum number by name"""
        return getattr(self, quantum_number)


@dataclass(frozen=True)
class SchemeDef:
    infer: Callable[[str], bool]  # Determine if an amplitude matches this naming scheme
    parse: Callable[[str], ParsedAmplitude]  # index-based parser for the amplitude name
    sum_groups: tuple[tuple[str, ...], ...]  # possible quantum number groupings
    single_amplitudes: tuple[str, ...]  # base individual amplitude quantum numbers
    example: str


# --------------------------------------------------------------------------------------
# Scheme Implementations
# --------------------------------------------------------------------------------------
# NOTE: The following functions are used to infer the naming scheme of an amplitude
# based on its name. If you wish to add a new naming scheme, be very careful that its
# inference function does not overlap with any of the existing ones.


def _infer_jlme(amp: str) -> bool:
    # Examples: 1P+0p, 2D-1n, 3F+2p
    return amp[0].isdigit()


def _parse_jlme(amp: str) -> ParsedAmplitude:
    return ParsedAmplitude(amp_name=amp, J=amp[0], L=amp[1], m=amp[2:-1], e=amp[-1])


def _infer_lme(amp: str) -> bool:
    # Examples: S0+, P+1-, D-2+, F+3-
    return amp[0].isupper() and amp[-1] in ["+", "-"]


def _parse_lme(amp: str) -> ParsedAmplitude:
    return ParsedAmplitude(
        amp_name=amp,
        L=amp[0],
        J=amp[0],  # this scheme is for two-ps, where L=J
        m=amp[1:-1],
        e=amp[2],
    )


def _infer_ejpml(amp: str) -> bool:
    # Examples: p1p0S, m1mmP, p3mqF
    return amp[0] in ["p", "m"] and amp[-1].isupper()


def _parse_ejpml(amp: str) -> ParsedAmplitude:
    return ParsedAmplitude(
        amp_name=amp, e=amp[0], J=amp[1], P=amp[2], m=amp[2:-1], L=amp[-1]
    )


# The idea behind the coherent sum groupings is that any quantum number excluded has
# been summed over in the amplitude. For example, if the sum group is ("J", "L"),
# then the amplitude has been summed over all possible values of "e" and "m".
SCHEMES: dict[NamingScheme, SchemeDef] = {
    NamingScheme.JLME: SchemeDef(
        infer=_infer_jlme,
        parse=_parse_jlme,
        example="1P+0p",
        sum_groups=(
            ("e",),
            ("J",),
            ("J", "e"),
            ("J", "L"),
            ("J", "L", "e"),
            ("J", "L", "m"),
        ),
        single_amplitudes=("J", "L", "m", "e"),
    ),
    NamingScheme.LME: SchemeDef(
        infer=_infer_lme,
        parse=_parse_lme,
        example="S0+",
        sum_groups=(
            ("e",),
            ("L",),
            ("L", "e"),
            ("L", "m"),
        ),
        single_amplitudes=("L", "m", "e"),
    ),
    NamingScheme.EJPML: SchemeDef(
        infer=_infer_ejpml,
        parse=_parse_ejpml,
        example="p1p0S",
        sum_groups=(
            ("e",),
            ("J", "P"),
            ("e", "J", "P"),
            ("J", "P", "L"),
            ("e", "J", "P", "L"),
        ),
        single_amplitudes=("e", "J", "P", "m", "L"),
    ),
}

# --------------------------------------------------------------------------------------
# Orbital angular momentum letter <-> integer (standard spectroscopic notation, skips J)
# --------------------------------------------------------------------------------------
_L_LETTERS = "SPDFGHIKLM"
_L_TO_INT = {letter: i for i, letter in enumerate(_L_LETTERS)}


def _l_letter_to_int(letter: str) -> int:
    try:
        return _L_TO_INT[letter]
    except KeyError:
        raise ValueError(f"Unrecognized orbital angular momentum letter: '{letter}'")


# --------------------------------------------------------------------------------------
# Amplitude Parser
# --------------------------------------------------------------------------------------


class AmplitudeParser:
    """Parses amplitudes and groups according to naming scheme and quantum numbers.

    Can be initialized with a specific naming scheme, or allow the parser to attempt
    to infer the scheme from the amplitude name.
    """

    def __init__(
        self,
        scheme: str | NamingScheme = NamingScheme.AUTO,
        final_state_parity: int | None = None,
    ) -> None:
        if isinstance(scheme, str):
            try:
                self.requested_scheme = NamingScheme(scheme)
            except ValueError:
                self.requested_scheme = NamingScheme.AUTO
        else:
            self.requested_scheme = scheme
        if final_state_parity is not None and final_state_parity not in [-1, 1]:
            raise ValueError("final_state_parity must be either -1 or 1")
        self.final_state_parity = final_state_parity

    # ----------------------------------------------------------------------------------
    # Public Methods
    # ----------------------------------------------------------------------------------

    @staticmethod
    def infer_naming_scheme(label: str) -> NamingScheme:
        if not label:
            raise ValueError("Amplitude name cannot be empty")
        for scheme, scheme_def in SCHEMES.items():
            if scheme_def.infer(label):
                return scheme
        return NamingScheme.AUTO

    def parse_amplitude(self, label: str) -> ParsedAmplitude:
        """Parse an amplitude name into its quantum numbers

        Args:
            label (str): the amplitude name to parse

        Returns:
            ParsedAmplitude: a dataclass containing the parsed quantum numbers
        Raises:
            ValueError: if the naming scheme cannot be inferred or is invalid
        """
        scheme = self.requested_scheme
        if scheme == NamingScheme.AUTO:
            scheme = self.infer_naming_scheme(label)
        if scheme == NamingScheme.AUTO:
            raise ValueError(f"Could not infer naming scheme for amplitude: {label}")
        scheme_def = SCHEMES[scheme]
        parsed = scheme_def.parse(label)
        return self._apply_final_state_parity(parsed, scheme_def)

    def get_coherent_sums(self, columns: list[str]) -> dict[str, list[str]]:
        """Return a dictionary of coherent sum labels and coherent sums found

        The function will first attempt to find all possible coherent sums from the
        "base" amplitudes in the provided columns. Then, it returns only those coherent
        sums that are actually present in the columns.

        Example:
            If the columns contain the following labels:
                ["1P+0p", "1P+1p", "1S-1n", "1P", "p"]
            The function determines that the possible coherent sums are (based off the
            JLme naming scheme):
                "J" -> ["1"]
                "e" -> ["p", "n"]
                "Je" -> ["1p", "1n"]
                "JL" -> ["1P", "1S"]
                "JLe" -> ["1Pp", "1Sn"]
                "JLm" -> ["1P0", "1P1", "1S-1"]
            Then, it returns only those sums that are actually present in the columns:
                {"JL": ["1P"], "e": ["p"]}

        Args:
            columns: list of column names from a results dataframe.
        Returns:
            dict[str, list[str]]: a dictionary of coherent sum labels and coherent sums
                The keys are the sum labels (e.g. "JL", "e", etc.) and the values are
                lists of coherent sums found in the columns.
        """

        base_amplitudes = self.get_amplitudes(columns)

        # These are all possible sums that could be present in the columns, based on
        # the naming scheme from the "base" amplitudes
        expected_sum_groups = self._build_sum_groups(
            base_amplitudes, self.requested_scheme
        )

        # Of the possible sums, only keep those that are actually present in the
        # columns
        found_sums = {}
        for sum_label, sums in expected_sum_groups.items():
            found_sums[sum_label] = [s for s in sums if s in columns]

        return found_sums

    def get_amplitudes(self, columns: list[str]) -> list[str]:
        """Return a list of base amplitude labels found in the provided columns.

        Amplitudes are identified by those strings who always have a "_re" and "_im"
        part attached to them. Of course, other parameters may be written like this,
        but should most likely not get confused with an amplitude naming scheme. The
        reason we have to filter our amplitudes is because our columns also have
        coherent sums, and these formats can be easily confused for a different
        naming scheme.

        Args:
            columns: list of column names from a results dataframe.
        Returns:
            list[str]: a list of base amplitude labels found in the provided columns.
        """
        amplitudes = [c[:-3] for c in columns if c.endswith("_re") or c.endswith("_im")]

        return self._filter_by_scheme(amplitudes, self.requested_scheme)

    def get_phase_differences(self, columns: list[str]) -> list[str]:
        """Return a list of phase difference labels found in the provided columns.

        Args:
            columns: list of column names from a results dataframe.
        Returns:
            list[str]: a list of phase difference labels found in the provided columns.
        """
        base_amplitudes = self.get_amplitudes(columns)
        all_possible_pairs = list(itertools.combinations(base_amplitudes, 2))
        all_possible_phase_diffs = [f"{a1}_{a2}" for a1, a2 in all_possible_pairs] + [
            f"{a2}_{a1}" for a1, a2 in all_possible_pairs
        ]
        return [p for p in all_possible_phase_diffs if p in columns]

    def to_latex(self, label: str) -> str:
        """LaTeX string for a given amplitude or phase difference.

        For coherent sums, use sum_to_latex, as a bare coherent sum string does not
        carry enough info to identify its scheme.

        Args:
            label (str): the amplitude or phase difference label to convert to LaTeX
        Returns:
            str: a $J^P L_m^{(e)}$ style LaTeX string for the amplitude or phase
                difference
        """
        if "_" in label:
            # phase difference
            a1, a2 = label.split("_", 1)
            return f"${self._amp_latex(a1)} - {self._amp_latex(a2)}$"
        return f"${self._amp_latex(label)}$"

    def sum_to_latex(self, group_key: str, sum_string: str) -> str:
        """LaTeX string for a given coherent sum.

        Args:
            group_key (str): key from get_coherent_sums() output e.g. 'JL', 'e', etc.
                Pass it alongside the sum_string to identify the naming scheme and
                quantum numbers.
            sum_string (str): the coherent sum string to convert to LaTeX

        """
        scheme, group = self._find_group(group_key)
        scheme_def = SCHEMES[scheme]
        raw = self._parse_group_from_string(sum_string, group)
        parsed = self._apply_final_state_parity(
            ParsedAmplitude(amp_name=sum_string, **raw), scheme_def
        )
        return f"${self._render(self._values_from_parsed(parsed))}$"

    # ----------------------------------------------------------------------------------
    # Private Methods
    # ----------------------------------------------------------------------------------

    def _apply_final_state_parity(
        self, parsed: ParsedAmplitude, scheme_def: SchemeDef
    ) -> ParsedAmplitude:
        """Fill in P= final_state_parity * (-1)^L for non-parity explicit schemes

        Schemes like JLme and Lme do not explicitly contain the parity quantum
        number. For these schemes, we can calculate the parity from the final state
        parity and the orbital angular momentum L.

        Args:
            parsed (ParsedAmplitude): The parsed amplitude object
            scheme_def (SchemeDef): The scheme definition for the current naming scheme

        Returns:
            ParsedAmplitude: A new ParsedAmplitude object with the P quantum number
                filled in if applicable.
        """
        if "P" in scheme_def.single_amplitudes:
            return parsed  # P is already explicitly defined in the amplitude name
        if self.final_state_parity is None or not parsed.L:
            return parsed  # No final state parity provided, cannot calculate P
        L = _l_letter_to_int(parsed.L)
        total_parity = self.final_state_parity * (-1) ** L
        return replace(parsed, P="p" if total_parity == 1 else "m")

    def _filter_by_scheme(self, labels: list[str], scheme: NamingScheme) -> list[str]:
        """Returns labels that match a particular naming scheme

        Note that if NamingScheme is set to "Auto", the function will attempt to infer
        a common scheme from the provided labels. If multiple schemes are found,
        a ValueError will be raised.

        Args:
            labels (list[str]): strings to be filtered by naming scheme
            scheme (NamingScheme): an amplitude naming scheme based on quantum numbers
                to filter on.

        Returns:
            list[str]: a list of labels that match the specified naming scheme

        Raises:
            ValueError: if no labels are provided, an invalid naming scheme is
                specified, or if multiple schemes are found when using AUTO.
        """
        if not labels:
            raise ValueError("No labels provided to filter_by_scheme")
        filtered = []

        # try to infer a common naming scheme from the given labels if scheme is AUTO
        if scheme == NamingScheme.AUTO:
            scheme = self._find_common_scheme(labels)

        # filter the labels by the specified scheme
        for label in labels:
            if self.infer_naming_scheme(label) == scheme:
                filtered.append(label)

        return filtered

    def _find_common_scheme(self, labels: list[str]) -> NamingScheme:
        """Find a common naming scheme from a list of labels.

        Args:
            labels (list[str]): strings to be checked for a common naming scheme
        Returns:
            NamingScheme: the common naming scheme found in the labels
        Raises:
            ValueError: if multiple schemes are found
        """
        found_schemes = {self.infer_naming_scheme(l) for l in labels}

        # non-amplitude labels will still be marked "AUTO", so filter those out
        found_schemes.discard(NamingScheme.AUTO)
        if not found_schemes:
            raise ValueError("No valid amplitude labels found")

        if len(found_schemes) > 1:
            raise ValueError(
                f"Multiple naming schemes found: {found_schemes}."
                "Please ensure all amplitudes use the same naming scheme, or"
                " specify a scheme explicitly."
            )

        return found_schemes.pop()

    def _build_sum_groups(
        self, amps: list[str], scheme: NamingScheme
    ) -> dict[str, list[str]]:

        if not amps:
            warnings.warn("No amplitudes provided to build_sum_groups")
            return {}

        # ensure that we have a common scheme to work with across all labels
        if scheme == NamingScheme.AUTO:
            scheme = self._find_common_scheme(amps)

        # get the base amplitude names that match the specified scheme
        filtered_labels = self._filter_by_scheme(amps, scheme)

        scheme_def = SCHEMES[scheme]
        groups: dict[str, set[str]] = {}  # e.g. "JL" -> {"1P", "2D", "3F"}

        for label in filtered_labels:
            # parse amplitude name into its quantum numbers
            parsed = scheme_def.parse(label)
            parsed = self._apply_final_state_parity(parsed, scheme_def)
            for group in scheme_def.sum_groups:
                sum_label = "".join(group)
                amp_sum = "".join(parsed.get(qn) for qn in group)
                if amp_sum:
                    groups.setdefault(sum_label, set()).add(amp_sum)

        return {sum_label: sorted(list(sums)) for sum_label, sums in groups.items()}

    def _render(self, values: dict[str, str]) -> str:
        """Bare LaTeX (no $$) string for a given set of quantum numbers."""

        _CHAR_TO_SIGN = {"p": 1, "m": -1}

        if set(values) == {"e"}:
            return rf"\varepsilon = {_CHAR_TO_SIGN.get(values['e'], values['e'])}"

        latex = ""

        if "J" in values:
            latex += values["J"]
        if "P" in values:
            latex += f"^{{{_CHAR_TO_SIGN.get(values['P'], values['P'])}}}"

        if "L" in values:
            latex += f" {values['L']}" if latex else values["L"]
            if "m" in values:
                latex += f"_{{{values['m']}}}"
            if "e" in values:
                latex += f"^{{({_CHAR_TO_SIGN.get(values['e'], values['e'])})}}"
        elif "e" in values:
            latex += rf" \varepsilon={_CHAR_TO_SIGN.get(values['e'], values['e'])}"

        return latex

    def _parse_group_from_string(
        self, sum_string: str, group: tuple[str, ...]
    ) -> dict[str, str]:

        values: dict[str, str] = {}
        pos = 0

        for i, qn in enumerate(group):
            if i == len(group) - 1:
                values[qn] = sum_string[pos:]
            else:
                values[qn] = sum_string[pos : pos + 1]
                pos += 1

        return values

    def _find_group(self, group_key: str) -> tuple[NamingScheme, tuple[str, ...]]:
        """Resolve a get_coherent_sums() dict key back to its scheme + group tuple."""
        schemes_to_check = (
            [self.requested_scheme]
            if self.requested_scheme != NamingScheme.AUTO
            else list(SCHEMES)
        )
        for scheme in schemes_to_check:
            for group in SCHEMES[scheme].sum_groups:
                if "".join(group) == group_key:
                    # "e" matches every scheme identically; first hit is fine
                    return scheme, group
        raise ValueError(f"Unrecognized coherent-sum group key: '{group_key}'")

    def _values_from_parsed(self, parsed: ParsedAmplitude) -> dict[str, str]:
        return {
            qn: getattr(parsed, qn)
            for qn in ("J", "P", "L", "m", "e")
            if getattr(parsed, qn)
        }

    def _amp_latex(self, amp: str) -> str:
        """Convert amp label to latex string. Infer scheme and parity if necessary."""
        scheme = self.requested_scheme
        if scheme == NamingScheme.AUTO:
            scheme = self.infer_naming_scheme(amp)
        if scheme == NamingScheme.AUTO:
            raise ValueError(f"Could not infer naming scheme for '{amp}'")
        scheme_def = SCHEMES[scheme]
        parsed = self._apply_final_state_parity(scheme_def.parse(amp), scheme_def)
        return self._render(self._values_from_parsed(parsed))
