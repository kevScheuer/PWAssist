import pytest

from pwassist.core.parameters import (
    coefficient_bases,
    coefficient_columns,
    group_by_unique,
    matches_unique,
    name_suffixes,
    reaction_scale_columns,
    split_full_name,
)


class TestNames:

    def test_coefficient_columns_and_bases(self):
        columns = ["likelihood", "a_re", "a_im", "b_re", "b_im", "a", "a_b", "a_err"]
        assert coefficient_columns(columns) == ["a_re", "a_im", "b_re", "b_im"]
        assert coefficient_bases(columns) == ["a", "b"]

    def test_name_suffixes(self):
        assert name_suffixes("r::s::amp_re") == ["r::s::amp_re", "s::amp_re", "amp_re"]
        assert name_suffixes("amp_re") == ["amp_re"]

    def test_split_full_name(self):
        assert split_full_name("r::s::amp_re") == ("r", "s", "amp_re")
        assert split_full_name("s::amp_re") == (None, "s", "amp_re")
        assert split_full_name("amp_re") == (None, None, "amp_re")

    @pytest.mark.parametrize(
        "full, unique, expected",
        [
            ("r::s::m1p0S_re", "m1p0S_re", True),
            ("r::s::m1p0S_re", "s::m1p0S_re", True),
            ("r::s::m1p0S_re", "m1p0S_re", True),
            ("m1p0S_re", "m1p0S_re", True),
            # substring, but not a '::'-aligned suffix
            ("r::s::m1p0S_re", "1p0S_re", False),
            ("r::s::xm1p0S_re", "m1p0S_re", False),
            ("r::s::m1p0S_re", "m1p0S_im", False),
        ],
    )
    def test_matches_unique(self, full, unique, expected):
        assert matches_unique(full, unique) is expected


class TestGroupByUnique:

    FULL = [
        "r1::sA::a_re",
        "r1::sB::a_re",
        "r2::sA::a_re",
        "r1::sA::ba_re",  # 'a_re' is a substring, but not a copy
        "parScale_r2",
    ]

    def test_groups_constrained_copies(self):
        groups = group_by_unique(self.FULL, ["a_re", "ba_re"])
        assert groups == {
            "a_re": ["r1::sA::a_re", "r1::sB::a_re", "r2::sA::a_re"],
            "ba_re": ["r1::sA::ba_re"],
        }

    def test_sum_level_names(self):
        groups = group_by_unique(self.FULL, ["sA::a_re", "sB::a_re"])
        assert groups == {
            "sA::a_re": ["r1::sA::a_re", "r2::sA::a_re"],
            "sB::a_re": ["r1::sB::a_re"],
        }

    def test_strict_raises_on_missing(self):
        assert group_by_unique(self.FULL, ["c_re"]) == {"c_re": []}
        with pytest.raises(KeyError):
            group_by_unique(self.FULL, ["c_re"], strict=True)


class TestReactionScaleColumns:

    def test_matches_reaction_suffix(self):
        columns = ["parScale_-0.05", "parScale_-0.05_err", "parScale_0.71"]
        mapping = reaction_scale_columns(columns, ["omegapi_-0.05", "omegapi_0.71"])
        assert mapping == {
            "omegapi_-0.05": "parScale_-0.05",
            "omegapi_0.71": "parScale_0.71",
        }

    def test_no_scale_columns(self):
        assert reaction_scale_columns(["a_re"], ["r1"]) == {"r1": None}

    def test_unmatched_reaction_raises(self):
        with pytest.raises(KeyError):
            reaction_scale_columns(["parScale_0.71"], ["omegapi_-0.05"])
