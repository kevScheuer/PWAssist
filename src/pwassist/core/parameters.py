"""Helpers for mapping AmpTools parameter names onto their unique fit-frame names.

AmpTools names every production coefficient by its full path,
'<reaction>::<sum>::<amplitude>_<part>'. When coefficients are constrained across
reactions and/or sums, the fit dataframe only stores the unique ones, written as
'<sum>::<amplitude>_<part>' (reactions constrained) or '<amplitude>_<part>' (reactions
and sums constrained). The functions here connect the two naming levels by matching
the trailing '::'-separated segments of a full name exactly, never by substring.
"""

from collections.abc import Iterable

PRODUCTION_PARTS: tuple[str, str] = ("_re", "_im")


def is_production_coefficient(name: str) -> bool:
    """Whether a parameter name is a production coefficient '<...>_re' or '<...>_im'."""
    return name.endswith(PRODUCTION_PARTS)


def coefficient_columns(columns: Iterable[str]) -> list[str]:
    """Return the production coefficient columns ('_re'/'_im') in their given order."""
    return [c for c in columns if is_production_coefficient(c)]


def coefficient_bases(columns: Iterable[str]) -> list[str]:
    """Return unique production coefficient names without their '_re'/'_im' part.

    The order of first appearance is preserved.
    """
    return list(dict.fromkeys(c[:-3] for c in coefficient_columns(columns)))


def name_suffixes(name: str) -> list[str]:
    """All '::'-aligned trailing segments of a name, longest first.

    Example:
        >>> name_suffixes("r::s::amp_re")
        ['r::s::amp_re', 's::amp_re', 'amp_re']
    """
    parts = name.split("::")
    return ["::".join(parts[i:]) for i in range(len(parts))]


def matches_unique(full_name: str, unique_name: str) -> bool:
    """Whether 'full_name' is a (possibly constrained) copy of 'unique_name'.

    True if the names are equal, or 'full_name' ends with '::<unique_name>'.
    """
    return full_name == unique_name or full_name.endswith(f"::{unique_name}")


def group_by_unique(
    full_names: Iterable[str], unique_names: Iterable[str], strict: bool = False
) -> dict[str, list[str]]:
    """Group full parameter names under the unique name they are a copy of.

    Args:
        full_names (Iterable[str]): Parameter names, e.g. from a covariance matrix,
            normally '<reaction>::<sum>::<name>'.
        unique_names (Iterable[str]): Unique names, e.g. production coefficient
            columns of the fit dataframe.
        strict (bool): If True, raise a KeyError when a unique name has no matching
            full name. Defaults to False.

    Returns:
        dict[str, list[str]]: unique name -> matching full names, both in the order
            they were given. Unique names without a match map to an empty list.
            Full names that match no unique name are not included.

    Raises:
        KeyError: If 'strict' and any unique name has no matching full name.
    """
    groups: dict[str, list[str]] = {u: [] for u in unique_names}
    for full in full_names:
        for suffix in name_suffixes(full):
            if suffix in groups:
                groups[suffix].append(full)
                break

    if strict:
        missing = [u for u, fulls in groups.items() if not fulls]
        if missing:
            raise KeyError(
                f"The parameters {missing} have no corresponding 'full'"
                " '<reaction>::<sum>::<name>' parameter name"
            )
    return groups


def split_full_name(name: str) -> tuple[str | None, str | None, str]:
    """Split a parameter name into its (reaction, sum, name) parts.

    Missing leading parts are returned as None, so 'sum::amp_re' gives
    (None, 'sum', 'amp_re') and 'amp_re' gives (None, None, 'amp_re').
    """
    *prefix, base = name.split("::")
    reaction = prefix[-2] if len(prefix) >= 2 else None
    sum_name = prefix[-1] if prefix else None
    return reaction, sum_name, base


def reaction_scale_columns(
    columns: Iterable[str], reactions: Iterable[str], prefix: str = "parScale_"
) -> dict[str, str | None]:
    """Map each reaction to its scale-parameter column, e.g. 'parScale_<x>'.

    A reaction is matched to the scale column whose suffix '<x>' the reaction name
    ends with. The longest suffix wins when several match.

    Args:
        columns (Iterable[str]): Columns of the fit dataframe.
        reactions (Iterable[str]): Reaction names, e.g. 'omegapi_-0.054105'.
        prefix (str): Prefix of the scale parameter columns. Defaults to 'parScale_'.

    Returns:
        dict[str, str | None]: reaction -> scale column, or None if the fit has no
            scale columns at all (all scales are then 1).

    Raises:
        KeyError: If scale columns exist but a reaction cannot be matched to one.
    """
    scale_cols = [c for c in columns if c.startswith(prefix) and not c.endswith("_err")]
    reactions = list(reactions)
    if not scale_cols:
        return {r: None for r in reactions}

    mapping: dict[str, str | None] = {}
    for reaction in reactions:
        candidates = [c for c in scale_cols if reaction.endswith(c[len(prefix) :])]
        if not candidates:
            raise KeyError(
                f"Could not match reaction '{reaction}' to any scale column"
                f" {scale_cols}"
            )
        mapping[reaction] = max(candidates, key=len)
    return mapping
