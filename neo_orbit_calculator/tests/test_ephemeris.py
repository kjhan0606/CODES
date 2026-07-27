import pytest

from neo_orbit_calculator.jpl import (
    DE441_PART2_START_JD_TDB,
    DE442_START_JD_TDB,
    DE442_STOP_JD_TDB,
    select_planetary_ephemeris,
)


def test_auto_selects_de442_for_modern_interval() -> None:
    selected = select_planetary_ephemeris(2_460_000.5, 2_500_000.5)
    assert selected.name == "DE442"
    assert selected.kernel_names == ("de442.bsp",)


def test_auto_selects_de441_outside_modern_coverage() -> None:
    selected = select_planetary_ephemeris(2_000_000.5, 2_100_000.5)
    assert selected.name == "DE441"
    assert selected.kernel_names == ("de441_part-1.bsp",)


def test_de441_loads_both_parts_across_1969_overlap() -> None:
    selected = select_planetary_ephemeris(
        DE441_PART2_START_JD_TDB - 100.0,
        DE441_PART2_START_JD_TDB + 100.0,
        requested="de441",
    )
    assert selected.kernel_names == (
        "de441_part-1.bsp",
        "de441_part-2.bsp",
    )


def test_explicit_de441_can_be_used_inside_modern_coverage() -> None:
    selected = select_planetary_ephemeris(
        DE442_START_JD_TDB,
        DE442_STOP_JD_TDB,
        requested="de441",
    )
    assert selected.name == "DE441"


def test_explicit_de442_rejects_long_term_interval() -> None:
    with pytest.raises(ValueError, match="Use DE441"):
        select_planetary_ephemeris(
            DE442_START_JD_TDB - 1.0,
            DE442_STOP_JD_TDB,
            requested="de442",
        )
