from neo_orbit_calculator.comets import parse_apparition_result
from neo_orbit_calculator.historical import (
    JOSEON_HALLEY_1759,
    epoch_grid,
    evaluate_joseon_constraints,
)


SAMPLE = """
JPL/HORIZONS                      1P/Halley
Rec #:90000018                  Soln.date: -
EC= .96887              QR= .5745               TP= 2110493.4339999999
OM= 47.624              W= 102.473              IN= 163.112
A= 18.45486668808224    MA= 359.8392057323519
PER= 79.281995157322    N= .012431906
B= 16.477891                                    TP= 1066-Mar-20.9339999999
"""


def test_parse_historical_comet_apparition() -> None:
    apparition = parse_apparition_result(SAMPLE, "1P")
    assert apparition.record == 90000018
    assert apparition.return_year == 1066
    assert apparition.perihelion_calendar == "1066-Mar-20.9339999999"
    assert apparition.osculating_period_year == 79.281995157322
    assert apparition.semimajor_axis_au == 18.45486668808224


def test_historical_epoch_grid_is_centered() -> None:
    epochs = epoch_grid("1759-04-06T20:30:00", span_days=4.0, samples=5)
    assert epochs[2].startswith("1759-04-06T20:30:00")
    assert epochs[0].startswith("1759-04-04T20:30:00")
    assert epochs[-1].startswith("1759-04-08T20:30:00")


def test_joseon_constraints_separate_raw_polar_distance() -> None:
    row = {
        "epoch_utc": "1759-04-06T20:30:00",
        "apparition_record": 90000027,
        "ra_icrs_deg": 329.728813679,
        "dec_icrs_deg": -9.650470171,
        "dec_apparent_deg": -10.791848942,
        "elevation_deg": 17.897941226,
        "solar_elongation_deg": 51.6742,
    }
    result = evaluate_joseon_constraints(row, JOSEON_HALLEY_1759)
    assert result["inside_xu_ra_interval"] is True
    assert result["north_of_liyu"] is True
    assert result["north_polar_distance_used_in_score"] is False
    assert result["reported_north_polar_distance_raw_deg"] == 116.0
