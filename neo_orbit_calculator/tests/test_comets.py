from neo_orbit_calculator.comets import parse_apparition_result


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
