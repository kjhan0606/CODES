from pathlib import Path

import numpy as np

from neo_orbit_calculator.core import ForceModel
from neo_orbit_calculator.jpl import normalize_command


def test_force_model_defaults_are_physical() -> None:
    model = ForceModel()
    assert model.relativity_1pn
    assert model.area_mass_m2_kg == 0.0
    assert model.poynting_robertson
    assert model.solar_wind_drag
    assert np.isclose(model.solar_wind_to_pr, 0.35)


def test_designation_normalization() -> None:
    assert normalize_command("99942") == "99942;"
    assert normalize_command("DES=2004 MN4;") == "DES=2004 MN4;"
