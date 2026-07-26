from pathlib import Path

import numpy as np

from neo_orbit_calculator.core import (
    AU_KM,
    ForceModel,
    full_multibody_1pn_acceleration,
    kepler_shift_position,
    marsden_outgassing_scale,
    photon_radiation_acceleration,
    solar_1pn_acceleration,
    solar_wind_acceleration,
    zonal_harmonic_acceleration,
)
from neo_orbit_calculator.jpl import normalize_command


def test_force_model_defaults_are_physical() -> None:
    model = ForceModel()
    assert model.relativity_1pn
    assert model.full_multibody_1pn
    assert model.area_mass_m2_kg == 0.0
    assert model.poynting_robertson
    assert model.solar_wind_drag
    assert model.planetary_zonal_harmonics
    assert model.nongrav_law == "inverse_square"


def test_solar_1pn_is_radial_for_a_stationary_test_particle() -> None:
    mu = 1.3271244004127942e11
    acceleration = solar_1pn_acceleration(
        np.array([AU_KM, 0.0, 0.0]),
        np.zeros(3),
        mu,
    )
    assert acceleration[0] > 0.0
    assert np.allclose(acceleration[1:], 0.0)


def test_full_multibody_1pn_reduces_to_solar_schwarzschild() -> None:
    mu = 1.3271244004127942e11
    sun_state = np.zeros((1, 6))
    position = np.array([0.81 * AU_KM, 0.37 * AU_KM, -0.04 * AU_KM])
    velocity = np.array([-12.1, 31.4, 2.3])
    expected = solar_1pn_acceleration(position, velocity, mu)
    actual = full_multibody_1pn_acceleration(
        position,
        velocity,
        sun_state,
        np.array([mu]),
    )
    np.testing.assert_allclose(actual, expected, rtol=2.0e-15, atol=1.0e-24)


def test_pr_and_solar_wind_remove_tangential_momentum() -> None:
    position = np.array([AU_KM, 0.0, 0.0])
    velocity = np.array([0.0, 30.0, 0.0])
    photon = photon_radiation_acceleration(
        position,
        velocity,
        area_mass_m2_kg=1.0,
        radiation_coefficient=1.0,
    )
    wind = solar_wind_acceleration(
        position,
        velocity,
        area_mass_m2_kg=1.0,
        density_cm3=5.0,
        speed_km_s=400.0,
        momentum_factor=1.2,
    )
    assert photon[0] > 0.0 and photon[1] < 0.0
    assert wind[0] > 0.0 and wind[1] < 0.0


def test_equatorial_j2_acceleration_matches_closed_form() -> None:
    mu = 398600.435507
    radius = 6378.1363
    distance = 7000.0
    j2 = 1.08262668e-3
    acceleration = zonal_harmonic_acceleration(
        np.array([distance, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        mu,
        radius,
        np.array([j2, 0.0, 0.0]),
    )
    expected_x = -1.5 * mu * j2 * radius**2 / distance**4
    assert np.isclose(acceleration[0], expected_x, rtol=1.0e-14)
    assert np.allclose(acceleration[1:], 0.0)


def test_marsden_law_and_kepler_time_shift() -> None:
    assert np.isclose(marsden_outgassing_scale(1.0), 1.0, rtol=3.0e-5)
    mu = 1.3271244004127942e11
    circular_speed = np.sqrt(mu / AU_KM)
    period = 2.0 * np.pi * np.sqrt(AU_KM**3 / mu)
    shifted = kepler_shift_position(
        np.array([AU_KM, 0.0, 0.0]),
        np.array([0.0, circular_speed, 0.0]),
        period / 4.0,
        mu,
    )
    assert abs(shifted[0]) < 1.0e-7 * AU_KM
    assert np.isclose(shifted[1], AU_KM, rtol=1.0e-10)


def test_designation_normalization() -> None:
    assert normalize_command("99942") == "99942;"
    assert normalize_command("DES=2004 MN4;") == "DES=2004 MN4;"
