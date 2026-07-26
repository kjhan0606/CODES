"""Local high-accuracy NEO propagation using JPL DE440 perturber states."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import spiceypy as spice
from scipy.integrate import solve_ivp

from .jpl import ensure_generic_kernels, ensure_sb441_n16, horizons_vectors

AU_KM = 149_597_870.700
DAY_S = 86_400.0
C_KM_S = 299_792.458
SOLAR_PRESSURE_N_M2 = 4.56e-6
PROTON_MASS_KG = 1.67262192595e-27
J2000_JD = 2_451_545.0
JULIAN_CENTURY_S = 36_525.0 * DAY_S

PERTURBERS = (
    "SUN",
    "MERCURY BARYCENTER",
    "VENUS BARYCENTER",
    "EARTH",
    "MOON",
    "MARS BARYCENTER",
    "JUPITER BARYCENTER",
    "SATURN BARYCENTER",
    "URANUS BARYCENTER",
    "NEPTUNE BARYCENTER",
    "PLUTO BARYCENTER",
)

# JPL IOM 392R-21-005, Table 1. The file SB441-N16 contains these
# 16 most massive main-belt perturbers. GMs are converted from au^3/day^2.
_GM_CONVERSION = AU_KM**3 / DAY_S**2
LARGE_ASTEROIDS = (
    ("2000001", "Ceres", 1.3964518123081070e-13 * _GM_CONVERSION),
    ("2000004", "Vesta", 3.8548000225257904e-14 * _GM_CONVERSION),
    ("2000002", "Pallas", 3.0471146330043200e-14 * _GM_CONVERSION),
    ("2000010", "Hygiea", 1.2542530761640810e-14 * _GM_CONVERSION),
    ("2000511", "Davida", 8.6836253492286545e-15 * _GM_CONVERSION),
    ("2000704", "Interamnia", 6.3110343420878887e-15 * _GM_CONVERSION),
    ("2000052", "Europa", 5.9824315264869841e-15 * _GM_CONVERSION),
    ("2000087", "Sylvia", 4.8345606546105521e-15 * _GM_CONVERSION),
    ("2000015", "Eunomia", 4.5107799051436795e-15 * _GM_CONVERSION),
    ("2000003", "Juno", 4.2823439677995011e-15 * _GM_CONVERSION),
    ("2000016", "Psyche", 3.5445002842488978e-15 * _GM_CONVERSION),
    ("2000107", "Camilla", 3.2191392075878588e-15 * _GM_CONVERSION),
    ("2000088", "Thisbe", 2.6529436610356353e-15 * _GM_CONVERSION),
    ("2000007", "Iris", 2.5416014973471498e-15 * _GM_CONVERSION),
    ("2000031", "Euphrosyne", 2.4067012218937576e-15 * _GM_CONVERSION),
    ("2000065", "Cybele", 2.0917175955133682e-15 * _GM_CONVERSION),
)
PERTURBER_KEYS = PERTURBERS + tuple(item[0] for item in LARGE_ASTEROIDS)

# Reference equatorial radii and unnormalized zonal coefficients. The
# expansion is evaluated through J6. A zero coefficient leaves that degree
# inactive. Planet-system barycenter positions are used where DE440s does not
# provide a separate planet-center state, which is adequate only outside the
# satellite-system scale.
ZONAL_GRAVITY = {
    "SUN": (695_700.0, 2.20e-7, 0.0, 0.0, 10),
    "MERCURY BARYCENTER": (2_439.7, 5.031e-5, 0.0, 0.0, 199),
    "VENUS BARYCENTER": (6_051.8, 4.458e-6, 0.0, 0.0, 299),
    "EARTH": (
        6_378.1363,
        1.08262668e-3,
        -1.61962159137e-6,
        5.40681239107e-7,
        399,
    ),
    "MOON": (1_738.1, 2.032337e-4, -9.59193e-6, 1.4450e-6, 301),
    "MARS BARYCENTER": (
        3_396.19,
        1.96045e-3,
        -1.5377e-5,
        1.60e-6,
        499,
    ),
    "JUPITER BARYCENTER": (
        71_492.0,
        1.4696572e-2,
        -5.86609e-4,
        3.4198e-5,
        599,
    ),
    "SATURN BARYCENTER": (
        60_268.0,
        1.629071e-2,
        -9.3583e-4,
        8.614e-5,
        699,
    ),
    "URANUS BARYCENTER": (
        25_559.0,
        3.5107e-3,
        -3.42e-5,
        0.0,
        799,
    ),
    "NEPTUNE BARYCENTER": (
        24_764.0,
        3.40843e-3,
        -3.34e-5,
        0.0,
        899,
    ),
}


@dataclass(frozen=True)
class ForceModel:
    """Optional force terms beyond Newtonian major-body gravity."""

    relativity_1pn: bool = True
    full_multibody_1pn: bool = True
    area_mass_m2_kg: float = 0.0
    radiation_coefficient: float = 1.0
    poynting_robertson: bool = True
    solar_wind_drag: bool = True
    solar_wind_density_cm3: float = 5.0
    solar_wind_speed_km_s: float = 400.0
    solar_wind_momentum_factor: float = 1.2
    planetary_zonal_harmonics: bool = True
    a1_au_day2: float = 0.0
    a2_au_day2: float = 0.0
    a3_au_day2: float = 0.0
    nongrav_law: str = "inverse_square"
    outgassing_r0_au: float = 2.808
    outgassing_m: float = 2.15
    outgassing_n: float = 5.093
    outgassing_k: float = 4.6142
    outgassing_alpha: float = 0.111262
    outgassing_lag_days: float = 0.0

    def __post_init__(self) -> None:
        if self.area_mass_m2_kg < 0:
            raise ValueError("area_mass_m2_kg must be non-negative.")
        if self.radiation_coefficient < 0:
            raise ValueError("radiation_coefficient must be non-negative.")
        if self.solar_wind_density_cm3 < 0:
            raise ValueError("solar_wind_density_cm3 must be non-negative.")
        if self.solar_wind_speed_km_s <= 0:
            raise ValueError("solar_wind_speed_km_s must be positive.")
        if self.solar_wind_momentum_factor < 0:
            raise ValueError("solar_wind_momentum_factor must be non-negative.")
        if self.nongrav_law not in {"inverse_square", "marsden"}:
            raise ValueError(
                "nongrav_law must be 'inverse_square' or 'marsden'."
            )
        if self.outgassing_r0_au <= 0:
            raise ValueError("outgassing_r0_au must be positive.")
        if min(
            self.outgassing_m,
            self.outgassing_n,
            self.outgassing_k,
            self.outgassing_alpha,
        ) < 0:
            raise ValueError("Outgassing-law coefficients must be non-negative.")


@dataclass
class PropagationResult:
    designation: str
    jd_tdb: np.ndarray
    state_km_kms: np.ndarray
    horizons_state_km_kms: np.ndarray | None
    position_residual_km: np.ndarray | None
    velocity_residual_mm_s: np.ndarray | None
    function_evaluations: int
    kernel: str
    warning: str


class DE440Environment:
    """SPICE-backed planetary positions and gravity parameters."""

    def __init__(self, kernel_dir: Path, include_large_asteroids: bool = True):
        paths = ensure_generic_kernels(kernel_dir)
        spice.kclear()
        spice.furnsh(str(paths["naif0012.tls"]))
        spice.furnsh(str(paths["gm_de440.tpc"]))
        spice.furnsh(str(paths["pck00011.tpc"]))
        spice.furnsh(str(paths["de440s.bsp"]))
        if include_large_asteroids:
            spice.furnsh(str(ensure_sb441_n16(kernel_dir)))
        self.gm = {
            body: float(spice.bodvrd(body, "GM", 1)[1][0])
            for body in PERTURBERS
        }
        for spice_id, _name, gm in LARGE_ASTEROIDS:
            self.gm[spice_id] = gm if include_large_asteroids else 0.0
        self.include_large_asteroids = include_large_asteroids
        self.zonal = {}
        for body, (radius, j2, j4, j6, orientation_id) in ZONAL_GRAVITY.items():
            try:
                pole_ra = np.asarray(
                    spice.bodvcd(orientation_id, "POLE_RA", 3)[1],
                    dtype=float,
                )
                pole_dec = np.asarray(
                    spice.bodvcd(orientation_id, "POLE_DEC", 3)[1],
                    dtype=float,
                )
            except spice.SpiceyError:
                pole_ra = np.array([0.0, 0.0, 0.0])
                pole_dec = np.array([90.0, 0.0, 0.0])
            self.zonal[body] = {
                "radius_km": radius,
                "coefficients": np.array([j2, j4, j6], dtype=float),
                "pole_ra_deg": pole_ra,
                "pole_dec_deg": pole_dec,
            }

    @staticmethod
    def jd_to_et(jd_tdb: float) -> float:
        return float(spice.unitim(float(jd_tdb), "JDTDB", "ET"))

    @staticmethod
    def state(body: str, et: float) -> np.ndarray:
        return np.asarray(
            spice.spkezr(body, et, "J2000", "NONE", "SOLAR SYSTEM BARYCENTER")[0],
            dtype=float,
        )

    def pole(self, body: str, et: float) -> np.ndarray:
        """Return the IAU north-pole direction in the J2000 frame."""
        model = self.zonal[body]
        centuries = et / JULIAN_CENTURY_S
        powers = np.array([1.0, centuries, centuries**2])
        ra = np.deg2rad(float(np.dot(model["pole_ra_deg"], powers)))
        dec = np.deg2rad(float(np.dot(model["pole_dec_deg"], powers)))
        return np.array(
            [
                np.cos(dec) * np.cos(ra),
                np.cos(dec) * np.sin(ra),
                np.sin(dec),
            ]
        )


def _unit(vector: np.ndarray) -> tuple[np.ndarray, float]:
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        raise FloatingPointError("Encountered a zero-length dynamical vector.")
    return vector / norm, norm


def solar_1pn_acceleration(
    position_km: np.ndarray,
    velocity_km_s: np.ndarray,
    mu_sun_km3_s2: float,
) -> np.ndarray:
    """Solar Schwarzschild test-particle acceleration in harmonic coordinates."""
    r_hat, radius = _unit(position_km)
    del r_hat
    speed2 = float(np.dot(velocity_km_s, velocity_km_s))
    radial_product = float(np.dot(position_km, velocity_km_s))
    return (
        mu_sun_km3_s2
        / (C_KM_S**2 * radius**3)
        * (
            (4.0 * mu_sun_km3_s2 / radius - speed2) * position_km
            + 4.0 * radial_product * velocity_km_s
        )
    )


def full_multibody_1pn_acceleration(
    position_km: np.ndarray,
    velocity_km_s: np.ndarray,
    body_states_km_kms: np.ndarray,
    gm_km3_s2: np.ndarray,
) -> np.ndarray:
    """Return the EIH 1PN correction for a massless barycentric target.

    This is the test-particle limit of the N-body expression implemented by
    REBOUNDx ``gr_full``. Newtonian source accelerations are sufficient
    because their post-Newtonian corrections enter only at order c^-4.
    """
    states = np.asarray(body_states_km_kms, dtype=float)
    gm = np.asarray(gm_km3_s2, dtype=float)
    if states.ndim != 2 or states.shape[1] != 6:
        raise ValueError("body_states_km_kms must have shape (N, 6).")
    if gm.shape != (states.shape[0],):
        raise ValueError("gm_km3_s2 must have one value per body state.")

    active = gm > 0.0
    states = states[active]
    gm = gm[active]
    source_acceleration = np.zeros((len(gm), 3))
    source_potential = np.zeros(len(gm))
    for source in range(len(gm)):
        for other in range(len(gm)):
            if source == other:
                continue
            displacement = states[other, :3] - states[source, :3]
            distance = float(np.linalg.norm(displacement))
            if distance <= 0.0:
                raise ValueError("Two relativistic source bodies coincide.")
            source_acceleration[source] += (
                gm[other] * displacement / distance**3
            )
            source_potential[source] += gm[other] / distance

    displacement = position_km - states[:, :3]
    distance = np.linalg.norm(displacement, axis=1)
    if np.any(distance <= 0.0):
        raise ValueError("The target coincides with a relativistic source.")
    target_potential = float(np.sum(gm / distance))
    target_speed2 = float(np.dot(velocity_km_s, velocity_km_s))
    correction = np.zeros(3)
    c2 = C_KM_S**2
    for source in range(len(gm)):
        source_velocity = states[source, 3:]
        projected_velocity = float(
            np.dot(displacement[source], source_velocity)
        )
        factor1 = (
            4.0 * target_potential
            + source_potential[source]
            - target_speed2
            - 2.0 * float(np.dot(source_velocity, source_velocity))
            + 4.0 * float(np.dot(velocity_km_s, source_velocity))
            + 1.5 * projected_velocity**2 / distance[source] ** 2
            + 0.5
            * float(
                np.dot(
                    displacement[source],
                    source_acceleration[source],
                )
            )
        ) / c2
        correction += (
            gm[source]
            * displacement[source]
            / distance[source] ** 3
            * factor1
        )

        relative_velocity = velocity_km_s - source_velocity
        factor2 = float(
            np.dot(
                displacement[source],
                4.0 * velocity_km_s - 3.0 * source_velocity,
            )
        )
        correction += gm[source] / c2 * (
            factor2 * relative_velocity / distance[source] ** 3
            + 3.5 * source_acceleration[source] / distance[source]
        )
    return correction


def photon_radiation_acceleration(
    position_km: np.ndarray,
    velocity_km_s: np.ndarray,
    area_mass_m2_kg: float,
    radiation_coefficient: float,
    include_pr: bool = True,
) -> np.ndarray:
    """Direct radiation pressure plus the complete first-order PR term."""
    r_hat, radius = _unit(position_km)
    amplitude = (
        SOLAR_PRESSURE_N_M2
        * radiation_coefficient
        * area_mass_m2_kg
        / 1000.0
        * (AU_KM / radius) ** 2
    )
    if not include_pr:
        return amplitude * r_hat
    radial_velocity = float(np.dot(velocity_km_s, r_hat))
    return amplitude * (
        (1.0 - radial_velocity / C_KM_S) * r_hat
        - velocity_km_s / C_KM_S
    )


def solar_wind_acceleration(
    position_km: np.ndarray,
    velocity_km_s: np.ndarray,
    area_mass_m2_kg: float,
    density_cm3: float,
    speed_km_s: float,
    momentum_factor: float,
) -> np.ndarray:
    """Momentum transfer from a stationary, radial proton solar wind."""
    r_hat, radius = _unit(position_km)
    wind_velocity = speed_km_s * r_hat
    relative_velocity = wind_velocity - velocity_km_s
    relative_speed = float(np.linalg.norm(relative_velocity))
    pressure_1au = (
        density_cm3
        * 1.0e6
        * PROTON_MASS_KG
        * (speed_km_s * 1000.0) ** 2
        * momentum_factor
    )
    amplitude = (
        pressure_1au
        * area_mass_m2_kg
        / 1000.0
        * (AU_KM / radius) ** 2
    )
    return (
        amplitude
        * relative_speed
        * relative_velocity
        / speed_km_s**2
    )


def zonal_harmonic_acceleration(
    relative_position_km: np.ndarray,
    pole: np.ndarray,
    mu_km3_s2: float,
    reference_radius_km: float,
    coefficients: np.ndarray,
) -> np.ndarray:
    """Axisymmetric J2, J4, and J6 acceleration in an inertial frame."""
    r_hat, radius = _unit(relative_position_km)
    pole_hat, _ = _unit(pole)
    sine_latitude = float(np.dot(r_hat, pole_hat))
    acceleration = np.zeros(3)
    polynomials = (
        (
            0.5 * (3.0 * sine_latitude**2 - 1.0),
            3.0 * sine_latitude,
        ),
        (
            (35.0 * sine_latitude**4 - 30.0 * sine_latitude**2 + 3.0)
            / 8.0,
            (140.0 * sine_latitude**3 - 60.0 * sine_latitude) / 8.0,
        ),
        (
            (
                231.0 * sine_latitude**6
                - 315.0 * sine_latitude**4
                + 105.0 * sine_latitude**2
                - 5.0
            )
            / 16.0,
            (
                1386.0 * sine_latitude**5
                - 1260.0 * sine_latitude**3
                + 210.0 * sine_latitude
            )
            / 16.0,
        ),
    )
    for degree, coefficient, (polynomial, derivative) in zip(
        (2, 4, 6),
        coefficients,
        polynomials,
        strict=True,
    ):
        if coefficient == 0.0:
            continue
        scale = (
            mu_km3_s2
            * coefficient
            * reference_radius_km**degree
            / radius ** (degree + 2)
        )
        acceleration += scale * (
            (
                (degree + 1) * polynomial
                + sine_latitude * derivative
            )
            * r_hat
            - derivative * pole_hat
        )
    return acceleration


def _stumpff(z: float) -> tuple[float, float]:
    if z > 1.0e-8:
        root = np.sqrt(z)
        return (1.0 - np.cos(root)) / z, (root - np.sin(root)) / root**3
    if z < -1.0e-8:
        root = np.sqrt(-z)
        return (np.cosh(root) - 1.0) / (-z), (
            np.sinh(root) - root
        ) / root**3
    return (
        0.5 - z / 24.0 + z**2 / 720.0,
        1.0 / 6.0 - z / 120.0 + z**2 / 5040.0,
    )


def kepler_shift_position(
    position_km: np.ndarray,
    velocity_km_s: np.ndarray,
    dt_s: float,
    mu_km3_s2: float,
) -> np.ndarray:
    """Shift an osculating two-body state with universal variables."""
    if dt_s == 0.0:
        return np.asarray(position_km, dtype=float).copy()
    radius = float(np.linalg.norm(position_km))
    speed2 = float(np.dot(velocity_km_s, velocity_km_s))
    radial_velocity = float(np.dot(position_km, velocity_km_s)) / radius
    alpha = 2.0 / radius - speed2 / mu_km3_s2
    sqrt_mu = np.sqrt(mu_km3_s2)
    if abs(alpha) > 1.0e-12:
        anomaly = sqrt_mu * abs(alpha) * dt_s
    else:
        anomaly = sqrt_mu * dt_s / radius
    for _ in range(50):
        z = alpha * anomaly**2
        c_value, s_value = _stumpff(z)
        function = (
            radius
            * radial_velocity
            / sqrt_mu
            * anomaly**2
            * c_value
            + (1.0 - alpha * radius) * anomaly**3 * s_value
            + radius * anomaly
            - sqrt_mu * dt_s
        )
        derivative = (
            radius
            * radial_velocity
            / sqrt_mu
            * anomaly
            * (1.0 - z * s_value)
            + (1.0 - alpha * radius) * anomaly**2 * c_value
            + radius
        )
        update = function / derivative
        anomaly -= update
        if abs(update) < 1.0e-11 * max(1.0, abs(anomaly)):
            break
    z = alpha * anomaly**2
    c_value, s_value = _stumpff(z)
    f_value = 1.0 - anomaly**2 / radius * c_value
    g_value = dt_s - anomaly**3 / sqrt_mu * s_value
    return f_value * position_km + g_value * velocity_km_s


def marsden_outgassing_scale(
    radius_au: float,
    r0_au: float = 2.808,
    m_value: float = 2.15,
    n_value: float = 5.093,
    k_value: float = 4.6142,
    alpha_value: float = 0.111262,
) -> float:
    """Evaluate the normalized Marsden water-ice sublimation law."""
    ratio = radius_au / r0_au
    return float(
        alpha_value
        * ratio ** (-m_value)
        * (1.0 + ratio**n_value) ** (-k_value)
    )


def _rhs_factory(
    environment: DE440Environment,
    epoch_et: float,
    model: ForceModel,
):
    mu_sun = environment.gm["SUN"]
    a1_scale = AU_KM / DAY_S**2

    def rhs(seconds: float, state: np.ndarray) -> np.ndarray:
        position = state[:3]
        velocity = state[3:]
        et = epoch_et + seconds
        acceleration = np.zeros(3)
        body_states: dict[str, np.ndarray] = {}

        for body in PERTURBER_KEYS:
            if environment.gm[body] == 0.0:
                continue
            body_state = environment.state(body, et)
            body_states[body] = body_state
            displacement = body_state[:3] - position
            _, distance = _unit(displacement)
            acceleration += environment.gm[body] * displacement / distance**3
            if (
                model.planetary_zonal_harmonics
                and body in environment.zonal
            ):
                zonal = environment.zonal[body]
                acceleration += zonal_harmonic_acceleration(
                    -displacement,
                    environment.pole(body, et),
                    environment.gm[body],
                    float(zonal["radius_km"]),
                    np.asarray(zonal["coefficients"]),
                )

        sun_state = body_states["SUN"]
        r_vec = position - sun_state[:3]
        v_vec = velocity - sun_state[3:]
        r_hat, r = _unit(r_vec)
        h_vec = np.cross(r_vec, v_vec)
        h_hat, _ = _unit(h_vec)
        t_hat = np.cross(h_hat, r_hat)

        if model.relativity_1pn:
            if model.full_multibody_1pn:
                active_bodies = [
                    body for body in PERTURBER_KEYS if body in body_states
                ]
                acceleration += full_multibody_1pn_acceleration(
                    position,
                    velocity,
                    np.asarray(
                        [body_states[body] for body in active_bodies]
                    ),
                    np.asarray(
                        [environment.gm[body] for body in active_bodies]
                    ),
                )
            else:
                acceleration += solar_1pn_acceleration(
                    r_vec,
                    v_vec,
                    mu_sun,
                )

        if model.area_mass_m2_kg > 0:
            acceleration += photon_radiation_acceleration(
                r_vec,
                v_vec,
                model.area_mass_m2_kg,
                model.radiation_coefficient,
                include_pr=model.poynting_robertson,
            )
            if model.solar_wind_drag:
                acceleration += solar_wind_acceleration(
                    r_vec,
                    v_vec,
                    model.area_mass_m2_kg,
                    model.solar_wind_density_cm3,
                    model.solar_wind_speed_km_s,
                    model.solar_wind_momentum_factor,
                )

        if model.nongrav_law == "marsden":
            lag_position = kepler_shift_position(
                r_vec,
                v_vec,
                -model.outgassing_lag_days * DAY_S,
                mu_sun,
            )
            radial_scale = marsden_outgassing_scale(
                float(np.linalg.norm(lag_position)) / AU_KM,
                model.outgassing_r0_au,
                model.outgassing_m,
                model.outgassing_n,
                model.outgassing_k,
                model.outgassing_alpha,
            )
        else:
            radial_scale = (AU_KM / r) ** 2
        acceleration += (
            radial_scale
            * a1_scale
            * (
                model.a1_au_day2 * r_hat
                + model.a2_au_day2 * t_hat
                + model.a3_au_day2 * h_hat
            )
        )
        return np.concatenate((velocity, acceleration))

    return rhs


def propagate_custom(
    designation: str,
    start_jd_tdb: float,
    stop_jd_tdb: float,
    samples: int = 401,
    model: ForceModel | None = None,
    kernel_dir: Path | str = Path("kernels"),
    rtol: float = 3e-12,
    atol_position_km: float = 1e-6,
    atol_velocity_kms: float = 1e-12,
    validate_horizons: bool = True,
    include_large_asteroids: bool = True,
    backend: str = "fortran",
) -> PropagationResult:
    """Integrate a massless NEO in the ICRF barycentric frame.

    The initial state is fetched from Horizons. DE440s supplies the positions
    and velocities of the Sun, planets, Earth, and Moon. This local model is a
    sensitivity tool. Horizons SPK output remains the authoritative trajectory.
    """
    if stop_jd_tdb <= start_jd_tdb:
        raise ValueError("stop_jd_tdb must be later than start_jd_tdb.")
    if samples < 2:
        raise ValueError("samples must be at least 2.")
    model = model or ForceModel()
    environment = DE440Environment(
        Path(kernel_dir),
        include_large_asteroids=include_large_asteroids,
    )
    initial, _ = horizons_vectors(designation, [start_jd_tdb])
    initial_state = initial[0, 1:7]
    epoch_et = environment.jd_to_et(start_jd_tdb)
    output_seconds = np.linspace(
        0.0,
        (stop_jd_tdb - start_jd_tdb) * DAY_S,
        samples,
    )
    atol = np.array(
        [atol_position_km] * 3 + [atol_velocity_kms] * 3,
        dtype=float,
    )
    if backend == "fortran":
        from .fortran_backend import FortranIntegrator

        integrator = FortranIntegrator(environment)
        states, function_evaluations = integrator.propagate(
            initial_state,
            epoch_et,
            output_seconds,
            model,
            rtol=min(rtol, 3e-14),
            atol_position_km=atol_position_km,
            atol_velocity_kms=atol_velocity_kms,
            max_step_days=2.0,
        )
        kernel_description = (
            f"Fortran real{integrator.precision_digits} arithmetic + "
            "JPL DE440s/SB441-N16 binary64 SPK"
        )
        if model.relativity_1pn:
            kernel_description += (
                " + full multi-body 1PN"
                if model.full_multibody_1pn
                else " + solar-only 1PN"
            )
        if model.planetary_zonal_harmonics:
            kernel_description += " + NAIF PCK J2/J4/J6"
    elif backend == "scipy":
        solution = solve_ivp(
            _rhs_factory(environment, epoch_et, model),
            (float(output_seconds[0]), float(output_seconds[-1])),
            initial_state,
            method="DOP853",
            t_eval=output_seconds,
            rtol=rtol,
            atol=atol,
            max_step=5.0 * DAY_S,
        )
        if not solution.success:
            raise RuntimeError(solution.message)
        states = solution.y.T
        function_evaluations = solution.nfev
        kernel_description = "SciPy DOP853 binary64 + JPL DE440s/SB441-N16"
        if model.relativity_1pn:
            kernel_description += (
                " + full multi-body 1PN"
                if model.full_multibody_1pn
                else " + solar-only 1PN"
            )
        if model.planetary_zonal_harmonics:
            kernel_description += " + NAIF PCK J2/J4/J6"
    else:
        raise ValueError("backend must be 'fortran' or 'scipy'.")
    jd = start_jd_tdb + output_seconds / DAY_S

    reference = None
    position_residual = None
    velocity_residual = None
    if validate_horizons:
        indices = np.unique(np.linspace(0, samples - 1, min(101, samples), dtype=int))
        reference_rows, _ = horizons_vectors(designation, jd[indices])
        reference = np.full_like(states, np.nan)
        reference[indices] = reference_rows[:, 1:7]
        delta = states[indices] - reference_rows[:, 1:7]
        position_residual = np.full(samples, np.nan)
        velocity_residual = np.full(samples, np.nan)
        position_residual[indices] = np.linalg.norm(delta[:, :3], axis=1)
        velocity_residual[indices] = np.linalg.norm(delta[:, 3:], axis=1) * 1e6

    warning = (
        "A 100-year numerical trajectory is not a 100-year impact prediction. "
        "Orbit-solution covariance, close encounters, asteroid perturbers, "
        "shape, spin, thermal inertia, and fitted non-gravitational parameters "
        "usually dominate long-horizon uncertainty."
    )
    return PropagationResult(
        designation=designation,
        jd_tdb=jd,
        state_km_kms=states,
        horizons_state_km_kms=reference,
        position_residual_km=position_residual,
        velocity_residual_mm_s=velocity_residual,
        function_evaluations=function_evaluations,
        kernel=kernel_description,
        warning=warning,
    )
