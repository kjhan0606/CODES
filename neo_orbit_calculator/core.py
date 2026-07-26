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


@dataclass(frozen=True)
class ForceModel:
    """Optional force terms beyond Newtonian major-body gravity."""

    relativity_1pn: bool = True
    area_mass_m2_kg: float = 0.0
    radiation_coefficient: float = 1.0
    poynting_robertson: bool = True
    solar_wind_drag: bool = True
    solar_wind_to_pr: float = 0.35
    a1_au_day2: float = 0.0
    a2_au_day2: float = 0.0
    a3_au_day2: float = 0.0

    def __post_init__(self) -> None:
        if self.area_mass_m2_kg < 0:
            raise ValueError("area_mass_m2_kg must be non-negative.")
        if self.radiation_coefficient < 0:
            raise ValueError("radiation_coefficient must be non-negative.")
        if self.solar_wind_to_pr < 0:
            raise ValueError("solar_wind_to_pr must be non-negative.")


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

    @staticmethod
    def jd_to_et(jd_tdb: float) -> float:
        return float(spice.unitim(float(jd_tdb), "JDTDB", "ET"))

    @staticmethod
    def state(body: str, et: float) -> np.ndarray:
        return np.asarray(
            spice.spkezr(body, et, "J2000", "NONE", "SOLAR SYSTEM BARYCENTER")[0],
            dtype=float,
        )


def _unit(vector: np.ndarray) -> tuple[np.ndarray, float]:
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        raise FloatingPointError("Encountered a zero-length dynamical vector.")
    return vector / norm, norm


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

        sun_state = body_states["SUN"]
        r_vec = position - sun_state[:3]
        v_vec = velocity - sun_state[3:]
        r_hat, r = _unit(r_vec)
        h_vec = np.cross(r_vec, v_vec)
        h_hat, _ = _unit(h_vec)
        t_hat = np.cross(h_hat, r_hat)

        if model.relativity_1pn:
            v2 = float(np.dot(v_vec, v_vec))
            rv = float(np.dot(r_vec, v_vec))
            acceleration += (
                mu_sun
                / (C_KM_S**2 * r**3)
                * ((4.0 * mu_sun / r - v2) * r_vec + 4.0 * rv * v_vec)
            )

        if model.area_mass_m2_kg > 0:
            srp = (
                SOLAR_PRESSURE_N_M2
                * model.radiation_coefficient
                * model.area_mass_m2_kg
                / 1000.0
                * (AU_KM / r) ** 2
            )
            acceleration += srp * r_hat
            if model.poynting_robertson:
                radial_velocity = float(np.dot(v_vec, r_hat))
                drag = -srp * (radial_velocity * r_hat + v_vec) / C_KM_S
                acceleration += drag
                if model.solar_wind_drag:
                    acceleration += model.solar_wind_to_pr * drag

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
            rtol=min(rtol, 1e-18),
            atol_position_km=min(atol_position_km, 1e-9),
            atol_velocity_kms=min(atol_velocity_kms, 1e-15),
            max_step_days=2.0,
        )
        kernel_description = (
            f"Fortran real{integrator.precision_digits} arithmetic + "
            "JPL DE440s/SB441-N16 binary64 SPK"
        )
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
