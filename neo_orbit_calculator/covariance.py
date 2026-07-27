"""JPL covariance ingestion and virtual-asteroid ensemble propagation."""

from __future__ import annotations

import csv
import json
import urllib.parse
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import spiceypy as spice
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar

from .core import AU_KM, DAY_S, ForceModel, PlanetaryEnvironment
from .fortran_backend import FortranIntegrator

SBDB_API = "https://ssd-api.jpl.nasa.gov/sbdb.api"
EARTH_SCREENING_RADIUS_KM = 6478.1363
SUPPORTED_COVARIANCE_LABELS = {
    "e",
    "q",
    "tp",
    "node",
    "peri",
    "i",
    "A1",
    "A2",
    "A3",
    "DT",
}


@dataclass(frozen=True)
class CovarianceSolution:
    designation: str
    fullname: str
    orbit_id: str
    epoch_jd_tdb: float
    labels: tuple[str, ...]
    mean: np.ndarray
    covariance: np.ndarray
    model_parameters: dict[str, float]
    source: str = "NASA/JPL Small-Body Database"


@dataclass
class VirtualAsteroidResult:
    solution: CovarianceSolution
    samples: np.ndarray
    output_jd_tdb: np.ndarray
    final_states_km_kms: np.ndarray
    closest_approach_jd_tdb: np.ndarray
    closest_approach_km: np.ndarray
    nominal_closest_approach_jd_tdb: float
    nominal_closest_approach_km: float
    function_evaluations: np.ndarray
    seed: int
    kernel: str

    @property
    def screening_impact_count(self) -> int:
        return int(
            np.count_nonzero(
                self.closest_approach_km <= EARTH_SCREENING_RADIUS_KM
            )
        )

    @property
    def screening_impact_fraction(self) -> float:
        return self.screening_impact_count / len(self.samples)


def _get_json(url: str) -> dict:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "CODES-Covariance/1.0"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if "error" in payload:
        raise RuntimeError(payload["error"])
    return payload


def fetch_sbdb_covariance(designation: str) -> CovarianceSolution:
    """Fetch the full JPL solution-epoch covariance matrix."""
    parameters = {
        "sstr": designation.strip(),
        "cov": "mat",
        "full-prec": "true",
    }
    payload = _get_json(
        SBDB_API + "?" + urllib.parse.urlencode(parameters)
    )
    orbit = payload["orbit"]
    covariance_data = orbit.get("covariance")
    if covariance_data is None:
        raise RuntimeError(
            f"JPL SBDB provides no covariance for {designation}."
        )
    labels = tuple(str(label) for label in covariance_data["labels"])
    unsupported = set(labels) - SUPPORTED_COVARIANCE_LABELS
    if unsupported:
        raise NotImplementedError(
            "Unsupported estimated covariance parameters: "
            + ", ".join(sorted(unsupported))
        )

    element_rows = covariance_data.get("elements")
    if element_rows is None:
        if float(orbit["epoch"]) != float(covariance_data["epoch"]):
            raise RuntimeError(
                "SBDB omitted solution-epoch elements for a different epoch."
            )
        element_rows = orbit["elements"]
    values: dict[str, float] = {}
    for row in element_rows:
        values[str(row["label"])] = float(row["value"])
        values[str(row["name"])] = float(row["value"])
    model_parameters = {
        str(row["name"]): float(row["value"])
        for row in orbit.get("model_pars", [])
        if row.get("value") is not None
    }
    values.update(model_parameters)
    try:
        mean = np.asarray([values[label] for label in labels], dtype=float)
    except KeyError as exc:
        raise RuntimeError(
            f"SBDB covariance label {exc.args[0]!r} has no nominal value."
        ) from exc

    covariance = np.asarray(covariance_data["data"], dtype=float)
    if covariance.shape != (len(labels), len(labels)):
        raise RuntimeError("SBDB covariance matrix has inconsistent shape.")
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    tolerance = max(1.0, float(np.max(np.abs(eigenvalues)))) * 1.0e-14
    if float(np.min(eigenvalues)) < -tolerance:
        raise RuntimeError("SBDB covariance matrix is not positive semidefinite.")
    covariance = (
        eigenvectors
        @ np.diag(np.clip(eigenvalues, 0.0, None))
        @ eigenvectors.T
    )
    return CovarianceSolution(
        designation=designation,
        fullname=str(payload["object"]["fullname"]),
        orbit_id=str(orbit["orbit_id"]),
        epoch_jd_tdb=float(covariance_data["epoch"]),
        labels=labels,
        mean=mean,
        covariance=covariance,
        model_parameters=model_parameters,
    )


def sample_virtual_asteroids(
    solution: CovarianceSolution,
    clones: int,
    seed: int,
) -> np.ndarray:
    """Draw a reproducible correlated Gaussian ensemble."""
    if clones < 1:
        raise ValueError("clones must be positive.")
    rng = np.random.default_rng(seed)
    samples = rng.multivariate_normal(
        solution.mean,
        solution.covariance,
        size=clones,
        method="eigh",
    )
    index = {label: position for position, label in enumerate(solution.labels)}
    if np.any(samples[:, index["e"]] <= 0.0):
        raise RuntimeError("A virtual asteroid has non-positive eccentricity.")
    if np.any(samples[:, index["q"]] <= 0.0):
        raise RuntimeError("A virtual asteroid has non-positive perihelion.")
    return samples


def elements_to_barycentric_state(
    sample: np.ndarray,
    labels: tuple[str, ...],
    epoch_jd_tdb: float,
    environment: PlanetaryEnvironment,
) -> np.ndarray:
    """Convert one SBDB covariance sample to a J2000 barycentric state."""
    values = dict(zip(labels, np.asarray(sample, dtype=float), strict=True))
    eccentricity = values["e"]
    perihelion_km = values["q"] * AU_KM
    semimajor_km = perihelion_km / (1.0 - eccentricity)
    mean_motion = np.sqrt(environment.gm["SUN"] / semimajor_km**3)
    mean_anomaly = (
        mean_motion
        * (epoch_jd_tdb - values["tp"])
        * DAY_S
    ) % (2.0 * np.pi)
    et = environment.jd_to_et(epoch_jd_tdb)
    conic_elements = np.asarray(
        [
            perihelion_km,
            eccentricity,
            np.deg2rad(values["i"]),
            np.deg2rad(values["node"]),
            np.deg2rad(values["peri"]),
            mean_anomaly,
            et,
            environment.gm["SUN"],
        ]
    )
    heliocentric_ecliptic = np.asarray(
        spice.conics(conic_elements, et),
        dtype=float,
    )
    rotation = np.asarray(
        spice.pxform("ECLIPJ2000", "J2000", et),
        dtype=float,
    )
    heliocentric_j2000 = np.concatenate(
        (
            rotation @ heliocentric_ecliptic[:3],
            rotation @ heliocentric_ecliptic[3:],
        )
    )
    return heliocentric_j2000 + environment.state("SUN", et)


def _model_for_sample(
    base_model: ForceModel,
    sample: np.ndarray,
    labels: tuple[str, ...],
    fixed_parameters: dict[str, float],
) -> ForceModel:
    values = dict(fixed_parameters)
    values.update(
        dict(zip(labels, np.asarray(sample, dtype=float), strict=True))
    )
    law = (
        "marsden"
        if values.get("NK", base_model.outgassing_k) > 0.0
        else "inverse_square"
    )
    return replace(
        base_model,
        a1_au_day2=values.get("A1", base_model.a1_au_day2),
        a2_au_day2=values.get("A2", base_model.a2_au_day2),
        a3_au_day2=values.get("A3", base_model.a3_au_day2),
        nongrav_law=law,
        outgassing_r0_au=values.get("R0", base_model.outgassing_r0_au),
        outgassing_m=values.get("NM", base_model.outgassing_m),
        outgassing_n=values.get("NN", base_model.outgassing_n),
        outgassing_k=values.get("NK", base_model.outgassing_k),
        outgassing_alpha=values.get(
            "ALN",
            base_model.outgassing_alpha,
        ),
        outgassing_lag_days=values.get(
            "DT",
            base_model.outgassing_lag_days,
        ),
    )


def _refined_earth_approach(
    jd_tdb: np.ndarray,
    states: np.ndarray,
    earth_positions: np.ndarray,
    environment: PlanetaryEnvironment,
) -> tuple[float, float]:
    distance = np.linalg.norm(states[:, :3] - earth_positions, axis=1)
    index = int(np.argmin(distance))
    if index < 2 or index > len(jd_tdb) - 3:
        return float(jd_tdb[index]), float(distance[index])
    lower = max(0, index - 3)
    upper = min(len(jd_tdb), index + 4)
    splines = [
        CubicSpline(jd_tdb[lower:upper], states[lower:upper, component])
        for component in range(3)
    ]

    def separation(epoch: float) -> float:
        asteroid = np.asarray([spline(epoch) for spline in splines])
        earth = environment.state(
            "EARTH",
            environment.jd_to_et(epoch),
        )[:3]
        return float(np.linalg.norm(asteroid - earth))

    solution = minimize_scalar(
        separation,
        bounds=(jd_tdb[index - 2], jd_tdb[index + 2]),
        method="bounded",
        options={"xatol": 1.0e-10},
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    return float(solution.x), float(solution.fun)


def _dense_encounter_grid(
    coarse_seconds: np.ndarray,
    closest_index: int,
) -> tuple[np.ndarray, float, float]:
    half_window = min(
        5.0 * DAY_S,
        0.25 * float(coarse_seconds[-1]),
    )
    lower = max(
        float(coarse_seconds[0]),
        float(coarse_seconds[closest_index]) - half_window,
    )
    upper = min(
        float(coarse_seconds[-1]),
        float(coarse_seconds[closest_index]) + half_window,
    )
    count = max(
        5,
        int(np.ceil((upper - lower) / 600.0)) + 1,
    )
    dense = np.linspace(lower, upper, count)
    return np.unique(np.concatenate((coarse_seconds, dense))), lower, upper


def propagate_virtual_asteroids(
    solution: CovarianceSolution,
    stop_jd_tdb: float,
    clones: int = 100,
    samples: int = 1001,
    seed: int = 42,
    kernel_dir: Path | str = Path("kernels"),
    base_model: ForceModel | None = None,
    include_large_asteroids: bool = True,
    include_jupiter_system: bool = False,
    ephemeris: str = "auto",
) -> VirtualAsteroidResult:
    """Propagate a complete correlated virtual-asteroid ensemble."""
    if stop_jd_tdb <= solution.epoch_jd_tdb:
        raise ValueError("stop_jd_tdb must follow the covariance epoch.")
    if samples < 5:
        raise ValueError("samples must be at least 5.")
    ensemble = sample_virtual_asteroids(solution, clones, seed)
    environment = PlanetaryEnvironment(
        Path(kernel_dir),
        start_jd_tdb=solution.epoch_jd_tdb,
        stop_jd_tdb=stop_jd_tdb,
        ephemeris=ephemeris,
        include_large_asteroids=include_large_asteroids,
        include_jupiter_system=include_jupiter_system,
    )
    environment.configure_jupiter_system(
        environment.jd_to_et(solution.epoch_jd_tdb),
        environment.jd_to_et(stop_jd_tdb),
    )
    integrator = FortranIntegrator(environment)
    coarse_seconds = np.linspace(
        0.0,
        (stop_jd_tdb - solution.epoch_jd_tdb) * DAY_S,
        samples,
    )
    base_model = base_model or ForceModel()
    nominal_state = elements_to_barycentric_state(
        solution.mean,
        solution.labels,
        solution.epoch_jd_tdb,
        environment,
    )
    nominal_model = _model_for_sample(
        base_model,
        solution.mean,
        solution.labels,
        solution.model_parameters,
    )
    nominal_coarse, _ = integrator.propagate(
        nominal_state,
        environment.jd_to_et(solution.epoch_jd_tdb),
        coarse_seconds,
        nominal_model,
        rtol=3.0e-14,
        atol_position_km=1.0e-6,
        atol_velocity_kms=1.0e-12,
        max_step_days=2.0,
    )
    coarse_jd = solution.epoch_jd_tdb + coarse_seconds / DAY_S
    coarse_earth = np.asarray(
        [
            environment.state("EARTH", environment.jd_to_et(epoch))[:3]
            for epoch in coarse_jd
        ]
    )
    closest_index = int(
        np.argmin(
            np.linalg.norm(
                nominal_coarse[:, :3] - coarse_earth,
                axis=1,
            )
        )
    )
    output_seconds, dense_lower, dense_upper = _dense_encounter_grid(
        coarse_seconds,
        closest_index,
    )
    output_jd = solution.epoch_jd_tdb + output_seconds / DAY_S
    earth_positions = np.asarray(
        [
            environment.state("EARTH", environment.jd_to_et(epoch))[:3]
            for epoch in output_jd
        ]
    )
    nominal_states, _ = integrator.propagate(
        nominal_state,
        environment.jd_to_et(solution.epoch_jd_tdb),
        output_seconds,
        nominal_model,
        rtol=3.0e-14,
        atol_position_km=1.0e-6,
        atol_velocity_kms=1.0e-12,
        max_step_days=2.0,
    )
    nominal_approach_jd, nominal_approach_km = _refined_earth_approach(
        output_jd,
        nominal_states,
        earth_positions,
        environment,
    )
    final_states = np.empty((clones, 6))
    approach_jd = np.empty(clones)
    approach_km = np.empty(clones)
    evaluations = np.empty(clones, dtype=int)
    coarse_output_index = np.searchsorted(output_seconds, coarse_seconds)
    for clone, sample in enumerate(ensemble):
        initial_state = elements_to_barycentric_state(
            sample,
            solution.labels,
            solution.epoch_jd_tdb,
            environment,
        )
        model = _model_for_sample(
            base_model,
            sample,
            solution.labels,
            solution.model_parameters,
        )
        states, evaluations[clone] = integrator.propagate(
            initial_state,
            environment.jd_to_et(solution.epoch_jd_tdb),
            output_seconds,
            model,
            rtol=3.0e-14,
            atol_position_km=1.0e-6,
            atol_velocity_kms=1.0e-12,
            max_step_days=2.0,
        )
        final_states[clone] = states[-1]
        clone_coarse_distance = np.linalg.norm(
            states[coarse_output_index, :3] - coarse_earth,
            axis=1,
        )
        clone_closest_index = int(np.argmin(clone_coarse_distance))
        clone_closest_seconds = coarse_seconds[clone_closest_index]
        clone_output_jd = output_jd
        clone_earth_positions = earth_positions
        if not dense_lower <= clone_closest_seconds <= dense_upper:
            clone_output_seconds, _, _ = _dense_encounter_grid(
                coarse_seconds,
                clone_closest_index,
            )
            states, extra_evaluations = integrator.propagate(
                initial_state,
                environment.jd_to_et(solution.epoch_jd_tdb),
                clone_output_seconds,
                model,
                rtol=3.0e-14,
                atol_position_km=1.0e-6,
                atol_velocity_kms=1.0e-12,
                max_step_days=2.0,
            )
            evaluations[clone] += extra_evaluations
            final_states[clone] = states[-1]
            clone_output_jd = (
                solution.epoch_jd_tdb + clone_output_seconds / DAY_S
            )
            clone_earth_positions = np.asarray(
                [
                    environment.state(
                        "EARTH",
                        environment.jd_to_et(epoch),
                    )[:3]
                    for epoch in clone_output_jd
                ]
            )
        approach_jd[clone], approach_km[clone] = _refined_earth_approach(
            clone_output_jd,
            states,
            clone_earth_positions,
            environment,
        )
    return VirtualAsteroidResult(
        solution=solution,
        samples=ensemble,
        output_jd_tdb=output_jd,
        final_states_km_kms=final_states,
        closest_approach_jd_tdb=approach_jd,
        closest_approach_km=approach_km,
        nominal_closest_approach_jd_tdb=nominal_approach_jd,
        nominal_closest_approach_km=nominal_approach_km,
        function_evaluations=evaluations,
        seed=seed,
        kernel=(
            f"Fortran real{integrator.precision_digits} + "
            f"{environment.ephemeris_description}"
            + (
                " + SB441-N16"
                if environment.include_large_asteroids
                else ""
            )
            + (
                " + JUP365 resolved Jupiter system "
                f"({', '.join(environment.active_jupiter_system)})"
                if environment.jupiter_system_enabled
                else ""
            )
            + " + full multi-body 1PN"
        ),
    )


def write_virtual_asteroid_products(
    result: VirtualAsteroidResult,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    """Write clone-level CSV, summary JSON, and a diagnostic figure."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = result.solution.designation.replace(" ", "_")
    csv_path = output_dir / f"{stem}_virtual_asteroids.csv"
    json_path = output_dir / f"{stem}_virtual_asteroids_summary.json"
    figure_path = output_dir / f"{stem}_virtual_asteroids.png"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = [
            "clone",
            *result.solution.labels,
            "closest_approach_jd_tdb",
            "closest_approach_km",
            "function_evaluations",
        ]
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        for clone, sample in enumerate(result.samples):
            row = {
                "clone": clone,
                **dict(
                    zip(
                        result.solution.labels,
                        sample,
                        strict=True,
                    )
                ),
                "closest_approach_jd_tdb": (
                    result.closest_approach_jd_tdb[clone]
                ),
                "closest_approach_km": (
                    result.closest_approach_km[clone]
                ),
                "function_evaluations": (
                    result.function_evaluations[clone]
                ),
            }
            writer.writerow(row)

    summary = {
        "designation": result.solution.designation,
        "fullname": result.solution.fullname,
        "orbit_id": result.solution.orbit_id,
        "covariance_epoch_jd_tdb": result.solution.epoch_jd_tdb,
        "covariance_labels": result.solution.labels,
        "clones": len(result.samples),
        "seed": result.seed,
        "stop_jd_tdb": float(result.output_jd_tdb[-1]),
        "minimum_earth_distance_km": float(
            np.min(result.closest_approach_km)
        ),
        "nominal_closest_approach_jd_tdb": (
            result.nominal_closest_approach_jd_tdb
        ),
        "nominal_closest_approach_km": (
            result.nominal_closest_approach_km
        ),
        "median_earth_distance_km": float(
            np.median(result.closest_approach_km)
        ),
        "screening_radius_km": EARTH_SCREENING_RADIUS_KM,
        "screening_impact_count": result.screening_impact_count,
        "screening_impact_fraction": result.screening_impact_fraction,
        "kernel": result.kernel,
        "adaptive_step": (
            "embedded error control plus 5% gravity/crossing-time limiter"
        ),
        "warning": (
            "Monte Carlo screening is not a Sentry-equivalent impact "
            "probability. Rare-event completeness requires targeted line-of-"
            "variation sampling and encounter-plane analysis."
        ),
    }
    json_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    epoch_offset_days = (
        result.closest_approach_jd_tdb
        - np.median(result.closest_approach_jd_tdb)
    )
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))
    axes[0].hist(
        result.closest_approach_km / 1000.0,
        bins=min(30, max(8, len(result.samples) // 4)),
        color="#245AA6",
        edgecolor="white",
    )
    axes[0].axvline(
        EARTH_SCREENING_RADIUS_KM / 1000.0,
        color="#A7354D",
        ls="--",
        label="Earth + 100 km",
    )
    axes[0].axvline(
        result.nominal_closest_approach_km / 1000.0,
        color="#E3A008",
        lw=1.8,
        label="nominal orbit",
    )
    axes[0].set_xlabel("minimum geocentric distance [$10^3$ km]")
    axes[0].set_ylabel("virtual asteroids")
    axes[0].legend(frameon=False)
    axes[1].scatter(
        epoch_offset_days * 24.0,
        result.closest_approach_km / 1000.0,
        s=20,
        color="#007C77",
        alpha=0.75,
    )
    axes[1].scatter(
        (
            result.nominal_closest_approach_jd_tdb
            - np.median(result.closest_approach_jd_tdb)
        )
        * 24.0,
        result.nominal_closest_approach_km / 1000.0,
        marker="D",
        s=58,
        color="#E3A008",
        edgecolor="#151B23",
        label="nominal orbit",
        zorder=4,
    )
    axes[1].set_xlabel("encounter time relative to median [h]")
    axes[1].set_ylabel("minimum geocentric distance [$10^3$ km]")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(alpha=0.22)
    figure.suptitle(
        f"{result.solution.fullname}: correlated virtual asteroids"
    )
    figure.tight_layout()
    figure.savefig(figure_path, dpi=220, facecolor="white")
    plt.close(figure)
    return csv_path, json_path, figure_path
