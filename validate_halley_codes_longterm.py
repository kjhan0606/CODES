"""Validate a 150-year CODES integration of 1P/Halley against JPL Horizons."""

from __future__ import annotations

import csv
import json
import time
import urllib.error
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import spiceypy as spice
from astropy.time import Time
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar

from neo_orbit_calculator.core import (
    AU_KM,
    DE440Environment,
    ForceModel,
    propagate_custom,
)
from neo_orbit_calculator.jpl import horizons_elements, horizons_vectors

ROOT = Path(__file__).resolve().parent
KERNEL_DIR = ROOT / "neo_orbit_calculator" / "kernels"
FIGURE_PATH = ROOT / "appendixC_assets" / "halley_codes_longterm_validation.png"
CSV_PATH = ROOT / "halley_codes_longterm_validation.csv"
PROVENANCE_PATH = ROOT / "halley_codes_longterm_provenance.json"
CACHE_PATH = (
    ROOT
    / "neo_orbit_calculator"
    / "validation_data"
    / "halley_codes_1850_2000_full1pn.npz"
)

HALLEY_RECORD = "90000030"
START_TDB = "1850-01-02"
STOP_TDB = "2000-01-01"
A1_AU_DAY2 = 4.887055233121e-10
A2_AU_DAY2 = 1.554720290005e-10

# Seed epochs identify the two perihelia inside the DE440s-supported interval.
OFFICIAL_RETURNS = (
    (1910, "90000029", 2418781.6785),
    (1986, "90000030", 2446469.9736161465),
)


def _heliocentric_distance(
    jd_tdb: np.ndarray,
    states: np.ndarray,
    environment: DE440Environment,
) -> np.ndarray:
    sun = np.asarray(
        [
            environment.state("SUN", environment.jd_to_et(epoch))[:3]
            for epoch in jd_tdb
        ]
    )
    return np.linalg.norm(states[:, :3] - sun, axis=1)


def _osculating_elements(
    state: np.ndarray,
    jd_tdb: float,
    environment: DE440Environment,
) -> tuple[float, float]:
    et = environment.jd_to_et(jd_tdb)
    heliocentric = state - environment.state("SUN", et)
    rotation = np.asarray(
        spice.pxform("J2000", "ECLIPJ2000", et),
        dtype=float,
    )
    ecliptic_state = np.concatenate(
        (
            rotation @ heliocentric[:3],
            rotation @ heliocentric[3:],
        )
    )
    elements = spice.oscelt(
        ecliptic_state,
        et,
        environment.gm["SUN"],
    )
    perihelion_km = float(elements[0])
    eccentricity = float(elements[1])
    return (
        perihelion_km / (1.0 - eccentricity) / AU_KM,
        perihelion_km / AU_KM,
    )


def _period_years(semimajor_axis_au: np.ndarray | float, mu_sun: float):
    semimajor_axis_km = np.asarray(semimajor_axis_au) * AU_KM
    period_seconds = (
        2.0 * np.pi * np.sqrt(semimajor_axis_km**3 / mu_sun)
    )
    result = period_seconds / (365.25 * 86400.0)
    if np.ndim(result) == 0:
        return float(result)
    return result


def _refine_codes_perihelion(
    jd_tdb: np.ndarray,
    states: np.ndarray,
    distance_km: np.ndarray,
    seed_jd: float,
    environment: DE440Environment,
) -> tuple[float, np.ndarray, float]:
    candidate = np.flatnonzero(
        (jd_tdb >= seed_jd - 20.0) & (jd_tdb <= seed_jd + 20.0)
    )
    index = int(candidate[np.argmin(distance_km[candidate])])
    lower = max(0, index - 6)
    upper = min(len(jd_tdb), index + 7)
    splines = [
        CubicSpline(jd_tdb[lower:upper], states[lower:upper, component])
        for component in range(6)
    ]

    def distance(epoch: float) -> float:
        state = np.asarray([spline(epoch) for spline in splines])
        sun = environment.state(
            "SUN",
            environment.jd_to_et(epoch),
        )[:3]
        return float(np.linalg.norm(state[:3] - sun))

    solution = minimize_scalar(
        distance,
        bounds=(jd_tdb[index - 2], jd_tdb[index + 2]),
        method="bounded",
        options={"xatol": 1e-10},
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    state = np.asarray([spline(solution.x) for spline in splines])
    return float(solution.x), state, float(solution.fun)


def _official_perihelion(
    record: str,
    seed_jd: float,
) -> tuple[float, float, float]:
    epochs = np.linspace(seed_jd - 5.0, seed_jd + 5.0, 41)
    for attempt in range(4):
        try:
            vectors, _ = horizons_vectors(
                record,
                epochs,
                center="500@10",
            )
            break
        except urllib.error.HTTPError:
            if attempt == 3:
                raise
            time.sleep(2.0**attempt)
    splines = [
        CubicSpline(vectors[:, 0], vectors[:, component])
        for component in range(1, 7)
    ]

    def distance(epoch: float) -> float:
        return float(
            np.linalg.norm(
                [spline(epoch) for spline in splines[:3]]
            )
        )

    coarse = int(np.argmin(np.linalg.norm(vectors[:, 1:4], axis=1)))
    solution = minimize_scalar(
        distance,
        bounds=(epochs[coarse - 2], epochs[coarse + 2]),
        method="bounded",
        options={"xatol": 1e-10},
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    for attempt in range(4):
        try:
            elements, _ = horizons_elements(
                record,
                [float(solution.x)],
            )
            break
        except urllib.error.HTTPError:
            if attempt == 3:
                raise
            time.sleep(2.0**attempt)
    return (
        float(solution.x),
        float(elements[0, 7]),
        float(elements[0, 2]),
    )


def run_validation() -> tuple[
    list[dict[str, float | int | str]],
    np.ndarray,
    np.ndarray,
    str,
    int,
]:
    start_jd = float(Time(START_TDB, scale="tdb").jd)
    stop_jd = float(Time(STOP_TDB, scale="tdb").jd)
    samples = int(np.ceil((stop_jd - start_jd) / 2.0)) + 1
    model = ForceModel(
        relativity_1pn=True,
        full_multibody_1pn=True,
        a1_au_day2=A1_AU_DAY2,
        a2_au_day2=A2_AU_DAY2,
        a3_au_day2=0.0,
        nongrav_law="marsden",
        outgassing_lag_days=0.0,
    )
    if CACHE_PATH.exists():
        cache = np.load(CACHE_PATH)
        result_jd = np.asarray(cache["jd"], dtype=float)
        result_state = np.asarray(cache["state"], dtype=float)
        function_evaluations = int(cache["nfev"])
        kernel = str(cache["kernel"])
    else:
        result = propagate_custom(
            HALLEY_RECORD,
            start_jd,
            stop_jd,
            samples=samples,
            model=model,
            kernel_dir=KERNEL_DIR,
            validate_horizons=False,
            include_large_asteroids=True,
            backend="fortran",
        )
        result_jd = result.jd_tdb
        result_state = result.state_km_kms
        function_evaluations = result.function_evaluations
        kernel = result.kernel
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            CACHE_PATH,
            jd=result_jd,
            state=result_state,
            nfev=function_evaluations,
            kernel=kernel,
        )
    if "full multi-body 1PN" not in kernel:
        kernel += " + full multi-body 1PN"
    environment = DE440Environment(
        KERNEL_DIR,
        include_large_asteroids=True,
    )
    distance_km = _heliocentric_distance(
        result_jd,
        result_state,
        environment,
    )
    rows: list[dict[str, float | int | str]] = []
    for year, record, seed_jd in OFFICIAL_RETURNS:
        codes_jd, codes_state, codes_distance = _refine_codes_perihelion(
            result_jd,
            result_state,
            distance_km,
            seed_jd,
            environment,
        )
        codes_a, codes_q = _osculating_elements(
            codes_state,
            codes_jd,
            environment,
        )
        official_jd, official_a, official_q = _official_perihelion(
            record,
            seed_jd,
        )
        rows.append(
            {
                "return_year": year,
                "jpl_record": record,
                "codes_perihelion_jd_tdb": codes_jd,
                "jpl_perihelion_jd_tdb": official_jd,
                "delta_t_s": (codes_jd - official_jd) * 86400.0,
                "codes_a_au": codes_a,
                "jpl_a_au": official_a,
                "delta_a_km": (codes_a - official_a) * AU_KM,
                "codes_period_years": _period_years(
                    codes_a,
                    environment.gm["SUN"],
                ),
                "jpl_period_years": _period_years(
                    official_a,
                    environment.gm["SUN"],
                ),
                "delta_period_days": (
                    _period_years(codes_a, environment.gm["SUN"])
                    - _period_years(official_a, environment.gm["SUN"])
                )
                * 365.25,
                "codes_q_au": codes_q,
                "jpl_q_au": official_q,
                "delta_q_km": (codes_q - official_q) * AU_KM,
                "codes_distance_at_perihelion_au": (
                    codes_distance / AU_KM
                ),
            }
        )
    return (
        rows,
        result_jd,
        result_state,
        kernel,
        function_evaluations,
    )


def write_outputs(
    rows: list[dict[str, float | int | str]],
    jd_tdb: np.ndarray,
    states: np.ndarray,
    kernel: str,
    function_evaluations: int,
) -> None:
    with CSV_PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    years = np.asarray([int(row["return_year"]) for row in rows])
    codes_jd = np.asarray(
        [float(row["codes_perihelion_jd_tdb"]) for row in rows]
    )
    official_jd = np.asarray(
        [float(row["jpl_perihelion_jd_tdb"]) for row in rows]
    )
    codes_a = np.asarray([float(row["codes_a_au"]) for row in rows])
    official_a = np.asarray([float(row["jpl_a_au"]) for row in rows])
    codes_period = np.asarray(
        [float(row["codes_period_years"]) for row in rows]
    )
    official_period = np.asarray(
        [float(row["jpl_period_years"]) for row in rows]
    )
    codes_q = np.asarray([float(row["codes_q_au"]) for row in rows])
    official_q = np.asarray([float(row["jpl_q_au"]) for row in rows])

    stride = 10
    timeline_jd = jd_tdb[::stride]
    timeline_year = Time(
        timeline_jd,
        format="jd",
        scale="tdb",
    ).decimalyear
    environment = DE440Environment(
        KERNEL_DIR,
        include_large_asteroids=True,
    )
    timeline_elements = np.asarray(
        [
            _osculating_elements(state, epoch, environment)
            for state, epoch in zip(
                states[::stride],
                timeline_jd,
                strict=True,
            )
        ]
    )
    timeline_a = timeline_elements[:, 0]
    timeline_q = timeline_elements[:, 1]
    timeline_period = _period_years(
        timeline_a,
        environment.gm["SUN"],
    )
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(11.2, 9.4),
        sharex=True,
    )
    axes[0].plot(
        timeline_year,
        timeline_period,
        color="#007C77",
        lw=1.5,
        label="CODES forward integration from one 1850 state",
    )
    axes[0].scatter(
        years,
        codes_period,
        marker="o",
        s=115,
        facecolor="none",
        edgecolor="#B55220",
        linewidth=2.0,
        zorder=6,
        label="CODES perihelia",
    )
    axes[0].scatter(
        years,
        official_period,
        marker="D",
        s=58,
        facecolor="#FFD166",
        edgecolor="#151B23",
        linewidth=1.1,
        zorder=7,
        label="NASA/JPL Horizons",
    )
    axes[0].set_ylabel("osculating period [yr]")
    axes[0].legend(frameon=False, ncol=3, fontsize=8.5)

    axes[1].plot(
        timeline_year,
        timeline_a,
        color="#007C77",
        lw=1.5,
        label="CODES trajectory",
    )
    axes[1].scatter(
        years,
        codes_a,
        marker="o",
        s=115,
        facecolor="none",
        edgecolor="#B55220",
        linewidth=2.0,
        zorder=6,
        label="CODES perihelia",
    )
    axes[1].scatter(
        years,
        official_a,
        marker="D",
        s=58,
        facecolor="#FFD166",
        edgecolor="#151B23",
        linewidth=1.1,
        zorder=6,
        label="NASA/JPL Horizons",
    )
    axes[1].set_ylabel("osculating semimajor axis [au]")
    axes[1].legend(frameon=False)

    axes[2].plot(
        timeline_year,
        timeline_q,
        color="#B55220",
        lw=1.5,
        label="CODES trajectory",
    )
    axes[2].scatter(
        years,
        codes_q,
        marker="o",
        s=115,
        facecolor="none",
        edgecolor="#007C77",
        linewidth=2.0,
        zorder=6,
        label="CODES perihelia",
    )
    axes[2].scatter(
        years,
        official_q,
        marker="D",
        s=58,
        facecolor="#FFD166",
        edgecolor="#151B23",
        linewidth=1.1,
        zorder=6,
        label="NASA/JPL Horizons",
    )
    axes[2].set_ylabel("osculating perihelion distance [au]")
    axes[2].set_xlabel("year [TDB]")
    axes[2].legend(frameon=False)

    for index, year in enumerate(years):
        axes[0].annotate(
            (
                "$\\Delta P$="
                f"{float(rows[index]['delta_period_days']):+.2f} d"
            ),
            (year, codes_period[index]),
            xytext=(8, 10 if index == 0 else -17),
            textcoords="offset points",
            color="#5C6875",
            fontsize=8.5,
        )
        axes[1].annotate(
            f"$\\Delta a$={float(rows[index]['delta_a_km']):+.0f} km",
            (year, codes_a[index]),
            xytext=(8, 10 if index == 0 else -17),
            textcoords="offset points",
            color="#5C6875",
            fontsize=8.5,
        )
        axes[2].annotate(
            f"$\\Delta q$={float(rows[index]['delta_q_km']):+.0f} km",
            (year, codes_q[index]),
            xytext=(8, 10 if index == 0 else -17),
            textcoords="offset points",
            color="#5C6875",
            fontsize=8.5,
        )
    for axis in axes:
        axis.grid(alpha=0.22)
    figure.suptitle(
        "Independent 150-year CODES propagation of 1P/Halley",
        fontsize=16,
    )
    figure.text(
        0.5,
        0.012,
        (
            "Initial state: JPL #75 at 1850-01-02 TDB. "
            "No comparison state is ingested after the start epoch."
        ),
        ha="center",
        color="#5C6875",
        fontsize=9.5,
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.97))
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=220, facecolor="white")
    plt.close(figure)

    provenance = {
        "generated_utc": "2026-07-26",
        "purpose": "Independent long-term propagation consistency test",
        "initial_state_source": (
            "NASA/JPL Horizons record 90000030 at 1850-01-02 TDB"
        ),
        "comparison_source": (
            "NASA/JPL Horizons records 90000029 and 90000030"
        ),
        "integration_interval_tdb": [START_TDB, STOP_TDB],
        "force_model": {
            "A1_au_day2": A1_AU_DAY2,
            "A2_au_day2": A2_AU_DAY2,
            "nongrav_law": "Marsden standard water-ice law",
            "outgassing_lag_days": 0.0,
            "relativity": (
                "full massless-target Einstein-Infeld-Hoffmann 1PN"
            ),
            "major_bodies": "DE440s",
            "large_asteroids": "SB441-N16",
        },
        "kernel": kernel,
        "adaptive_step": (
            "embedded error control plus 5% gravity/crossing-time limiter"
        ),
        "function_evaluations": function_evaluations,
        "interpretation": (
            "This validates forward propagation from one fitted initial "
            "state. It is not a blind orbit fit because JPL solution #75 "
            "uses observations through 1994."
        ),
    }
    PROVENANCE_PATH.write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    rows, jd_tdb, states, kernel, evaluations = run_validation()
    write_outputs(rows, jd_tdb, states, kernel, evaluations)
    print(json.dumps(rows, indent=2))
    print(FIGURE_PATH)
    print(CSV_PATH)
    print(PROVENANCE_PATH)


if __name__ == "__main__":
    main()
