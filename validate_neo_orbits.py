"""Reproduce ten historical Earth close approaches with the local propagator.

This is a propagation validation, not an independent orbit determination.
MPC ADES observations establish the real observing arcs. The initial state and
the comparison solution are the current JPL Horizons orbit solution.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import spiceypy as spice
from astropy.time import Time
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar

from neo_orbit_calculator.core import (
    AU_KM,
    DAY_S,
    DE440Environment,
    ForceModel,
    propagate_custom,
)
from neo_orbit_calculator.jpl import horizons_elements, jpl_close_approaches
from neo_orbit_calculator.mpc import fetch_mpc_ades, summarize_mpc_ades

CASES = (
    ("2019 OK", "2019-07-20", "2019-07-30"),
    ("2019 UN13", "2019-10-27", "2019-11-04"),
    ("2020 QG", "2020-08-12", "2020-08-20"),
    ("2020 VT4", "2020-11-10", "2020-11-18"),
    ("2021 GW4", "2021-04-08", "2021-04-16"),
    ("2021 UA1", "2021-10-21", "2021-10-29"),
    ("2022 NF", "2022-07-03", "2022-07-11"),
    ("2023 BU", "2023-01-23", "2023-01-31"),
    ("2024 MK", "2024-06-25", "2024-07-03"),
    ("2024 ON", "2024-09-13", "2024-09-21"),
)

ROOT = Path(__file__).resolve().parent
KERNEL_DIR = ROOT / "neo_orbit_calculator" / "kernels"
CACHE_DIR = ROOT / "neo_orbit_calculator" / "validation_data" / "mpc"
CSV_PATH = ROOT / "neo_orbit_validation_10cases.csv"
PROVENANCE_PATH = ROOT / "neo_orbit_validation_provenance.json"
FIGURE_PATH = ROOT / "appendixC_assets" / "neo_orbit_validation_10cases.png"
TEX_PATH = ROOT / "neo_orbit_validation_tables.tex"


def _wrapped_angle_difference_deg(value: float, reference: float) -> float:
    return (value - reference + 180.0) % 360.0 - 180.0


def _osculating_elements(
    barycentric_state: np.ndarray,
    jd_tdb: float,
    environment: DE440Environment,
) -> dict[str, float]:
    et = environment.jd_to_et(jd_tdb)
    heliocentric = barycentric_state - environment.state("SUN", et)
    rotation = np.asarray(spice.pxform("J2000", "ECLIPJ2000", et), dtype=float)
    ecliptic_state = np.concatenate(
        (rotation @ heliocentric[:3], rotation @ heliocentric[3:])
    )
    elements = spice.oscelt(ecliptic_state, et, environment.gm["SUN"])
    perihelion_km, eccentricity, inclination, node, argument, mean_anomaly = (
        elements[:6]
    )
    return {
        "e": float(eccentricity),
        "q_au": float(perihelion_km / AU_KM),
        "i_deg": float(np.degrees(inclination)),
        "node_deg": float(np.degrees(node) % 360.0),
        "argperi_deg": float(np.degrees(argument) % 360.0),
        "mean_anomaly_deg": float(np.degrees(mean_anomaly) % 360.0),
        "a_au": float(perihelion_km / (1.0 - eccentricity) / AU_KM),
    }


def _closest_approach(
    jd: np.ndarray,
    states: np.ndarray,
    environment: DE440Environment,
) -> tuple[float, float]:
    earth_positions = np.asarray(
        [
            environment.state("EARTH", environment.jd_to_et(epoch))[:3]
            for epoch in jd
        ]
    )
    distances = np.linalg.norm(states[:, :3] - earth_positions, axis=1)
    index = int(np.argmin(distances))
    if index == 0 or index == len(jd) - 1:
        raise RuntimeError("Closest approach falls on the propagation boundary.")

    seconds = (jd - jd[0]) * DAY_S
    state_spline = [
        CubicSpline(seconds, states[:, component])
        for component in range(3)
    ]

    def separation(value: float) -> float:
        asteroid = np.array([spline(value) for spline in state_spline])
        epoch = jd[0] + value / DAY_S
        earth = environment.state(
            "EARTH", environment.jd_to_et(epoch)
        )[:3]
        return float(np.linalg.norm(asteroid - earth))

    result = minimize_scalar(
        separation,
        bounds=(seconds[index - 2], seconds[index + 2]),
        method="bounded",
        options={"xatol": 1e-4},
    )
    if not result.success:
        raise RuntimeError(result.message)
    return jd[0] + float(result.x) / DAY_S, float(result.fun)


def validate_case(
    designation: str,
    date_min: str,
    date_max: str,
) -> dict[str, object]:
    observations = fetch_mpc_ades(designation, CACHE_DIR)
    observation_summary = summarize_mpc_ades(observations)
    approaches = jpl_close_approaches(
        designation, date_min, date_max, distance_max_au=0.05
    )
    if not approaches:
        raise RuntimeError(f"JPL CAD returned no encounter for {designation}.")
    official = min(approaches, key=lambda row: float(row["dist"]))
    official_jd = float(official["jd"])
    official_distance_km = float(official["dist"]) * AU_KM

    start_jd = official_jd - 2.0
    stop_jd = official_jd + 2.0
    result = propagate_custom(
        designation,
        start_jd,
        stop_jd,
        samples=2001,
        model=ForceModel(),
        kernel_dir=KERNEL_DIR,
        validate_horizons=False,
        include_large_asteroids=True,
        backend="fortran",
    )
    environment = DE440Environment(KERNEL_DIR, include_large_asteroids=True)
    local_jd, local_distance_km = _closest_approach(
        result.jd_tdb, result.state_km_kms, environment
    )

    element_jd = official_jd - 1.0
    local_index = int(np.argmin(np.abs(result.jd_tdb - element_jd)))
    local_elements = _osculating_elements(
        result.state_km_kms[local_index], element_jd, environment
    )
    official_row, _ = horizons_elements(designation, [element_jd])
    official_elements = {
        "e": official_row[0, 1],
        "q_au": official_row[0, 2],
        "i_deg": official_row[0, 3],
        "node_deg": official_row[0, 4],
        "argperi_deg": official_row[0, 5],
        "mean_anomaly_deg": official_row[0, 6],
        "a_au": official_row[0, 7],
    }

    official_utc = Time(official_jd, format="jd", scale="tdb").utc.datetime.replace(
        tzinfo=timezone.utc
    )
    first_utc = observation_summary["first_observation_utc"]
    discovery_utc = observation_summary["discovery_observation_utc"]
    lead_hours = (official_utc - discovery_utc).total_seconds() / 3600.0
    row: dict[str, object] = {
        "designation": designation,
        "mpc_observation_count": observation_summary["observation_count"],
        "mpc_station_count": observation_summary["station_count"],
        "mpc_first_observation_utc": first_utc.isoformat(),
        "mpc_discovery_observation_utc": discovery_utc.isoformat(),
        "mpc_discovery_marker_available": observation_summary[
            "discovery_marker_available"
        ],
        "mpc_last_observation_utc": observation_summary[
            "last_observation_utc"
        ].isoformat(),
        "mpc_arc_days": observation_summary["arc_days"],
        "discovery_lead_hours": lead_hours,
        "official_ca_utc": official_utc.isoformat(),
        "official_ca_jd_tdb": official_jd,
        "official_distance_km": official_distance_km,
        "official_velocity_km_s": float(official["v_rel"]),
        "local_ca_jd_tdb": local_jd,
        "ca_time_error_s": (local_jd - official_jd) * DAY_S,
        "local_distance_km": local_distance_km,
        "ca_distance_error_km": local_distance_km - official_distance_km,
        "element_epoch_jd_tdb": element_jd,
        "function_evaluations": result.function_evaluations,
        "backend": result.kernel,
    }
    for key in ("a_au", "e", "i_deg", "node_deg", "argperi_deg"):
        row[f"local_{key}"] = local_elements[key]
        row[f"official_{key}"] = official_elements[key]
    row["delta_a_km"] = (
        local_elements["a_au"] - official_elements["a_au"]
    ) * AU_KM
    row["delta_e"] = local_elements["e"] - official_elements["e"]
    row["delta_i_arcsec"] = (
        local_elements["i_deg"] - official_elements["i_deg"]
    ) * 3600.0
    row["delta_node_arcsec"] = (
        _wrapped_angle_difference_deg(
            local_elements["node_deg"], official_elements["node_deg"]
        )
        * 3600.0
    )
    row["delta_argperi_arcsec"] = (
        _wrapped_angle_difference_deg(
            local_elements["argperi_deg"],
            official_elements["argperi_deg"],
        )
        * 3600.0
    )
    return row


def _make_figure(rows: list[dict[str, object]]) -> None:
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    labels = [str(row["designation"]) for row in rows]
    y = np.arange(len(labels))
    time_error = np.array([float(row["ca_time_error_s"]) for row in rows])
    distance_error = np.array(
        [float(row["ca_distance_error_km"]) for row in rows]
    )
    lead = np.array([float(row["discovery_lead_hours"]) for row in rows])

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.labelcolor": "#151B23",
            "axes.edgecolor": "#5C6875",
            "xtick.color": "#151B23",
            "ytick.color": "#151B23",
            "text.color": "#151B23",
            "axes.titleweight": "bold",
        }
    )
    figure, axes = plt.subplots(
        1, 3, figsize=(15.2, 7.0), sharey=True, facecolor="white"
    )
    colors = ["#B86B00" if value >= 0 else "#A7354D" for value in lead]
    panels = (
        (lead, "MPC discovery relative to CA [h]", colors),
        (time_error, "Local - JPL CA time [s]", "#007C77"),
        (distance_error, "Local - JPL CA distance [km]", "#245AA6"),
    )
    for axis, (values, xlabel, color) in zip(axes, panels, strict=True):
        axis.set_facecolor("white")
        axis.axvline(0, color="#5C6875", lw=0.8, alpha=0.7)
        axis.barh(y, values, color=color, height=0.62)
        axis.grid(axis="x", color="#5C6875", alpha=0.20, lw=0.7)
        axis.set_xlabel(xlabel)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_title("MPC observing context")
    axes[1].set_title("Close-approach epoch")
    axes[2].set_title("Geocentric miss distance")
    figure.suptitle(
        "Ten historical close approaches: MPC observations + JPL-referenced propagation",
        fontsize=17,
        y=0.98,
    )
    figure.text(
        0.5,
        0.015,
        "Yellow: detected before closest approach; red: first reported after flyby. "
        "Initial state and official comparison use the current JPL orbit solution.",
        ha="center",
        color="#5C6875",
        fontsize=10.5,
    )
    figure.tight_layout(rect=(0.02, 0.05, 1.0, 0.94))
    figure.savefig(
        FIGURE_PATH,
        dpi=220,
        facecolor="white",
    )
    plt.close(figure)


def _write_tex_tables(rows: list[dict[str, object]]) -> None:
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\footnotesize",
        r"\caption{Historical close-approach reproduction. "
        r"$N_{\rm obs}/N_{\rm stn}$ and lead time are measured from MPC ADES "
        r"observations \cite{MPCObservations}; the official epoch and "
        r"geocentric distance are NASA/JPL CNEOS CAD values \cite{JPLCAD}. "
        r"Residuals are local Fortran minus JPL. Positive lead time means "
        r"discovery before closest approach.}",
        r"\label{tab:neo-ca-validation}",
        r"\begin{tabularx}{\textwidth}{@{}l r r r r r r@{}}",
        r"\toprule",
        r"\color{cGold}\textbf{Object} & "
        r"\color{cGold}\textbf{$N_{\rm obs}/N_{\rm stn}$} & "
        r"\color{cGold}\textbf{Lead [h]} & "
        r"\color{cGold}\textbf{JPL CA [UTC]} & "
        r"\color{cGold}\textbf{$d_{\rm JPL}$ [km]} & "
        r"\color{cGold}\textbf{$\Delta t$ [s]} & "
        r"\color{cGold}\textbf{$\Delta d$ [km]}\\",
        r"\midrule",
    ]
    for row in rows:
        ca = datetime.fromisoformat(str(row["official_ca_utc"])).strftime(
            "%Y-%m-%d %H:%M"
        )
        lines.append(
            f"{row['designation']} & {int(row['mpc_observation_count'])}/"
            f"{int(row['mpc_station_count'])} & "
            f"{float(row['discovery_lead_hours']):+.1f} & {ca} & "
            f"{float(row['official_distance_km']):.1f} & "
            f"{float(row['ca_time_error_s']):+.3f} & "
            f"{float(row['ca_distance_error_km']):+.3f}\\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabularx}",
            r"\end{table}",
            "",
            r"\begin{table}[H]",
            r"\centering",
            r"\small",
            r"\caption{Osculating-element residuals one day before closest "
            r"approach. Columns labeled JPL are the current NASA/JPL Horizons "
            r"heliocentric ecliptic solution \cite{Horizons}; each "
            r"$\Delta$ is local Fortran minus JPL at the same TDB epoch.}",
            r"\label{tab:neo-element-validation}",
            r"\begin{tabularx}{\textwidth}{@{}l r r r r r r@{}}",
            r"\toprule",
            r"\color{cGold}\textbf{Object} & "
            r"\color{cGold}\textbf{$a_{\rm JPL}$ [au]} & "
            r"\color{cGold}\textbf{$\Delta a$ [km]} & "
            r"\color{cGold}\textbf{$e_{\rm JPL}$} & "
            r"\color{cGold}\textbf{$10^9\Delta e$} & "
            r"\color{cGold}\textbf{$i_{\rm JPL}$ [deg]} & "
            r"\color{cGold}\textbf{$\Delta i$ [mas]}\\",
            r"\midrule",
        ]
    )
    for row in rows:
        lines.append(
            f"{row['designation']} & {float(row['official_a_au']):.7f} & "
            f"{float(row['delta_a_km']):+.3f} & "
            f"{float(row['official_e']):.7f} & "
            f"{float(row['delta_e']) * 1e9:+.3f} & "
            f"{float(row['official_i_deg']):.5f} & "
            f"{float(row['delta_i_arcsec']) * 1000:+.3f}\\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabularx}",
            r"\end{table}",
            "",
        ]
    )
    TEX_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows = []
    for designation, date_min, date_max in CASES:
        print(f"Validating {designation}...", flush=True)
        row = validate_case(designation, date_min, date_max)
        rows.append(row)
        print(
            f"  dt={row['ca_time_error_s']:+.3f} s, "
            f"dd={row['ca_distance_error_km']:+.3f} km, "
            f"da={row['delta_a_km']:+.3f} km",
            flush=True,
        )

    with CSV_PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    provenance = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Observation-backed force-model propagation validation; not an "
            "independent orbit determination from the MPC astrometry."
        ),
        "mpc_observations_api": (
            "https://data.minorplanetcenter.net/api/get-obs"
        ),
        "jpl_close_approach_api": "https://ssd-api.jpl.nasa.gov/cad.api",
        "jpl_horizons_api": "https://ssd.jpl.nasa.gov/api/horizons.api",
        "cases": [row["designation"] for row in rows],
        "force_model": rows[0]["backend"],
    }
    PROVENANCE_PATH.write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    _make_figure(rows)
    _write_tex_tables(rows)
    print(f"Wrote {CSV_PATH}")
    print(f"Wrote {FIGURE_PATH}")
    print(f"Wrote {TEX_PATH}")


if __name__ == "__main__":
    main()
