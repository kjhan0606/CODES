"""Command-line interface for CODES."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time

from .comets import (
    collect_comet_apparitions,
    comet_sky_positions,
    plot_orbit_evolution,
    plot_sky_positions,
    write_apparitions_csv,
    write_sky_csv,
)
from .core import AU_KM, ForceModel, propagate_custom
from .jpl import download_horizons_spk, horizons_vectors


def _jd(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return float(Time(value, scale="tdb").jd)


def _write_csv(path: Path, rows: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["jd_tdb", "x_km", "y_km", "z_km", "vx_km_s", "vy_km_s", "vz_km_s"])
        writer.writerows(rows)


def _plot_orbit(path: Path, designation: str, jd: np.ndarray, states: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    position = states[:, :3] / AU_KM
    figure, axis = plt.subplots(figsize=(9.2, 7.2), facecolor="#0e1b18")
    axis.set_facecolor("#0e1b18")
    axis.plot(position[:, 0], position[:, 1], color="#56d6c2", lw=1.4)
    axis.scatter(position[0, 0], position[0, 1], color="#ffd166", s=45, label="start")
    axis.scatter(0, 0, color="#ff8a4c", s=75, label="SSB origin")
    axis.set(
        xlabel="ICRF x [au]",
        ylabel="ICRF y [au]",
        title=f"{designation}: barycentric projection, JD {jd[0]:.1f}-{jd[-1]:.1f}",
    )
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(color="#c4c9c1", alpha=0.16)
    axis.tick_params(colors="#f4f1e8")
    axis.xaxis.label.set_color("#f4f1e8")
    axis.yaxis.label.set_color("#f4f1e8")
    axis.title.set_color("#ffd166")
    legend = axis.legend(frameon=False)
    for text in legend.get_texts():
        text.set_color("#f4f1e8")
    figure.tight_layout()
    figure.savefig(path, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "CODES, the NASA/JPL-backed Close-approach Orbit Dynamics "
            "and Ephemeris System"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    spk = subparsers.add_parser(
        "spk",
        help="Download an authoritative Horizons time-continuous SPK",
    )
    spk.add_argument("designation", help="Example: 99942 or '2004 MN4'")
    spk.add_argument("--start", default="2026-01-01")
    spk.add_argument("--stop", default="2126-01-01")
    spk.add_argument("--output-dir", type=Path, default=Path("output"))

    vectors = subparsers.add_parser(
        "vectors",
        help="Fetch authoritative Horizons state vectors",
    )
    vectors.add_argument("designation")
    vectors.add_argument("--start", default="2026-01-01")
    vectors.add_argument("--stop", default="2126-01-01")
    vectors.add_argument("--samples", type=int, default=101)
    vectors.add_argument("--output", type=Path, default=Path("output/horizons_vectors.csv"))

    custom = subparsers.add_parser(
        "propagate",
        help="Run the local DE440 sensitivity integrator",
    )
    custom.add_argument("designation")
    custom.add_argument("--start", default="2026-01-01")
    custom.add_argument("--stop", default="2126-01-01")
    custom.add_argument("--samples", type=int, default=401)
    custom.add_argument("--kernel-dir", type=Path, default=Path("kernels"))
    custom.add_argument("--output-dir", type=Path, default=Path("output"))
    custom.add_argument("--area-mass", type=float, default=0.0, help="m^2/kg")
    custom.add_argument("--cr", type=float, default=1.0)
    custom.add_argument("--solar-wind-ratio", type=float, default=0.35)
    custom.add_argument("--a1", type=float, default=0.0, help="au/day^2")
    custom.add_argument("--a2", type=float, default=0.0, help="au/day^2")
    custom.add_argument("--a3", type=float, default=0.0, help="au/day^2")
    custom.add_argument("--no-relativity", action="store_true")
    custom.add_argument("--no-pr", action="store_true")
    custom.add_argument("--no-solar-wind", action="store_true")
    custom.add_argument("--no-validation", action="store_true")
    custom.add_argument("--backend", choices=("fortran", "scipy"), default="fortran")
    custom.add_argument(
        "--major-bodies-only",
        action="store_true",
        help="Skip the 16 SB441-N16 main-belt perturbers",
    )

    comet_orbits = subparsers.add_parser(
        "comet-orbits",
        help="Plot apparition-to-apparition orbit evolution for a comet",
    )
    comet_orbits.add_argument("designation", help="Example: 1P")
    comet_orbits.add_argument("--start-year", type=int, default=800)
    comet_orbits.add_argument("--stop-year", type=int, default=2100)
    comet_orbits.add_argument(
        "--return-years",
        type=int,
        nargs="+",
        help="Known return years for a comet not resolved by Horizons aliases",
    )
    comet_orbits.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/comet_orbits"),
    )

    comet_sky = subparsers.add_parser(
        "comet-sky",
        help="Plot apparent comet positions and report IAU constellations",
    )
    comet_sky.add_argument("designation", help="Example: 1P")
    comet_sky.add_argument(
        "--epochs",
        nargs="+",
        required=True,
        help="UTC epochs such as 2061-07-28",
    )
    comet_sky.add_argument(
        "--observer",
        default="500@399",
        help="Horizons observer center, default geocentric",
    )
    comet_sky.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/comet_sky"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "comet-orbits":
        apparitions = collect_comet_apparitions(
            args.designation,
            args.start_year,
            args.stop_year,
            args.return_years,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = args.output_dir / "comet_apparitions.csv"
        plot_path = args.output_dir / "comet_orbit_evolution.png"
        write_apparitions_csv(csv_path, apparitions)
        plot_orbit_evolution(plot_path, apparitions)
        print(
            json.dumps(
                {
                    "designation": args.designation,
                    "apparitions": len(apparitions),
                    "csv": str(csv_path.resolve()),
                    "plot": str(plot_path.resolve()),
                },
                indent=2,
            )
        )
        return

    if args.command == "comet-sky":
        rows = comet_sky_positions(
            args.designation,
            args.epochs,
            observer=args.observer,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = args.output_dir / "comet_sky_positions.csv"
        plot_path = args.output_dir / "comet_sky_positions.png"
        write_sky_csv(csv_path, rows)
        plot_sky_positions(plot_path, args.designation, rows)
        print(
            json.dumps(
                {
                    "designation": args.designation,
                    "positions": rows,
                    "csv": str(csv_path.resolve()),
                    "plot": str(plot_path.resolve()),
                },
                indent=2,
            )
        )
        return

    if args.command == "spk":
        path, spk_id = download_horizons_spk(
            args.designation,
            args.start,
            args.stop,
            args.output_dir,
        )
        print(json.dumps({"spk_id": spk_id, "path": str(path.resolve())}, indent=2))
        return

    start = _jd(args.start)
    stop = _jd(args.stop)
    jd = np.linspace(start, stop, args.samples)
    if args.command == "vectors":
        rows, source = horizons_vectors(args.designation, jd)
        _write_csv(args.output, rows)
        print(json.dumps({"source": source, "path": str(args.output.resolve())}, indent=2))
        return

    model = ForceModel(
        relativity_1pn=not args.no_relativity,
        area_mass_m2_kg=args.area_mass,
        radiation_coefficient=args.cr,
        poynting_robertson=not args.no_pr,
        solar_wind_drag=not args.no_solar_wind,
        solar_wind_to_pr=args.solar_wind_ratio,
        a1_au_day2=args.a1,
        a2_au_day2=args.a2,
        a3_au_day2=args.a3,
    )
    result = propagate_custom(
        args.designation,
        start,
        stop,
        samples=args.samples,
        model=model,
        kernel_dir=args.kernel_dir,
        validate_horizons=not args.no_validation,
        include_large_asteroids=not args.major_bodies_only,
        backend=args.backend,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "custom_states.csv"
    plot_path = args.output_dir / "custom_orbit.png"
    _write_csv(csv_path, np.column_stack((result.jd_tdb, result.state_km_kms)))
    _plot_orbit(plot_path, args.designation, result.jd_tdb, result.state_km_kms)
    valid = (
        np.isfinite(result.position_residual_km)
        if result.position_residual_km is not None
        else np.array([], dtype=bool)
    )
    summary = {
        "designation": args.designation,
        "kernel": result.kernel,
        "function_evaluations": result.function_evaluations,
        "state_csv": str(csv_path.resolve()),
        "plot": str(plot_path.resolve()),
        "warning": result.warning,
    }
    if valid.any():
        summary["max_horizons_position_residual_km"] = float(
            np.nanmax(result.position_residual_km)
        )
        summary["end_horizons_position_residual_km"] = float(
            result.position_residual_km[np.where(valid)[0][-1]]
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
