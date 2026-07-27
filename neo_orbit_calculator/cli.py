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
from .covariance import (
    fetch_sbdb_covariance,
    propagate_virtual_asteroids,
    write_virtual_asteroid_products,
)
from .historical import (
    HISTORICAL_RECORDS,
    ObserverSite,
    SEOUL_GWANSANGGAM,
    epoch_grid,
    historical_comet_positions,
    write_historical_products,
)
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
        help="Run the local DE442/DE441 sensitivity integrator",
    )
    custom.add_argument("designation")
    custom.add_argument("--start", default="2026-01-01")
    custom.add_argument("--stop", default="2126-01-01")
    custom.add_argument("--samples", type=int, default=401)
    custom.add_argument("--kernel-dir", type=Path, default=Path("kernels"))
    custom.add_argument("--output-dir", type=Path, default=Path("output"))
    custom.add_argument("--area-mass", type=float, default=0.0, help="m^2/kg")
    custom.add_argument("--cr", type=float, default=1.0)
    custom.add_argument("--solar-wind-density", type=float, default=5.0)
    custom.add_argument("--solar-wind-speed", type=float, default=400.0)
    custom.add_argument("--solar-wind-momentum-factor", type=float, default=1.2)
    custom.add_argument("--a1", type=float, default=0.0, help="au/day^2")
    custom.add_argument("--a2", type=float, default=0.0, help="au/day^2")
    custom.add_argument("--a3", type=float, default=0.0, help="au/day^2")
    custom.add_argument(
        "--nongrav-law",
        choices=("inverse_square", "marsden"),
        default="inverse_square",
    )
    custom.add_argument("--outgassing-r0", type=float, default=2.808)
    custom.add_argument("--outgassing-m", type=float, default=2.15)
    custom.add_argument("--outgassing-n", type=float, default=5.093)
    custom.add_argument("--outgassing-k", type=float, default=4.6142)
    custom.add_argument("--outgassing-alpha", type=float, default=0.111262)
    custom.add_argument("--outgassing-lag-days", type=float, default=0.0)
    custom.add_argument("--no-relativity", action="store_true")
    custom.add_argument(
        "--solar-1pn-only",
        action="store_true",
        help="Use the legacy Sun-only Schwarzschild term for comparison",
    )
    custom.add_argument("--no-pr", action="store_true")
    custom.add_argument("--no-solar-wind", action="store_true")
    custom.add_argument("--no-zonal-harmonics", action="store_true")
    custom.add_argument("--no-validation", action="store_true")
    custom.add_argument("--backend", choices=("fortran", "scipy"), default="fortran")
    custom.add_argument(
        "--ephemeris",
        choices=("auto", "de442", "de441"),
        default="auto",
        help="DE442 for modern work, DE441 for long-term work, or auto by coverage",
    )
    custom.add_argument(
        "--major-bodies-only",
        action="store_true",
        help="Skip the 16 SB441-N16 main-belt perturbers",
    )
    custom.add_argument(
        "--jupiter-system",
        action="store_true",
        help=(
            "Replace the Jupiter barycenter monopole with Jupiter and the "
            "mass-resolved JUP365 satellites; downloads a large optional kernel"
        ),
    )

    virtual = subparsers.add_parser(
        "virtual-asteroids",
        help="Propagate the full JPL covariance as correlated virtual asteroids",
    )
    virtual.add_argument("designation")
    virtual.add_argument("--stop", default="2030-01-01")
    virtual.add_argument("--clones", type=int, default=100)
    virtual.add_argument(
        "--samples",
        type=int,
        default=1001,
        help="Coarse output samples used to locate the closest approach",
    )
    virtual.add_argument("--seed", type=int, default=42)
    virtual.add_argument("--kernel-dir", type=Path, default=Path("kernels"))
    virtual.add_argument("--output-dir", type=Path, default=Path("output"))
    virtual.add_argument(
        "--ephemeris",
        choices=("auto", "de442", "de441"),
        default="auto",
        help="DE442 for modern work, DE441 for long-term work, or auto by coverage",
    )
    virtual.add_argument("--no-relativity", action="store_true")
    virtual.add_argument("--solar-1pn-only", action="store_true")
    virtual.add_argument("--no-zonal-harmonics", action="store_true")
    virtual.add_argument(
        "--major-bodies-only",
        action="store_true",
        help="Skip the 16 SB441-N16 main-belt perturbers",
    )
    virtual.add_argument(
        "--jupiter-system",
        action="store_true",
        help=(
            "Replace the Jupiter barycenter monopole with Jupiter and the "
            "mass-resolved JUP365 satellites"
        ),
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

    historical = subparsers.add_parser(
        "historical-comet",
        help="Plot one fitted comet apparition against a historical sky record",
    )
    historical.add_argument("designation", nargs="?", default="1P")
    historical.add_argument(
        "--record",
        choices=tuple(HISTORICAL_RECORDS),
        help="Bundled historical record and observing metadata",
    )
    historical.add_argument(
        "--epoch",
        help="Center UTC epoch. A bundled record supplies its own epoch.",
    )
    historical.add_argument("--span-days", type=float, default=4.0)
    historical.add_argument("--samples", type=int, default=17)
    historical.add_argument(
        "--apparition-record",
        type=int,
        help="Numeric JPL Horizons apparition record",
    )
    historical.add_argument(
        "--observer-lon",
        type=float,
        default=SEOUL_GWANSANGGAM.longitude_deg_east,
    )
    historical.add_argument(
        "--observer-lat",
        type=float,
        default=SEOUL_GWANSANGGAM.latitude_deg,
    )
    historical.add_argument(
        "--observer-elevation-km",
        type=float,
        default=SEOUL_GWANSANGGAM.elevation_km,
    )
    historical.add_argument("--field-radius", type=float, default=12.0)
    historical.add_argument("--kernel-dir", type=Path, default=Path("kernels"))
    historical.add_argument(
        "--jpl-only",
        action="store_true",
        help="Skip the local CODES DE441 propagation overlay",
    )
    historical.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/historical_comet"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "historical-comet":
        record = HISTORICAL_RECORDS.get(args.record)
        if record is not None:
            designation = record.designation
            center_epoch = record.center_epoch_utc
            site = record.site
        else:
            designation = args.designation
            if not args.epoch:
                raise ValueError("--epoch is required without --record.")
            center_epoch = args.epoch
            site = ObserverSite(
                name="User-specified historical observer",
                longitude_deg_east=args.observer_lon,
                latitude_deg=args.observer_lat,
                elevation_km=args.observer_elevation_km,
            )
        epochs = epoch_grid(center_epoch, args.span_days, args.samples)
        rows, apparition_record = historical_comet_positions(
            designation,
            epochs,
            site,
            apparition_record=args.apparition_record,
            kernel_dir=args.kernel_dir,
            run_codes=not args.jpl_only,
        )
        csv_path, summary_path, plot_path = write_historical_products(
            args.output_dir,
            designation,
            rows,
            site,
            record=record,
            field_radius_deg=args.field_radius,
        )
        print(
            json.dumps(
                {
                    "designation": designation,
                    "apparition_record": apparition_record,
                    "record": args.record,
                    "csv": str(csv_path.resolve()),
                    "summary": str(summary_path.resolve()),
                    "plot": str(plot_path.resolve()),
                },
                indent=2,
            )
        )
        return

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

    if args.command == "virtual-asteroids":
        solution = fetch_sbdb_covariance(args.designation)
        model = ForceModel(
            relativity_1pn=not args.no_relativity,
            full_multibody_1pn=not args.solar_1pn_only,
            planetary_zonal_harmonics=not args.no_zonal_harmonics,
        )
        result = propagate_virtual_asteroids(
            solution,
            _jd(args.stop),
            clones=args.clones,
            samples=args.samples,
            seed=args.seed,
            kernel_dir=args.kernel_dir,
            base_model=model,
            include_large_asteroids=not args.major_bodies_only,
            include_jupiter_system=args.jupiter_system,
            ephemeris=args.ephemeris,
        )
        csv_path, summary_path, figure_path = (
            write_virtual_asteroid_products(result, args.output_dir)
        )
        print(
            json.dumps(
                {
                    "designation": solution.designation,
                    "orbit_id": solution.orbit_id,
                    "covariance_epoch_jd_tdb": solution.epoch_jd_tdb,
                    "covariance_dimensions": len(solution.labels),
                    "clones": len(result.samples),
                    "minimum_earth_distance_km": float(
                        np.min(result.closest_approach_km)
                    ),
                    "nominal_closest_approach_jd_tdb": (
                        result.nominal_closest_approach_jd_tdb
                    ),
                    "nominal_closest_approach_km": (
                        result.nominal_closest_approach_km
                    ),
                    "screening_impact_count": (
                        result.screening_impact_count
                    ),
                    "clone_csv": str(csv_path.resolve()),
                    "summary": str(summary_path.resolve()),
                    "plot": str(figure_path.resolve()),
                },
                indent=2,
            )
        )
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
        full_multibody_1pn=not args.solar_1pn_only,
        area_mass_m2_kg=args.area_mass,
        radiation_coefficient=args.cr,
        poynting_robertson=not args.no_pr,
        solar_wind_drag=not args.no_solar_wind,
        solar_wind_density_cm3=args.solar_wind_density,
        solar_wind_speed_km_s=args.solar_wind_speed,
        solar_wind_momentum_factor=args.solar_wind_momentum_factor,
        planetary_zonal_harmonics=not args.no_zonal_harmonics,
        a1_au_day2=args.a1,
        a2_au_day2=args.a2,
        a3_au_day2=args.a3,
        nongrav_law=args.nongrav_law,
        outgassing_r0_au=args.outgassing_r0,
        outgassing_m=args.outgassing_m,
        outgassing_n=args.outgassing_n,
        outgassing_k=args.outgassing_k,
        outgassing_alpha=args.outgassing_alpha,
        outgassing_lag_days=args.outgassing_lag_days,
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
        include_jupiter_system=args.jupiter_system,
        backend=args.backend,
        ephemeris=args.ephemeris,
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
        "ephemeris": result.ephemeris,
        "ephemeris_reason": result.ephemeris_reason,
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
