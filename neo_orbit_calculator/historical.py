"""Historical comet finder charts and Joseon-record comparisons."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astroquery.jplhorizons import Horizons
from astroquery.vizier import Vizier

from .comets import fetch_comet_apparition


@dataclass(frozen=True)
class ObserverSite:
    name: str
    longitude_deg_east: float
    latitude_deg: float
    elevation_km: float

    def horizons_location(self) -> dict[str, float]:
        return {
            "lon": self.longitude_deg_east,
            "lat": self.latitude_deg,
            "elevation": self.elevation_km,
        }


@dataclass(frozen=True)
class HistoricalRecord:
    record_id: str
    title: str
    designation: str
    center_epoch_utc: str
    local_civil_date: str
    lunar_date: str
    site: ObserverSite
    source_url: str
    source_credit: str
    reported_mansion: str
    reported_asterism_relation: str
    reported_north_polar_distance_deg: float | None
    note: str


SEOUL_GWANSANGGAM = ObserverSite(
    name="Hanyang, Gwansanggam reference site",
    longitude_deg_east=126.9780,
    latitude_deg=37.5665,
    elevation_km=0.05,
)

JOSEON_HALLEY_1759 = HistoricalRecord(
    record_id="joseon-halley-1759-03-11",
    title="Joseon observation of 1P/Halley on lunar 1759 March 11",
    designation="1P",
    center_epoch_utc="1759-04-06T20:30:00",
    local_civil_date="1759-04-07 near dawn",
    lunar_date="1759-03-11",
    site=SEOUL_GWANSANGGAM,
    source_url=(
        "https://www.kasi.re.kr/kor/publication/post/newsMaterial/29446"
    ),
    source_credit=(
        "Seongbyeon Cheukhu Danja, Yonsei University Library. "
        "Digital image provided by the Korea Astronomy and Space Science "
        "Institute."
    ),
    reported_mansion="Xu lunar mansion",
    reported_asterism_relation="north of the Liyu asterism",
    reported_north_polar_distance_deg=116.0,
    note=(
        "The raw north-polar-distance value is retained as documentary "
        "metadata. It is not included in the numerical score until the "
        "instrument scale, transcription, and epoch convention are "
        "calibrated from the full manuscript."
    ),
)

HISTORICAL_RECORDS = {
    JOSEON_HALLEY_1759.record_id: JOSEON_HALLEY_1759,
}

# J2000 ICRS positions. Xu is bounded by beta and alpha Aquarii in right
# ascension. The compact 42, 44, and 45 Capricorni triplet is the commonly
# adopted identification of Liyu in this part of the sky.
XU_RA_BOUNDARIES_DEG = (322.88971698347876, 331.4459814409479)
LIYU_STARS = (
    ("42 Cap", 325.3869116932721, -14.047612442258888),
    ("44 Cap", 325.76832108434, -14.39971028135),
    ("45 Cap", 326.00401721056, -14.74937218483),
)


def epoch_grid(
    center_epoch_utc: str,
    span_days: float,
    samples: int,
) -> list[str]:
    """Return a symmetric UTC grid formatted for Horizons."""
    if span_days <= 0:
        raise ValueError("span_days must be positive.")
    if samples < 3:
        raise ValueError("samples must be at least three.")
    center = Time(center_epoch_utc, scale="utc")
    offsets = np.linspace(-0.5 * span_days, 0.5 * span_days, samples)
    return [
        Time(center.jd + float(offset), format="jd", scale="utc").isot
        for offset in offsets
    ]


def resolve_apparition_record(
    designation: str,
    epoch_utc: str,
    apparition_record: int | None = None,
) -> int:
    """Select the fitted Horizons apparition that precedes the epoch year."""
    if apparition_record is not None:
        return apparition_record
    year = int(Time(epoch_utc, scale="utc").datetime.year)
    return fetch_comet_apparition(designation, year + 1).record


def historical_comet_positions(
    designation: str,
    epochs: list[str],
    site: ObserverSite,
    apparition_record: int | None = None,
) -> tuple[list[dict[str, object]], int]:
    """Return topocentric coordinates from one historical apparition fit."""
    if not epochs:
        raise ValueError("At least one epoch is required.")
    record = resolve_apparition_record(
        designation,
        epochs[len(epochs) // 2],
        apparition_record,
    )
    times = Time(epochs, scale="utc")
    table = Horizons(
        id=str(record),
        id_type=None,
        location=site.horizons_location(),
        epochs=list(times.jd),
    ).ephemerides(extra_precision=True)
    coordinates = SkyCoord(
        ra=np.asarray(table["RA"], dtype=float) * u.deg,
        dec=np.asarray(table["DEC"], dtype=float) * u.deg,
        frame="icrs",
    )
    constellations = coordinates.get_constellation(short_name=False)
    rows: list[dict[str, object]] = []
    for index, epoch in enumerate(epochs):
        rows.append(
            {
                "epoch_utc": epoch,
                "jd_utc": float(times.jd[index]),
                "apparition_record": record,
                "ra_icrs_deg": float(table["RA"][index]),
                "dec_icrs_deg": float(table["DEC"][index]),
                "ra_apparent_deg": float(table["RA_app"][index]),
                "dec_apparent_deg": float(table["DEC_app"][index]),
                "azimuth_deg": float(table["AZ"][index]),
                "elevation_deg": float(table["EL"][index]),
                "solar_elongation_deg": float(table["elong"][index]),
                "heliocentric_distance_au": float(table["r"][index]),
                "observer_distance_au": float(table["delta"][index]),
                "constellation": str(constellations[index]),
            }
        )
    return rows, record


def _wrapped_ra(values: np.ndarray, center_deg: float) -> np.ndarray:
    return center_deg + (values - center_deg + 180.0) % 360.0 - 180.0


def fetch_bright_stars(
    center_ra_deg: float,
    center_dec_deg: float,
    radius_deg: float,
    limiting_magnitude: float = 6.5,
) -> list[dict[str, float]]:
    """Retrieve Hipparcos stars for a finder chart."""
    query = Vizier(
        columns=["HIP", "Vmag", "_RA.icrs", "_DE.icrs"],
        column_filters={"Vmag": f"<{limiting_magnitude}"},
        row_limit=-1,
    )
    tables = query.query_region(
        SkyCoord(
            ra=center_ra_deg * u.deg,
            dec=center_dec_deg * u.deg,
            frame="icrs",
        ),
        radius=radius_deg * u.deg,
        catalog="I/239/hip_main",
    )
    if not tables:
        raise RuntimeError("The Hipparcos finder-chart query returned no stars.")
    table = tables[0]
    return [
        {
            "hip": float(row["HIP"]),
            "ra_deg": float(row["_RA.icrs"]),
            "dec_deg": float(row["_DE.icrs"]),
            "vmag": float(row["Vmag"]),
        }
        for row in table
        if not np.ma.is_masked(row["Vmag"])
    ]


def evaluate_joseon_constraints(
    row: dict[str, object],
    record: HistoricalRecord,
) -> dict[str, object]:
    """Evaluate only the record constraints with an explicit calibration."""
    ra = float(row["ra_icrs_deg"])
    dec = float(row["dec_icrs_deg"])
    xu_low, xu_high = XU_RA_BOUNDARIES_DEG
    liyu_mean_dec = float(np.mean([item[2] for item in LIYU_STARS]))
    apparent_dec = float(row.get("dec_apparent_deg", dec))
    predicted_npd = 90.0 - apparent_dec
    raw_npd = record.reported_north_polar_distance_deg
    return {
        "record_id": record.record_id,
        "apparition_record": int(row["apparition_record"]),
        "epoch_utc": str(row["epoch_utc"]),
        "inside_xu_ra_interval": bool(xu_low <= ra <= xu_high),
        "xu_ra_margin_deg": float(min(ra - xu_low, xu_high - ra)),
        "north_of_liyu": bool(dec > liyu_mean_dec),
        "north_of_liyu_deg": float(dec - liyu_mean_dec),
        "predicted_north_polar_distance_deg": predicted_npd,
        "north_polar_distance_coordinate": "apparent equator of date",
        "reported_north_polar_distance_raw_deg": raw_npd,
        "raw_north_polar_distance_residual_deg": (
            None if raw_npd is None else predicted_npd - raw_npd
        ),
        "north_polar_distance_used_in_score": False,
        "elevation_deg": float(row["elevation_deg"]),
        "solar_elongation_deg": float(row["solar_elongation_deg"]),
        "interpretation": record.note,
    }


def write_historical_products(
    output_dir: Path,
    designation: str,
    rows: list[dict[str, object]],
    site: ObserverSite,
    record: HistoricalRecord | None = None,
    field_radius_deg: float = 12.0,
) -> tuple[Path, Path, Path]:
    """Write the coordinate table, provenance summary, and finder chart."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "historical_comet_positions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    center_index = len(rows) // 2
    center = rows[center_index]
    summary: dict[str, object] = {
        "designation": designation,
        "apparition_record": int(center["apparition_record"]),
        "observer": asdict(site),
        "center_position": center,
    }
    if record is not None:
        summary["historical_record"] = asdict(record)
        summary["constraint_evaluation"] = evaluate_joseon_constraints(
            center,
            record,
        )
    summary_path = output_dir / "historical_comet_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    plot_path = output_dir / "historical_comet_finder_chart.png"
    plot_historical_finder_chart(
        plot_path,
        designation,
        rows,
        record=record,
        field_radius_deg=field_radius_deg,
    )
    return csv_path, summary_path, plot_path


def plot_historical_finder_chart(
    path: Path,
    designation: str,
    rows: list[dict[str, object]],
    record: HistoricalRecord | None = None,
    field_radius_deg: float = 12.0,
) -> None:
    """Plot a dark finder chart centered on a historical comet track."""
    ra = np.array([float(row["ra_icrs_deg"]) for row in rows])
    dec = np.array([float(row["dec_icrs_deg"]) for row in rows])
    center_ra = float(ra[len(ra) // 2])
    center_dec = float(dec[len(dec) // 2])
    ra = _wrapped_ra(ra, center_ra)
    stars = fetch_bright_stars(
        center_ra,
        center_dec,
        radius_deg=field_radius_deg * 1.05,
    )
    star_ra = _wrapped_ra(
        np.array([star["ra_deg"] for star in stars]),
        center_ra,
    )
    star_dec = np.array([star["dec_deg"] for star in stars])
    star_mag = np.array([star["vmag"] for star in stars])
    star_size = np.clip(100.0 * 10.0 ** (-0.32 * (star_mag - 1.0)), 5.0, 90.0)

    figure, axis = plt.subplots(figsize=(11.6, 8.0), facecolor="#071310")
    axis.set_facecolor("#071310")
    axis.scatter(
        star_ra,
        star_dec,
        s=star_size,
        color="#f4f1df",
        alpha=0.90,
        linewidths=0,
        zorder=1,
    )
    if record is not None and record.record_id == JOSEON_HALLEY_1759.record_id:
        xu_low, xu_high = XU_RA_BOUNDARIES_DEG
        axis.axvspan(
            xu_low,
            xu_high,
            color="#4cc9b0",
            alpha=0.10,
            label="Xu lunar mansion RA interval",
        )
        liyu_ra = _wrapped_ra(
            np.array([item[1] for item in LIYU_STARS]),
            center_ra,
        )
        liyu_dec = np.array([item[2] for item in LIYU_STARS])
        axis.plot(
            liyu_ra,
            liyu_dec,
            color="#56d6c2",
            lw=2.2,
            marker="o",
            markersize=5,
            label="Liyu asterism",
            zorder=3,
        )
        axis.annotate(
            "Liyu",
            (float(np.mean(liyu_ra)), float(np.mean(liyu_dec))),
            xytext=(-4, -17),
            textcoords="offset points",
            ha="center",
            color="#56d6c2",
            fontsize=11,
        )

    axis.plot(
        ra,
        dec,
        color="#ff9f43",
        lw=2.2,
        alpha=0.9,
        label=f"{designation} historical apparition fit",
        zorder=4,
    )
    center_index = len(rows) // 2
    axis.scatter(
        ra[center_index],
        dec[center_index],
        marker="*",
        s=260,
        color="#ffe066",
        edgecolor="#101713",
        linewidth=0.8,
        label=str(rows[center_index]["epoch_utc"]),
        zorder=5,
    )
    axis.annotate(
        str(rows[center_index]["constellation"]),
        (ra[center_index], dec[center_index]),
        xytext=(10, 10),
        textcoords="offset points",
        color="#ffe066",
        fontsize=11,
        fontweight="bold",
    )

    axis.set_xlim(center_ra + field_radius_deg, center_ra - field_radius_deg)
    axis.set_ylim(center_dec - field_radius_deg, center_dec + field_radius_deg)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(color="#b8c8c0", alpha=0.13, lw=0.7)
    axis.tick_params(colors="#e9eee9")
    axis.xaxis.label.set_color("#e9eee9")
    axis.yaxis.label.set_color("#e9eee9")
    axis.title.set_color("#ffe066")
    axis.set_xlabel("ICRS right ascension [deg, increasing to the left]")
    axis.set_ylabel("ICRS declination [deg]")
    title = f"{designation} on the historical sky"
    if record is not None:
        title += f"\n{record.local_civil_date} from {record.site.name}"
    axis.set_title(title, fontsize=15)
    legend = axis.legend(
        loc="upper left",
        frameon=True,
        facecolor="#10221c",
        edgecolor="#335a4d",
        fontsize=9,
    )
    for text in legend.get_texts():
        text.set_color("#f4f1e8")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, facecolor=figure.get_facecolor())
    plt.close(figure)
