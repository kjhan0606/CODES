"""Medium- and long-period comet history and sky-position products."""

from __future__ import annotations

import csv
import json
import re
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astroquery.jplhorizons import Horizons

from .jpl import HORIZONS_API

HORIZONS_LOOKUP_API = "https://ssd.jpl.nasa.gov/api/horizons_lookup.api"

HALLEY_RETURN_YEARS = (
    837,
    912,
    989,
    1066,
    1145,
    1222,
    1301,
    1378,
    1456,
    1531,
    1607,
    1682,
    1759,
    1835,
    1910,
    1986,
)


@dataclass(frozen=True)
class CometApparition:
    designation: str
    record: int
    perihelion_jd_tdb: float
    perihelion_calendar: str
    eccentricity: float
    perihelion_au: float
    semimajor_axis_au: float
    inclination_deg: float
    node_deg: float
    argument_perihelion_deg: float
    osculating_period_year: float

    @property
    def return_year(self) -> int:
        match = re.search(r"(-?\d{1,4})-", self.perihelion_calendar)
        if match is None:
            raise ValueError(
                f"Cannot parse return year from {self.perihelion_calendar!r}."
            )
        return int(match.group(1))


def _get_json(url: str) -> dict:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "CODES-Comet-Dynamics/1.0"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if "error" in payload:
        raise RuntimeError(payload["error"])
    return payload


def _number(result: str, key: str) -> float:
    match = re.search(
        rf"\b{re.escape(key)}=\s*"
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:E[+-]?\d+)?)",
        result,
        flags=re.IGNORECASE,
    )
    if match is None:
        raise ValueError(f"Horizons result has no {key} field.")
    return float(match.group(1))


def parse_apparition_result(
    result: str,
    designation: str,
) -> CometApparition:
    """Parse one Horizons comet-apparition object record."""
    record_match = re.search(r"Rec #:\s*(\d+)", result)
    calendar_matches = re.findall(
        r"\bTP=\s*((?:B\.C\.\s*)?\s*-?\d{1,4}-[A-Za-z]{3}-"
        r"\d{2}\.\d+)",
        result,
    )
    if record_match is None or not calendar_matches:
        raise ValueError("Horizons result is not a comet apparition record.")
    return CometApparition(
        designation=designation,
        record=int(record_match.group(1)),
        perihelion_jd_tdb=_number(result, "TP"),
        perihelion_calendar=" ".join(calendar_matches[-1].split()),
        eccentricity=_number(result, "EC"),
        perihelion_au=_number(result, "QR"),
        semimajor_axis_au=_number(result, "A"),
        inclination_deg=_number(result, "IN"),
        node_deg=_number(result, "OM"),
        argument_perihelion_deg=_number(result, "W"),
        osculating_period_year=_number(result, "PER"),
    )


def fetch_comet_apparition(
    designation: str,
    before_year: int,
) -> CometApparition:
    """Return the Horizons apparition immediately preceding ``before_year``."""
    command = f"NAME={designation.strip().rstrip(';')};CAP<{before_year}"
    parameters = {
        "format": "json",
        "COMMAND": f"'{command}'",
        "OBJ_DATA": "'YES'",
        "MAKE_EPHEM": "'NO'",
    }
    url = HORIZONS_API + "?" + urllib.parse.urlencode(parameters)
    payload = _get_json(url)
    return parse_apparition_result(payload["result"], designation)


def _lookup_alias_years(designation: str) -> list[int]:
    parameters = {"sstr": designation, "group": "com"}
    url = HORIZONS_LOOKUP_API + "?" + urllib.parse.urlencode(parameters)
    payload = _get_json(url)
    if payload.get("count", 0) != 1:
        return []
    aliases = payload["result"][0].get("alias", [])
    years = set()
    for alias in aliases:
        match = re.match(r"^(-?\d{1,4})(?:\s|$)", alias)
        if match is not None:
            years.add(int(match.group(1)))
    return sorted(years)


def collect_comet_apparitions(
    designation: str,
    start_year: int,
    stop_year: int,
    return_years: list[int] | tuple[int, ...] | None = None,
) -> list[CometApparition]:
    """Collect and deduplicate Horizons apparition records."""
    if start_year >= stop_year:
        raise ValueError("start_year must be earlier than stop_year.")
    normalized = designation.strip().upper().replace("/HALLEY", "")
    if return_years is None and normalized in {"1P", "HALLEY"}:
        candidates = list(HALLEY_RETURN_YEARS)
    elif return_years is None:
        candidates = _lookup_alias_years(designation)
    else:
        candidates = list(return_years)
    candidates = [
        year for year in candidates if start_year <= year <= stop_year
    ]
    if not candidates:
        raise ValueError(
            "No candidate return years were found. Supply --return-years."
        )

    by_record: dict[int, CometApparition] = {}
    for year in sorted(set(candidates)):
        apparition = fetch_comet_apparition(designation, year + 1)
        if start_year <= apparition.return_year <= stop_year:
            by_record[apparition.record] = apparition
    return sorted(by_record.values(), key=lambda item: item.perihelion_jd_tdb)


def write_apparitions_csv(
    path: Path,
    apparitions: list[CometApparition],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(apparitions[0]))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(asdict(item) for item in apparitions)


def plot_orbit_evolution(
    path: Path,
    apparitions: list[CometApparition],
) -> None:
    years = np.array([item.return_year for item in apparitions])
    tp = np.array([item.perihelion_jd_tdb for item in apparitions])
    periods = np.diff(tp) / 365.25
    semimajor = np.array([item.semimajor_axis_au for item in apparitions])
    perihelion = np.array([item.perihelion_au for item in apparitions])

    figure, axes = plt.subplots(3, 1, figsize=(11.2, 9.4), sharex=True)
    change_index = int(np.argmax(np.abs(np.diff(semimajor))))
    change_start = years[change_index]
    change_stop = years[change_index + 1]
    axes[0].plot(
        years[1:],
        periods,
        color="#B55220",
        marker="o",
        lw=1.8,
        label="interval between perihelia",
    )
    axes[0].axhline(
        76.0,
        color="#245AA6",
        ls="--",
        lw=1.3,
        label="fixed 76-year approximation",
    )
    axes[0].set_ylabel("return interval [yr]")
    axes[0].legend(frameon=False)

    axes[1].plot(
        years,
        semimajor,
        color="#007C77",
        marker="o",
        label="semimajor axis",
    )
    axes[1].set_ylabel("semimajor axis [au]")
    axes[1].legend(frameon=False)

    axes[2].plot(
        years,
        perihelion,
        color="#B86B00",
        marker="s",
        label="perihelion distance",
    )
    axes[2].set_ylabel("perihelion distance [au]")
    axes[2].set_xlabel("perihelion return year")
    axes[2].legend(frameon=False)
    for axis in axes:
        axis.axvspan(
            change_start,
            change_stop,
            color="#B55220",
            alpha=0.09,
        )
        axis.grid(alpha=0.22)
    axes[1].annotate(
        (
            f"largest |delta a|: {change_start}-{change_stop}\n"
            f"{semimajor[change_index + 1] - semimajor[change_index]:+.3f} au"
        ),
        (
            change_stop,
            semimajor[change_index + 1],
        ),
        xytext=(10, -32),
        textcoords="offset points",
        color="#B55220",
    )
    figure.suptitle(
        f"{apparitions[0].designation} apparition-to-apparition orbit evolution"
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, facecolor="white")
    plt.close(figure)


def comet_sky_positions(
    designation: str,
    epochs: list[str],
    observer: str = "500@399",
) -> list[dict[str, object]]:
    """Return geocentric apparent positions and IAU constellation names."""
    times = Time(epochs, scale="utc")
    command = f"DES={designation.strip().rstrip(';')};CAP"
    table = Horizons(
        id=command,
        id_type=None,
        location=observer,
        epochs=list(times.jd),
    ).ephemerides(extra_precision=True)
    coordinates = SkyCoord(
        ra=np.asarray(table["RA"], dtype=float) * u.deg,
        dec=np.asarray(table["DEC"], dtype=float) * u.deg,
        frame="icrs",
    )
    constellations = coordinates.get_constellation(short_name=False)
    rows = []
    for index, epoch in enumerate(epochs):
        rows.append(
            {
                "epoch_utc": epoch,
                "jd_utc": float(times.jd[index]),
                "ra_deg": float(table["RA"][index]),
                "dec_deg": float(table["DEC"][index]),
                "heliocentric_distance_au": float(table["r"][index]),
                "observer_distance_au": float(table["delta"][index]),
                "constellation": str(constellations[index]),
            }
        )
    return rows


def write_sky_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_sky_positions(
    path: Path,
    designation: str,
    rows: list[dict[str, object]],
) -> None:
    ra = np.array([float(row["ra_deg"]) for row in rows])
    dec = np.array([float(row["dec_deg"]) for row in rows])
    unwrapped_ra = np.degrees(np.unwrap(np.radians(ra)))
    figure, axis = plt.subplots(figsize=(11.0, 6.8))
    colors = np.linspace(0.0, 1.0, len(rows))
    axis.plot(unwrapped_ra, dec, color="#007C77", lw=1.4, alpha=0.75)
    points = axis.scatter(
        unwrapped_ra,
        dec,
        c=colors,
        cmap="plasma",
        s=54,
        zorder=3,
    )
    for x, y, row in zip(unwrapped_ra, dec, rows, strict=True):
        label = f"{row['epoch_utc']}  {row['constellation']}"
        axis.annotate(
            label,
            (x, y),
            xytext=(-6, 6),
            textcoords="offset points",
            ha="right",
        )
    axis.invert_xaxis()
    axis.set(
        xlabel="right ascension [deg, increasing to the left]",
        ylabel="declination [deg]",
        title=f"{designation} apparent sky positions",
    )
    axis.grid(alpha=0.22)
    figure.colorbar(points, ax=axis, label="epoch order")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, facecolor="white")
    plt.close(figure)
