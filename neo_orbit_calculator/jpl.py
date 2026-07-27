"""NASA/JPL API access and SPICE kernel management."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HORIZONS_API = "https://ssd.jpl.nasa.gov/api/horizons.api"
CAD_API = "https://ssd-api.jpl.nasa.gov/cad.api"
KERNEL_URLS = {
    "gm_de440.tpc": (
        "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/gm_de440.tpc"
    ),
    "naif0012.tls": (
        "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls"
    ),
    "pck00011.tpc": (
        "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/"
        "pck/pck00011.tpc"
    ),
}
PLANETARY_KERNEL_URLS = {
    "de442.bsp": (
        "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/"
        "spk/planets/de442.bsp"
    ),
    "de441_part-1.bsp": (
        "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/"
        "spk/planets/de441_part-1.bsp"
    ),
    "de441_part-2.bsp": (
        "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/"
        "spk/planets/de441_part-2.bsp"
    ),
}
PLANETARY_KERNEL_MD5 = {
    "de442.bsp": "446656322267e7b819a26cb08a0d8718",
    "de441_part-1.bsp": "7e5fcf9ecb5d08e1ab70c049baa60cd3",
    "de441_part-2.bsp": "ad8dfa4e505ef0e3a5d587a5b4705632",
}

# Official SPK coverage in Julian Date, TDB. The DE441 parts overlap by
# 32 days so integrations crossing the split remain continuous.
DE442_START_JD_TDB = 2_287_184.5
DE442_STOP_JD_TDB = 2_688_976.5
DE441_START_JD_TDB = -3_100_015.5
DE441_STOP_JD_TDB = 8_000_016.5
DE441_PART1_STOP_JD_TDB = 2_440_432.5
DE441_PART2_START_JD_TDB = 2_440_400.5
SB441_N16_URL = (
    "https://ssd.jpl.nasa.gov/ftp/eph/small_bodies/"
    "asteroids_de441/sb441-n16.bsp"
)
JUP365_URL = (
    "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/"
    "spk/satellites/jup365.bsp"
)


@dataclass(frozen=True)
class PlanetaryEphemeris:
    """Selected JPL planetary ephemeris and the SPKs needed for an interval."""

    name: str
    kernel_names: tuple[str, ...]
    start_jd_tdb: float
    stop_jd_tdb: float
    reason: str


def select_planetary_ephemeris(
    start_jd_tdb: float,
    stop_jd_tdb: float,
    requested: str = "auto",
) -> PlanetaryEphemeris:
    """Select DE442 for its modern interval and DE441 for long-term work."""
    if stop_jd_tdb < start_jd_tdb:
        raise ValueError("stop_jd_tdb must not precede start_jd_tdb.")
    mode = requested.lower()
    if mode not in {"auto", "de442", "de441"}:
        raise ValueError("ephemeris must be 'auto', 'de442', or 'de441'.")

    inside_de442 = (
        DE442_START_JD_TDB <= start_jd_tdb
        and stop_jd_tdb <= DE442_STOP_JD_TDB
    )
    if mode == "auto":
        mode = "de442" if inside_de442 else "de441"
        reason = (
            "requested interval is inside the DE442 modern-epoch coverage"
            if mode == "de442"
            else "requested interval extends beyond the DE442 coverage"
        )
    else:
        reason = f"explicit {mode.upper()} selection"

    if mode == "de442":
        if not inside_de442:
            raise ValueError(
                "DE442 covers JD TDB "
                f"{DE442_START_JD_TDB:.1f} through "
                f"{DE442_STOP_JD_TDB:.1f}. Use DE441 for this interval."
            )
        return PlanetaryEphemeris(
            name="DE442",
            kernel_names=("de442.bsp",),
            start_jd_tdb=DE442_START_JD_TDB,
            stop_jd_tdb=DE442_STOP_JD_TDB,
            reason=reason,
        )

    if not (
        DE441_START_JD_TDB <= start_jd_tdb
        and stop_jd_tdb <= DE441_STOP_JD_TDB
    ):
        raise ValueError(
            "DE441 covers JD TDB "
            f"{DE441_START_JD_TDB:.1f} through "
            f"{DE441_STOP_JD_TDB:.1f}."
        )
    if stop_jd_tdb <= DE441_PART1_STOP_JD_TDB:
        kernel_names = ("de441_part-1.bsp",)
    elif start_jd_tdb >= DE441_PART2_START_JD_TDB:
        kernel_names = ("de441_part-2.bsp",)
    else:
        kernel_names = ("de441_part-1.bsp", "de441_part-2.bsp")
    return PlanetaryEphemeris(
        name="DE441",
        kernel_names=kernel_names,
        start_jd_tdb=DE441_START_JD_TDB,
        stop_jd_tdb=DE441_STOP_JD_TDB,
        reason=reason,
    )


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _download(
    destination: Path,
    url: str,
    expected_md5: str | None = None,
    timeout: int = 1800,
) -> Path:
    if destination.exists() and destination.stat().st_size > 0:
        if expected_md5 is None or _md5(destination) == expected_md5:
            return destination
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "3.5ST-NEO-Orbit-Calculator/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        with temporary.open("wb") as stream:
            while block := response.read(4 * 1024 * 1024):
                stream.write(block)
    if expected_md5 is not None and _md5(temporary) != expected_md5:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Checksum verification failed for {destination.name}.")
    temporary.replace(destination)
    return destination


def _get_json(url: str, timeout: int = 180) -> dict:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "3.5ST-NEO-Orbit-Calculator/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if "error" in payload:
        raise RuntimeError(payload["error"])
    return payload


def _horizons_url(parameters: dict[str, str]) -> str:
    return HORIZONS_API + "?" + urllib.parse.urlencode(parameters)


def normalize_command(designation: str) -> str:
    target = designation.strip()
    if not target:
        raise ValueError("A JPL small-body designation is required.")
    if target.endswith(";"):
        return target
    return target + ";"


def ensure_generic_kernels(kernel_dir: Path) -> dict[str, Path]:
    """Download shared time, gravity, and orientation kernels."""
    kernel_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, url in KERNEL_URLS.items():
        destination = kernel_dir / name
        paths[name] = _download(destination, url, timeout=300)
    return paths


def ensure_planetary_ephemeris(
    kernel_dir: Path,
    start_jd_tdb: float,
    stop_jd_tdb: float,
    requested: str = "auto",
) -> tuple[PlanetaryEphemeris, tuple[Path, ...]]:
    """Download and return the selected modern or long-term SPICE kernels."""
    kernel_dir.mkdir(parents=True, exist_ok=True)
    selection = select_planetary_ephemeris(
        start_jd_tdb,
        stop_jd_tdb,
        requested=requested,
    )
    paths = tuple(
        _download(
            kernel_dir / name,
            PLANETARY_KERNEL_URLS[name],
            expected_md5=PLANETARY_KERNEL_MD5[name],
        )
        for name in selection.kernel_names
    )
    return selection, paths


def ensure_sb441_n16(kernel_dir: Path) -> Path:
    """Download the JPL ephemerides for the 16 most massive main-belt bodies."""
    kernel_dir.mkdir(parents=True, exist_ok=True)
    destination = kernel_dir / "sb441-n16.bsp"
    if destination.exists() and destination.stat().st_size > 600_000_000:
        return destination
    temporary = destination.with_suffix(".bsp.part")
    request = urllib.request.Request(
        SB441_N16_URL,
        headers={"User-Agent": "3.5ST-NEO-Orbit-Calculator/1.0"},
    )
    with urllib.request.urlopen(request, timeout=900) as response:
        with temporary.open("wb") as stream:
            while block := response.read(4 * 1024 * 1024):
                stream.write(block)
    temporary.replace(destination)
    return destination


def ensure_jup365(kernel_dir: Path) -> Path:
    """Download the JPL Jupiter and satellite ephemeris when requested."""
    kernel_dir.mkdir(parents=True, exist_ok=True)
    destination = kernel_dir / "jup365.bsp"
    if destination.exists() and destination.stat().st_size > 50_000_000:
        return destination
    temporary = destination.with_suffix(".bsp.part")
    request = urllib.request.Request(
        JUP365_URL,
        headers={"User-Agent": "3.5ST-NEO-Orbit-Calculator/1.0"},
    )
    with urllib.request.urlopen(request, timeout=1800) as response:
        with temporary.open("wb") as stream:
            while block := response.read(4 * 1024 * 1024):
                stream.write(block)
    temporary.replace(destination)
    return destination


def horizons_vectors(
    designation: str,
    jd_tdb: np.ndarray | list[float],
    center: str = "500@0",
    uncertainty: bool = False,
) -> tuple[np.ndarray, str]:
    """Return geometric ICRF states from Horizons in km and km/s."""
    epochs = np.asarray(jd_tdb, dtype=float)
    if epochs.ndim != 1 or len(epochs) == 0:
        raise ValueError("jd_tdb must be a non-empty one-dimensional sequence.")
    if len(epochs) > 900:
        chunks = [
            horizons_vectors(
                designation,
                epochs[index : index + 900],
                center=center,
                uncertainty=uncertainty,
            )[0]
            for index in range(0, len(epochs), 900)
        ]
        return np.vstack(chunks), "NASA/JPL Horizons API"

    time_list = " ".join(f"'{value:.12f}'" for value in epochs)
    parameters = {
        "format": "json",
        "COMMAND": f"'{normalize_command(designation)}'",
        "OBJ_DATA": "'YES'",
        "MAKE_EPHEM": "'YES'",
        "EPHEM_TYPE": "'VECTORS'",
        "CENTER": f"'{center}'",
        "TLIST": time_list,
        "TLIST_TYPE": "'JD'",
        "TIME_TYPE": "'TDB'",
        "VEC_TABLE": f"'2{'x' if uncertainty else ''}'",
        "OUT_UNITS": "'KM-S'",
        "CSV_FORMAT": "'YES'",
        "REF_PLANE": "'FRAME'",
        "REF_SYSTEM": "'ICRF'",
        "VEC_CORR": "'NONE'",
    }
    payload = _get_json(_horizons_url(parameters))
    result = payload["result"]
    try:
        table = result.split("$$SOE", 1)[1].split("$$EOE", 1)[0]
    except IndexError as exc:
        raise RuntimeError("Horizons returned no vector table.") from exc

    rows = []
    for row in csv.reader(io.StringIO(table.strip())):
        if not row:
            continue
        values = [float(value) for value in row[2:8]]
        rows.append([float(row[0]), *values])
    states = np.asarray(rows, dtype=float)
    if len(states) != len(epochs):
        raise RuntimeError(
            f"Horizons returned {len(states)} rows for {len(epochs)} epochs."
        )
    source = payload.get("signature", {}).get("source", "NASA/JPL Horizons API")
    return states, source


def horizons_elements(
    designation: str,
    jd_tdb: np.ndarray | list[float],
    center: str = "500@10",
) -> tuple[np.ndarray, str]:
    """Return heliocentric ecliptic osculating elements from Horizons.

    Columns are JD TDB, eccentricity, perihelion distance [au],
    inclination, longitude of ascending node, argument of perihelion,
    mean anomaly [deg], and semimajor axis [au].
    """
    epochs = np.asarray(jd_tdb, dtype=float)
    if epochs.ndim != 1 or len(epochs) == 0:
        raise ValueError("jd_tdb must be a non-empty one-dimensional sequence.")
    if len(epochs) > 900:
        chunks = [
            horizons_elements(
                designation,
                epochs[index : index + 900],
                center=center,
            )[0]
            for index in range(0, len(epochs), 900)
        ]
        return np.vstack(chunks), "NASA/JPL Horizons API"

    parameters = {
        "format": "json",
        "COMMAND": f"'{normalize_command(designation)}'",
        "OBJ_DATA": "'YES'",
        "MAKE_EPHEM": "'YES'",
        "EPHEM_TYPE": "'ELEMENTS'",
        "CENTER": f"'{center}'",
        "TLIST": " ".join(f"'{value:.12f}'" for value in epochs),
        "TLIST_TYPE": "'JD'",
        "TIME_TYPE": "'TDB'",
        "OUT_UNITS": "'AU-D'",
        "CSV_FORMAT": "'YES'",
        "REF_PLANE": "'ECLIPTIC'",
        "REF_SYSTEM": "'ICRF'",
    }
    payload = _get_json(_horizons_url(parameters))
    result = payload["result"]
    try:
        table = result.split("$$SOE", 1)[1].split("$$EOE", 1)[0]
    except IndexError as exc:
        raise RuntimeError("Horizons returned no element table.") from exc

    rows = []
    for row in csv.reader(io.StringIO(table.strip())):
        if not row:
            continue
        # Horizons columns: JD, calendar, EC, QR, IN, OM, W, Tp,
        # N, MA, TA, A, AD, PR.
        rows.append(
            [
                float(row[0]),
                float(row[2]),
                float(row[3]),
                float(row[4]),
                float(row[5]),
                float(row[6]),
                float(row[9]),
                float(row[11]),
            ]
        )
    elements = np.asarray(rows, dtype=float)
    if len(elements) != len(epochs):
        raise RuntimeError(
            f"Horizons returned {len(elements)} rows for {len(epochs)} epochs."
        )
    source = payload.get("signature", {}).get("source", "NASA/JPL Horizons API")
    return elements, source


def jpl_close_approaches(
    designation: str,
    date_min: str,
    date_max: str,
    distance_max_au: float = 0.05,
) -> list[dict[str, str]]:
    """Return official JPL CNEOS close approaches to Earth."""
    parameters = {
        "des": designation.strip(),
        "date-min": date_min,
        "date-max": date_max,
        "dist-max": f"{distance_max_au:.8g}",
        "body": "Earth",
        "fullname": "true",
    }
    payload = _get_json(CAD_API + "?" + urllib.parse.urlencode(parameters))
    fields = payload.get("fields", [])
    return [
        dict(zip(fields, values, strict=True))
        for values in payload.get("data", [])
    ]


def download_horizons_spk(
    designation: str,
    start: str,
    stop: str,
    output_dir: Path,
) -> tuple[Path, str]:
    """Download a time-continuous JPL small-body SPK from Horizons."""
    parameters = {
        "format": "json",
        "COMMAND": f"'{normalize_command(designation)}'",
        "MAKE_EPHEM": "'YES'",
        "EPHEM_TYPE": "'SPK'",
        "OBJ_DATA": "'NO'",
        "START_TIME": f"'{start}'",
        "STOP_TIME": f"'{stop}'",
    }
    payload = _get_json(_horizons_url(parameters), timeout=600)
    encoded = payload.get("spk")
    if encoded is None:
        message = payload.get("result", "Horizons did not return an SPK file.")
        raise RuntimeError(message.strip())
    output_dir.mkdir(parents=True, exist_ok=True)
    spk_id = payload.get("spk_file_id", "horizons_small_body")
    destination = output_dir / f"{spk_id}_{start}_{stop}.bsp"
    destination.write_bytes(base64.b64decode(encoded))
    return destination, spk_id
