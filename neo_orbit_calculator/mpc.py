"""Minor Planet Center observation ingestion for validation cases."""

from __future__ import annotations

import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

MPC_OBSERVATION_API = "https://data.minorplanetcenter.net/api/get-obs"


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(
        timezone.utc
    )


def fetch_mpc_ades(
    designation: str,
    cache_dir: Path | str | None = None,
) -> list[dict]:
    """Fetch MPC ADES observations, optionally retaining the raw response."""
    cache_path = None
    if cache_dir is not None:
        cache_root = Path(cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        slug = designation.replace(" ", "_").replace("/", "_")
        cache_path = cache_root / f"{slug}.json"
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))

    body = json.dumps(
        {"desigs": [designation], "output_format": ["ADES_DF"]}
    ).encode("utf-8")
    request = urllib.request.Request(
        MPC_OBSERVATION_API,
        data=body,
        method="GET",
        headers={
            "Content-Type": "application/json",
            "User-Agent": "3.5ST-NEO-Orbit-Calculator/1.0",
        },
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        payload = json.loads(response.read().decode("utf-8"))
    observations = payload[0].get("ADES_DF", [])
    if not observations:
        raise RuntimeError(f"MPC returned no ADES observations for {designation}.")
    if cache_path is not None:
        cache_path.write_text(
            json.dumps(observations, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return observations


def summarize_mpc_ades(observations: list[dict]) -> dict[str, object]:
    """Summarize usable astrometry and its time span."""
    optical = [
        row
        for row in observations
        if row.get("Obstype") == "optical"
        and row.get("obstime")
        and row.get("ra") is not None
        and row.get("dec") is not None
    ]
    if not optical:
        raise ValueError("No optical RA/Dec observations were found.")
    times = sorted(_parse_utc(row["obstime"]) for row in optical)
    discovery_times = sorted(
        _parse_utc(row["obstime"])
        for row in optical
        if row.get("disc") == "*"
    )
    stations = {row.get("stn") for row in optical if row.get("stn")}
    return {
        "observation_count": len(optical),
        "station_count": len(stations),
        "first_observation_utc": times[0],
        "discovery_observation_utc": (
            discovery_times[0] if discovery_times else times[0]
        ),
        "discovery_marker_available": bool(discovery_times),
        "last_observation_utc": times[-1],
        "arc_days": (times[-1] - times[0]).total_seconds() / 86_400.0,
    }
