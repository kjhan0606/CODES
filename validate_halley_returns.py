"""Validate historical 1P/Halley returns and identify large orbit changes."""

from __future__ import annotations

import csv
import io
import json
import urllib.parse
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time

from neo_orbit_calculator.comets import (
    collect_comet_apparitions,
)

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "halley_return_validation.csv"
PROVENANCE_PATH = ROOT / "halley_return_provenance.json"
FIGURE_PATH = ROOT / "appendixC_assets" / "halley_apparition_history.png"
TEX_PATH = ROOT / "halley_return_validation_table.tex"
AU_KM = 149_597_870.700

# T_obs - T_calc in days from Yeomans & Kiang (1981), Table 5.
HISTORICAL_OBSERVED_MINUS_CALCULATED_DAYS = {
    1531: -0.44,
    1456: -0.53,
    1378: -1.67,
    1301: -1.05,
    1222: +1.98,
    1145: +2.69,
    1066: +2.57,
    989: +3.31,
    912: -9.17,
    837: +1.83,
}

NASA_RETURN_YEARS = {1986, 1910, 1835, 1759}


def plot_validation(
    rows: list[dict[str, object]],
    apparitions: list[object],
) -> None:
    years = np.array([int(row["return_year"]) for row in rows])
    intervals = np.array(
        [float(row["return_interval_year"]) for row in rows]
    )
    semimajor = np.array(
        [float(row["semimajor_axis_au"]) for row in rows]
    )
    perihelion = np.array(
        [float(item.perihelion_au) for item in apparitions]
    )
    reference_interval = np.array(
        [float(row["reference_return_interval_year"]) for row in rows]
    )
    reference_type = np.array(
        [str(row["reference_type"]) for row in rows]
    )

    figure, axes = plt.subplots(
        3,
        1,
        figsize=(11.2, 9.4),
        sharex=True,
    )
    axes[0].plot(
        years[1:],
        intervals[1:],
        color="#B55220",
        marker="o",
        lw=1.8,
        label="CODES-assembled Horizons sequence",
    )
    historical = (
        np.isfinite(reference_interval)
        & (reference_type == "historical observation")
    )
    official = reference_type == "NASA/JPL official"
    official_interval = np.where(
        np.isfinite(reference_interval),
        reference_interval,
        intervals,
    )
    axes[0].scatter(
        years[historical],
        reference_interval[historical],
        marker="*",
        s=105,
        color="#245AA6",
        edgecolor="white",
        linewidth=0.6,
        zorder=4,
        label="observation-constrained returns",
    )
    axes[0].scatter(
        years[official],
        official_interval[official],
        marker="D",
        s=76,
        facecolor="#FFD166",
        edgecolor="#151B23",
        linewidth=1.2,
        zorder=6,
        label="NASA/JPL official return epochs",
    )
    axes[0].axhline(
        76.0,
        color="#5C6875",
        ls="--",
        lw=1.2,
        label="fixed 76-year approximation",
    )
    axes[0].set_ylabel("return interval [yr]")
    axes[0].legend(frameon=False, ncol=2, fontsize=8.5)

    change_index = int(np.argmax(np.abs(np.diff(semimajor))))
    change_start = years[change_index]
    change_stop = years[change_index + 1]
    axes[1].plot(
        years,
        semimajor,
        color="#007C77",
        marker="o",
        label="CODES-assembled JPL semimajor axis",
    )
    axes[1].scatter(
        years[official],
        semimajor[official],
        marker="D",
        s=76,
        facecolor="#FFD166",
        edgecolor="#151B23",
        linewidth=1.2,
        zorder=6,
        label="NASA/JPL official values",
    )
    axes[1].set_ylabel("semimajor axis [au]")
    axes[1].legend(frameon=False)
    axes[1].annotate(
        (
            f"largest |delta a|: {change_start}-{change_stop}\n"
            f"{semimajor[change_index + 1] - semimajor[change_index]:+.3f} au"
        ),
        (change_stop, semimajor[change_index + 1]),
        xytext=(10, -32),
        textcoords="offset points",
        color="#B55220",
    )

    axes[2].plot(
        years,
        perihelion,
        color="#B86B00",
        marker="s",
        label="CODES-assembled JPL perihelion distance",
    )
    axes[2].scatter(
        years[official],
        perihelion[official],
        marker="D",
        s=76,
        facecolor="#FFD166",
        edgecolor="#151B23",
        linewidth=1.2,
        zorder=6,
        label="NASA/JPL official values",
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
    figure.suptitle("1P/Halley orbit solutions and observed return timing")
    figure.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=220, facecolor="white")
    plt.close(figure)


def _range_vectors(
    command: str,
    start: str,
    stop: str,
    step: str,
) -> tuple[np.ndarray, list[str]]:
    parameters = {
        "format": "json",
        "COMMAND": f"'{command}'",
        "OBJ_DATA": "'NO'",
        "MAKE_EPHEM": "'YES'",
        "EPHEM_TYPE": "'VECTORS'",
        "CENTER": "'500@0'",
        "START_TIME": f"'{start}'",
        "STOP_TIME": f"'{stop}'",
        "STEP_SIZE": f"'{step}'",
        "TIME_TYPE": "'TDB'",
        "VEC_TABLE": "'2'",
        "OUT_UNITS": "'KM-S'",
        "CSV_FORMAT": "'YES'",
        "REF_PLANE": "'FRAME'",
        "REF_SYSTEM": "'ICRF'",
        "VEC_CORR": "'NONE'",
    }
    url = (
        "https://ssd.jpl.nasa.gov/api/horizons.api?"
        + urllib.parse.urlencode(parameters)
    )
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "CODES-Halley-Validation/1.0"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if "error" in payload:
        raise RuntimeError(payload["error"])
    table = payload["result"].split("$$SOE", 1)[1].split("$$EOE", 1)[0]
    rows = []
    calendars = []
    for row in csv.reader(io.StringIO(table.strip())):
        rows.append([float(row[0]), *[float(value) for value in row[2:8]]])
        calendars.append(row[1].strip())
    return np.asarray(rows), calendars


def jupiter_perturbation_epoch() -> dict[str, object]:
    """Locate the minimum 1P--Jupiter-barycenter distance after 1066."""
    command = "NAME=Halley;CAP<1070"
    halley, _ = _range_vectors(
        command,
        "1066-Jan-01",
        "1145-Dec-31",
        "10 d",
    )
    jupiter, _ = _range_vectors(
        "5",
        "1066-Jan-01",
        "1145-Dec-31",
        "10 d",
    )
    distance = np.linalg.norm(
        halley[:, 1:4] - jupiter[:, 1:4],
        axis=1,
    )
    coarse_index = int(np.argmin(distance))
    coarse_jd = float(halley[coarse_index, 0])
    start = Time(coarse_jd - 40.0, format="jd", scale="tdb").iso[:10]
    stop = Time(coarse_jd + 40.0, format="jd", scale="tdb").iso[:10]
    halley, calendars = _range_vectors(command, start, stop, "6 h")
    jupiter, _ = _range_vectors("5", start, stop, "6 h")
    distance = np.linalg.norm(
        halley[:, 1:4] - jupiter[:, 1:4],
        axis=1,
    )
    index = int(np.argmin(distance))
    return {
        "calendar_tdb": calendars[index],
        "jd_tdb": float(halley[index, 0]),
        "distance_au": float(distance[index] / AU_KM),
        "halley_solution": "Horizons NAME=Halley;CAP<1070",
        "jupiter_target": "Horizons Jupiter barycenter 5",
    }


def build_rows() -> tuple[list[dict[str, object]], dict[str, object]]:
    apparitions = collect_comet_apparitions("1P", 800, 2000)
    rows = []
    for index, apparition in enumerate(apparitions):
        previous = apparitions[index - 1] if index > 0 else None
        interval = (
            (apparition.perihelion_jd_tdb - previous.perihelion_jd_tdb)
            / 365.25
            if previous is not None
            else float("nan")
        )
        delta_a = (
            apparition.semimajor_axis_au - previous.semimajor_axis_au
            if previous is not None
            else float("nan")
        )
        observed_minus_calculated = (
            HISTORICAL_OBSERVED_MINUS_CALCULATED_DAYS.get(
                apparition.return_year
            )
        )
        if observed_minus_calculated is not None:
            reference_jd = (
                apparition.perihelion_jd_tdb
                + observed_minus_calculated
            )
            reference_type = "historical observation"
        elif apparition.return_year in NASA_RETURN_YEARS:
            reference_jd = apparition.perihelion_jd_tdb
            reference_type = "NASA/JPL official"
        else:
            reference_jd = float("nan")
            reference_type = "none"
        previous_reference_jd = (
            float(rows[-1]["reference_perihelion_jd_tdb"])
            if rows
            else float("nan")
        )
        reference_interval = (
            (reference_jd - previous_reference_jd) / 365.25
            if np.isfinite(reference_jd)
            and np.isfinite(previous_reference_jd)
            else float("nan")
        )
        rows.append(
            {
                "return_year": apparition.return_year,
                "horizons_record": apparition.record,
                "perihelion_calendar_tdb": apparition.perihelion_calendar,
                "perihelion_jd_tdb": apparition.perihelion_jd_tdb,
                "return_interval_year": interval,
                "semimajor_axis_au": apparition.semimajor_axis_au,
                "delta_semimajor_axis_au": delta_a,
                "osculating_period_year": apparition.osculating_period_year,
                "reference_perihelion_jd_tdb": reference_jd,
                "reference_return_interval_year": reference_interval,
                "reference_type": reference_type,
                "calculated_minus_historical_days": (
                    -observed_minus_calculated
                    if observed_minus_calculated is not None
                    else float("nan")
                ),
                "comparison_source": (
                    "NASA/JPL public return record"
                    if apparition.return_year in NASA_RETURN_YEARS
                    else "Yeomans and Kiang (1981)"
                ),
            }
        )

    finite = [
        (index, abs(float(row["delta_semimajor_axis_au"])))
        for index, row in enumerate(rows)
        if np.isfinite(float(row["delta_semimajor_axis_au"]))
    ]
    change_index = max(finite, key=lambda item: item[1])[0]
    strongest = {
        "from_year": int(rows[change_index - 1]["return_year"]),
        "to_year": int(rows[change_index]["return_year"]),
        "delta_semimajor_axis_au": float(
            rows[change_index]["delta_semimajor_axis_au"]
        ),
        "delta_osculating_period_year": float(
            rows[change_index]["osculating_period_year"]
            - rows[change_index - 1]["osculating_period_year"]
        ),
        "following_return_interval_year": float(
            rows[change_index + 1]["return_interval_year"]
        ),
    }
    plot_validation(rows, apparitions)
    return rows, strongest


def write_csv(rows: list[dict[str, object]]) -> None:
    with CSV_PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _display_residual(row: dict[str, object]) -> str:
    residual = float(row["calculated_minus_historical_days"])
    if np.isfinite(residual):
        return f"{residual:+.2f}\\,d"
    return "NASA/JPL"


def _short_calendar(value: str) -> str:
    prefix, day = value.rsplit("-", 1)
    return f"{prefix}-{float(day):06.3f}"


def write_tex(rows: list[dict[str, object]]) -> None:
    selected_years = {
        1986,
        1910,
        1835,
        1759,
        1607,
        1531,
        1145,
        1066,
        989,
        912,
        837,
    }
    selected = [
        row for row in reversed(rows)
        if int(row["return_year"]) in selected_years
    ]
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{CODES reconstruction of selected 1P/Halley perihelia. "
        "The residual is the calculated time minus the historical "
        "observation-based time reported by Yeomans and Kiang "
        "\\cite{Horizons,YeomansKiang1981}.}",
        "\\begin{tabularx}{\\textwidth}{@{}P{14mm}P{44mm}P{29mm}P{28mm}Y@{}}",
        "\\toprule",
        "\\color{cGold}\\textbf{Year} &",
        "\\color{cGold}\\textbf{Horizons perihelion, TDB} &",
        "\\color{cGold}\\textbf{Return interval} &",
        "\\color{cGold}\\textbf{$\\Delta a$} &",
        "\\color{cGold}\\textbf{Comparison}\\\\",
        "\\midrule",
    ]
    for row in selected:
        interval = float(row["return_interval_year"])
        delta_a = float(row["delta_semimajor_axis_au"])
        interval_text = f"{interval:.3f}\\,yr" if np.isfinite(interval) else "--"
        delta_text = f"{delta_a:+.3f}\\,au" if np.isfinite(delta_a) else "--"
        lines.append(
            f"{row['return_year']} & "
            f"{_short_calendar(str(row['perihelion_calendar_tdb']))} & "
            f"{interval_text} & "
            f"{delta_text} & "
            f"{_display_residual(row)}\\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabularx}",
            "\\label{tab:halley-returns}",
            "\\end{table}",
        ]
    )
    TEX_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows, strongest = build_rows()
    jupiter = jupiter_perturbation_epoch()
    write_csv(rows)
    write_tex(rows)
    provenance = {
        "generated_utc": "2026-07-26",
        "trajectory_source": "NASA/JPL Horizons apparition records",
        "historical_source": (
            "Yeomans and Kiang 1981, MNRAS 197, 633, Table 5"
        ),
        "strongest_apparition_change": strongest,
        "jupiter_barycenter_minimum_after_1066": jupiter,
        "interpretation": (
            "A fixed 76-year clock is rejected. Ancient records use separate "
            "apparition solutions because the current JPL#75 continuous "
            "ephemeris is not available before 1599."
        ),
    }
    PROVENANCE_PATH.write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(provenance, indent=2))
    print(CSV_PATH)
    print(FIGURE_PATH)
    print(TEX_PATH)


if __name__ == "__main__":
    main()
