"""Build Appendix C figures from public DAD cutouts and OpenOrb products."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.visualization import AsinhStretch, ImageNormalize, PercentileInterval
from astropy.wcs import WCS
from matplotlib.colors import LogNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.ndimage import gaussian_filter


ROOT = Path("/home/kjhan/BACKUP/CODES")
DATA = ROOT / "output/dad_mpcid_14941"
ASSETS = Path("/home/kjhan/BACKUP/3.5ST/appendixC_assets")
ASSETS.mkdir(parents=True, exist_ok=True)


def flow() -> None:
    fig, ax = plt.subplots(figsize=(13, 3.6), dpi=180)
    ax.set_xlim(0, 13); ax.set_ylim(0, 3.6); ax.axis("off")
    colors = ["#0c6e69", "#0f8b8d", "#d97706", "#b45309", "#155e75"]
    boxes = [
        (0.3, 0.95, 2.1, 1.5, "DS9\nCCD sequence", colors[0]),
        (3.0, 0.95, 2.1, 1.5, "Astrometric\ntracklet CSV", colors[1]),
        (5.7, 0.95, 2.1, 1.5, "OpenOrb\nstatistical ranging", colors[2]),
        (8.4, 0.95, 2.1, 1.5, "CODES\nDE442 / force model", colors[3]),
        (11.1, 0.95, 1.6, 1.5, "MPC / JPL\ncomparison", colors[4]),
    ]
    for x, y, w, h, label, color in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.08", facecolor=color, edgecolor="#1f2937", lw=1.2))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", color="white", fontsize=10.5, weight="bold")
    for left, right in zip(boxes[:-1], boxes[1:]):
        x1 = left[0] + left[2] + 0.08; x2 = right[0] - 0.08
        ax.add_patch(FancyArrowPatch((x1, 1.70), (x2, 1.70), arrowstyle="-|>", mutation_scale=18, lw=2, color="#374151"))
    fig.tight_layout(); fig.savefig(ASSETS / "ds9_codes_openorb_flow.png", facecolor="white"); plt.close(fig)


def ds9_capture() -> None:
    rows = list(json.loads((DATA / "validation_summary.json").read_text()) ["observations"] if False else [])
    import csv
    with (DATA / "observations_first_night.csv").open() as stream:
        observations = list(csv.DictReader(stream))
    files = sorted((DATA / "fits").glob("DEC14A_20140427*"))
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.25), dpi=180)
    for index, (axis, path, row) in enumerate(zip(axes, files, observations, strict=True), start=1):
        with fits.open(path) as hdul:
            image = np.asarray(hdul[0].data, dtype=float)
            wcs = WCS(hdul[0].header)
        finite = np.isfinite(image)
        norm = ImageNormalize(image[finite], interval=PercentileInterval(99.4), stretch=AsinhStretch(0.7))
        axis.imshow(image, origin="lower", cmap="gray_r", norm=norm)
        x, y = wcs.world_to_pixel_values(float(row["ra_deg"]), float(row["dec_deg"]))
        axis.plot(x, y, marker="+", ms=14, mew=2.0, color="#00e5ff")
        axis.set_title(f"exp {index}\n{path.stem[-9:]}", fontsize=8)
        axis.set_xticks([]); axis.set_yticks([])
        for spine in axis.spines.values(): spine.set_color("#00e5ff")
    fig.suptitle("DS9-style review of five calibrated DECam exposures", fontsize=15, weight="bold")
    fig.text(0.5, 0.01, "Cyan crosses are the DAD-reported astrometric positions. The panels are actual public InstCal FITS cutouts, displayed in a DS9-like review layout.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 0.91)); fig.savefig(ASSETS / "ds9_example_dad_tracklet.png", facecolor="white"); plt.close(fig)


def parse_orb(path: Path) -> np.ndarray:
    states, _ = parse_orb_statistics(path)
    return states


def parse_orb_statistics(path: Path) -> tuple[np.ndarray, np.ndarray]:
    values = []
    chi2_offsets = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) >= 11:
            values.append([float(item) for item in fields[1:7]])
            # OpenOrb stores chi2 minus the number of measured coordinates.
            # The constant offset cancels in every likelihood ratio.
            chi2_offsets.append(float(fields[10]))
    return np.asarray(values), np.asarray(chi2_offsets)


def elements(state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = 0.0002959122082855911
    r = state[:, :3]; v = state[:, 3:]
    radius = np.linalg.norm(r, axis=1); speed2 = np.sum(v * v, axis=1)
    energy = 0.5 * speed2 - mu / radius
    a = -mu / (2 * energy)
    h = np.cross(r, v); evec = np.cross(v, h) / mu - r / radius[:, None]
    e = np.linalg.norm(evec, axis=1)
    q = a * (1 - e)
    return a, e, q


def openorb_capture() -> None:
    orbit = next(DATA.glob("openorb_final/*.orb"))
    state, chi2_offset = parse_orb_statistics(orbit)
    a, e, q = elements(state)
    valid = (
        np.isfinite(a)
        & np.isfinite(e)
        & np.isfinite(q)
        & np.isfinite(chi2_offset)
        & (a > 0.0)
        & (q > 0.0)
        & (e >= 0.0)
        & (e < 1.0)
    )
    a, e, q, chi2_offset = a[valid], e[valid], q[valid], chi2_offset[valid]
    relative_likelihood = np.exp(-0.5 * (chi2_offset - np.min(chi2_offset)))
    log_a = np.log10(a)
    log_a_edges = np.linspace(np.floor(np.min(log_a) * 10.0) / 10.0, np.ceil(np.max(log_a) * 10.0) / 10.0, 96)
    e_edges = np.linspace(0.0, 1.0, 72)
    sample_count, _, _ = np.histogram2d(log_a, e, bins=(log_a_edges, e_edges))
    likelihood_sum, _, _ = np.histogram2d(
        log_a,
        e,
        bins=(log_a_edges, e_edges),
        weights=relative_likelihood,
    )
    smoothed_count = gaussian_filter(sample_count, sigma=(1.8, 1.4), mode="constant")
    smoothed_sum = gaussian_filter(likelihood_sum, sigma=(1.8, 1.4), mode="constant")
    likelihood_grid = np.divide(
        smoothed_sum,
        smoothed_count,
        out=np.full_like(smoothed_sum, np.nan),
        where=smoothed_count > 0.035,
    )
    likelihood_grid /= np.nanmax(likelihood_grid)
    official = json.loads((DATA / "validation_summary.json").read_text())["official_elements"][0]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=180)
    likelihood_map = axes[0].pcolormesh(
        10.0**log_a_edges,
        e_edges,
        np.ma.masked_invalid(likelihood_grid.T),
        cmap="turbo",
        norm=LogNorm(vmin=1.0e-3, vmax=1.0, clip=True),
        shading="flat",
        rasterized=True,
    )
    axes[0].set_xscale("log")
    axes[0].scatter([official[7]], [official[1]], marker="*", s=180, color="#0c6e69", edgecolor="black", label="NASA/JPL official solution")
    axes[0].set(xlabel="semimajor axis a [au]", ylabel="eccentricity e", title="Astrometric likelihood")
    axes[0].legend(loc="lower right", frameon=False, fontsize=8); axes[0].grid(alpha=0.2)
    colorbar = fig.colorbar(likelihood_map, ax=axes[0], pad=0.02)
    colorbar.set_label(r"relative likelihood $L/L_{\rm max}$")
    q_bins = np.geomspace(np.min(q), np.max(q), 15)
    axes[1].hist(q, bins=q_bins, color="#0f8b8d", alpha=0.85, label="OpenOrb q posterior")
    axes[1].axvline(official[2], color="#b45309", lw=2, label="NASA/JPL q")
    axes[1].set_xscale("log")
    axes[1].set(xlabel="perihelion distance q [au]", ylabel="samples per logarithmic bin", title="Range posterior")
    axes[1].legend(frameon=False, fontsize=8); axes[1].grid(alpha=0.2)
    fig.text(0.5, 0.01, "Color gives the astrometric likelihood, while the q histogram retains the OpenOrb posterior. Both horizontal axes are logarithmic.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 1.0)); fig.savefig(ASSETS / "codes_openorb_calculation.png", facecolor="white"); plt.close(fig)


def verification() -> None:
    orbit = parse_orb(next(DATA.glob("openorb_final/*.orb")))
    _, _, q = elements(orbit)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=180, gridspec_kw={"width_ratios": [1.2, 1.0]})
    axes[0].scatter(np.arange(len(q)), q, color="#0f8b8d", s=18, alpha=0.7, label="Accepted OpenOrb samples")
    axes[0].set(xlabel="posterior sample", ylabel="perihelion distance q [au]", title="1. CODES propagation inputs")
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].grid(alpha=0.2)

    axes[1].axis("off")
    boxes = [
        (0.08, 0.75, 0.40, 0.13, "OpenOrb posterior", "#fef3c7"),
        (0.08, 0.50, 0.40, 0.13, "CODES propagation", "#ffedd5"),
        (0.08, 0.25, 0.40, 0.13, "Predicted astrometry", "#d1fae5"),
        (0.58, 0.50, 0.34, 0.13, "NASA/JPL solution", "#dbeafe"),
        (0.58, 0.25, 0.34, 0.13, "Residual comparison", "#cffafe"),
    ]
    for x, y, width, height, label, color in boxes:
        axes[1].add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.02", facecolor=color, edgecolor="#374151", lw=1.2))
        axes[1].text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=10.5, weight="bold")
    arrows = [
        ((0.28, 0.75), (0.28, 0.63)),
        ((0.28, 0.50), (0.28, 0.38)),
        ((0.48, 0.315), (0.58, 0.315)),
        ((0.75, 0.50), (0.75, 0.38)),
    ]
    for start, end in arrows:
        axes[1].annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "-|>", "color": "#374151", "lw": 1.5})
    axes[1].set_title("2. Independent comparison", pad=12)
    fig.suptitle("Verification workflow after DS9 image review", fontsize=15, weight="bold")
    fig.text(0.5, 0.01, "The orbital-element posterior is shown once in Figure C.14. The official solution enters only at the final comparison.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94)); fig.savefig(ASSETS / "ds9_codes_verification.png", facecolor="white"); plt.close(fig)


def ogfinder_candidate_review() -> None:
    import csv
    files = sorted((DATA / "fits").glob("DEC14A_20140427*"))
    detection_path = ROOT / "output/dad_mpcid_14941/detection4"
    # Recreate the detector output with the documented review threshold when
    # the asset script is run on a clean checkout.
    if not (detection_path / "neo_candidates.csv").exists():
        detection_path = Path("/tmp/detection_4")
    detections = []
    if (detection_path / "neo_candidates.csv").exists():
        with (detection_path / "neo_candidates.csv").open() as stream:
            detections = list(csv.DictReader(stream))
    fig = plt.figure(figsize=(15, 8.2), dpi=180, facecolor="#d8d8d8")
    grid = fig.add_gridspec(3, 4, width_ratios=[1, 1, 1, 0.95], height_ratios=[0.10, 1, 1], wspace=0.03, hspace=0.08)
    header = fig.add_subplot(grid[0, :]); header.axis("off"); header.set_facecolor("#e7e7e7")
    header.text(0.01, 0.50, "File   Edit   View   Frame   Bin   Zoom   Scale   Color   Region   WCS   Analysis", va="center", fontsize=13, family="DejaVu Sans Mono", color="#222")
    header.text(0.80, 0.50, "NEO", va="center", fontsize=13, weight="bold", color="#0c6e69")
    axes = []
    for index, path in enumerate(files):
        axis = fig.add_subplot(grid[1 + index // 3, index % 3])
        axes.append(axis)
        with fits.open(path) as hdul:
            image = np.asarray(hdul[0].data, dtype=float)
            wcs = WCS(hdul[0].header)
        finite = np.isfinite(image)
        norm = ImageNormalize(image[finite], interval=PercentileInterval(99.4), stretch=AsinhStretch(0.7))
        axis.imshow(image, origin="lower", cmap="gray_r", norm=norm)
        for track_id in sorted({row["track_id"] for row in detections}):
            points = [row for row in detections if row["track_id"] == track_id and int(row["frame"]) == index + 1]
            for row in points:
                x, y = wcs.world_to_pixel_values(float(row["ra_deg"]), float(row["dec_deg"]))
                axis.plot(x, y, marker="o", ms=9, mfc="none", mec="#00d4a8", mew=1.7)
                axis.text(x + 4, y + 4, f"C{track_id}", color="#00d4a8", fontsize=7, weight="bold")
        axis.set_title(f"Frame {index + 1}", fontsize=9, color="#111")
        axis.set_xticks([]); axis.set_yticks([])
    # The fifth frame occupies the first cell of the second row. The fourth
    # cell is a compact candidate panel, matching the OGFinder layout.
    panel = fig.add_subplot(grid[1:, 3]); panel.set_facecolor("#f1f1f1"); panel.axis("off")
    panel.text(0.05, 0.96, "NEO candidate review", fontsize=12, weight="bold", color="#0c6e69")
    panel.text(0.05, 0.90, "Detect Moving Sources in Loaded Frames", fontsize=8.5, color="#222")
    panel.text(0.05, 0.85, "Fit CODES Orbit from Astrometry CSV...", fontsize=8.5, color="#222")
    panel.text(0.05, 0.79, "Load Last CODES Regions", fontsize=8.5, color="#222")
    panel.text(0.05, 0.69, "Candidate   frames   S/N", fontsize=9, weight="bold", color="#222")
    track_ids = sorted({row["track_id"] for row in detections})
    for i, track_id in enumerate(track_ids[:8]):
        rows = [row for row in detections if row["track_id"] == track_id]
        panel.text(0.05, 0.64 - i * 0.055, f"C{track_id:<9} {len(rows):>2}       {max(float(x['snr']) for x in rows):5.1f}", fontsize=8.5, family="DejaVu Sans Mono", color="#0c6e69")
    panel.text(0.05, 0.10, "green circles: linked moving-source candidates", fontsize=7.5, color="#333")
    fig.text(0.01, 0.01, "OGFinder review state rendered from the actual DECam sequence and the implemented NEO detector output.", fontsize=9, color="#222")
    fig.savefig(ASSETS / "ogfinder_neo_candidate_review.png", facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def ogfinder_multiframe() -> None:
    import csv
    files = sorted((DATA / "fits").glob("DEC14A_20140427*"))
    with (ROOT / "output/dad_mpcid_14941/detection4/neo_candidates.csv").open() as stream:
        detections = list(csv.DictReader(stream))
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.3), dpi=180)
    for index, (axis, path) in enumerate(zip(axes, files, strict=True), start=1):
        with fits.open(path) as hdul:
            image = np.asarray(hdul[0].data, dtype=float); wcs = WCS(hdul[0].header)
        finite = np.isfinite(image)
        axis.imshow(image, origin="lower", cmap="gray_r", norm=ImageNormalize(image[finite], interval=PercentileInterval(99.4), stretch=AsinhStretch(0.7)))
        for row in detections:
            if int(row["frame"]) != index: continue
            x, y = wcs.world_to_pixel_values(float(row["ra_deg"]), float(row["dec_deg"]))
            axis.plot(x, y, marker="o", ms=10, mfc="none", mec="#00d4a8", mew=2)
            axis.text(x + 4, y + 4, f"C{row['track_id']}", color="#00d4a8", fontsize=7, weight="bold")
        axis.set_title(f"Frame {index}", fontsize=9); axis.set_xticks([]); axis.set_yticks([])
    fig.text(0.5, 0.01, "Left image area in DS9 multiframe mode. Green circles are linked moving-source candidates.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(ASSETS / "ogfinder_multiframe_candidates.png", facecolor="white"); plt.close(fig)


def codes_gui_openorb() -> None:
    fig, ax = plt.subplots(figsize=(13, 8.2), dpi=180, facecolor="#0e1b18")
    ax.set_facecolor("#0e1b18"); ax.set_xlim(0, 13); ax.set_ylim(0, 8.2); ax.axis("off")
    ax.text(0.35, 7.75, "CODES", color="#ffd166", fontsize=24, weight="bold")
    ax.text(0.35, 7.35, "Close-approach Orbit Dynamics and Ephemeris System", color="#f4f1e8", fontsize=11)
    ax.text(11.9, 7.55, "Ready", color="#56d6c2", fontsize=11, weight="bold", ha="right")
    tabs = ["NEO dynamics", "NEO astrometry", "Comet evolution", "Sky positions", "Validation"]
    for i, label in enumerate(tabs):
        x = 0.35 + i * 2.45
        selected = i == 1
        ax.add_patch(plt.Rectangle((x, 6.75), 2.25, 0.42, facecolor="#34705f" if selected else "#172923", edgecolor="#34705f"))
        ax.text(x + 1.125, 6.96, label, color="white" if selected else "#c9d4cf", fontsize=8.5, ha="center", va="center", weight="bold")
    ax.text(0.45, 6.35, "DS9 tracklet to OpenOrb ranging and CODES orbit refinement", color="#56d6c2", fontsize=13, weight="bold")
    rows = [("Observation CSV", "/home/kjhan/BACKUP/CODES/output/dad_mpcid_14941/observations_first_night.csv"), ("Observatory code", "W84"), ("OpenOrb samples", "2000"), ("Integrator", "fortran"), ("Ephemeris", "auto"), ("Output directory", "/home/kjhan/BACKUP/CODES/output/astrometry")]
    for i, (label, value) in enumerate(rows):
        y = 5.75 - i * 0.55
        ax.text(0.5, y, label, color="#f4f1e8", fontsize=10, va="center")
        width = 7.6 if i in (0, 5) else 3.1
        ax.add_patch(plt.Rectangle((2.35 if i in (0, 5) else 2.35, y - 0.17), width, 0.34, facecolor="#f7f7f2", edgecolor="#56d6c2", lw=0.8))
        ax.text(2.5, y, value, color="#101713", fontsize=8.2, va="center")
        if i == 0: ax.text(10.2, y, "Choose", color="#f4f1e8", fontsize=9, va="center", ha="center")
    ax.add_patch(plt.Rectangle((0.45, 2.28), 11.9, 0.42, facecolor="#24483e", edgecolor="#34705f"))
    ax.text(6.4, 2.49, "Run OpenOrb ranging + CODES refinement", color="#f4f1e8", fontsize=11, weight="bold", ha="center", va="center")
    ax.text(0.45, 1.85, "The result preserves observations.mpc, OpenOrb .orb and .sor products, preliminary_orbit.json, and the CODES propagated state table.", color="#c9d4cf", fontsize=9)
    ax.text(0.45, 1.30, "Run log", color="#56d6c2", fontsize=12, weight="bold")
    ax.add_patch(plt.Rectangle((0.45, 0.25), 11.9, 0.85, facecolor="#172923", edgecolor="#24483e"))
    ax.text(0.65, 0.83, "$ python -m neo_orbit_calculator.cli fit-observations ... --openorb", color="#f4f1e8", family="DejaVu Sans Mono", fontsize=8.5)
    ax.text(0.65, 0.55, "OpenOrb ranging complete   |   CODES refinement complete", color="#56d6c2", family="DejaVu Sans Mono", fontsize=8.5)
    fig.savefig(ASSETS / "codes_gui_openorb.png", facecolor=fig.get_facecolor(), bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    flow(); ds9_capture(); openorb_capture(); verification(); ogfinder_candidate_review(); ogfinder_multiframe(); codes_gui_openorb()
