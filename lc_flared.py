"""
SoLEXS LC Plotter with HEK Flare Region Shading
=================================================
Plots SoLEXS light curves from .lc files and overlays coloured background
regions for each flare from the HEK CSV, colour-coded by GOES class
(A/B/C/M/X only).

Bugs fixed in this version:
  1. lc_date int64 vs str mismatch               → dtype={"lc_date": str}
  2. timezone-naive CSV times vs aware LC times   → utc=True on parse
  3. Rogue entries with duration > 3 hours        → filtered in load_hek_csv
     (HEK sometimes has entries spanning days/weeks due to bad end-time
      estimates — these stretch x-axis to weeks and wash out the LC plot)
  4. Flare times crossing midnight                → clamped to lc_date's
     00:00:00–23:59:59 UTC window
  5. axvspan stretching x-axis beyond LC range    → ax.set_xlim enforced

Usage:
    python plot_lc_with_flares.py

Dependencies:
    pip install astropy matplotlib pandas
"""

import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path
from astropy.io import fits
from datetime import timezone, datetime


# ── Configuration ─────────────────────────────────────────────────────────────
LC_DIR             = "data/lcfiles"
HEK_CSV            = "data/hek_flares.csv"
OUTPUT_DIR         = Path("data/lc_flared_plots")
FILES_PER_PAGE     = 4
DPI                = 150
MAX_FLARE_DURATION = 3.0      # hours — entries longer than this are HEK artefacts

# Y-axis scaling options
# GLOBAL_YLIM = None   → script computes global max from all LC files (default)
# GLOBAL_YLIM = (0, N) → use this fixed range (override, e.g. (0, 5000))
# YLIM_PERCENTILE = 100.0 → true global max (no cap)
# YLIM_PERCENTILE = 99.9  → cap at 99.9th percentile (suppresses single spikes)
GLOBAL_YLIM      = None
YLIM_PERCENTILE  = 100.0
# ─────────────────────────────────────────────────────────────────────────────

CLASS_COLORS = {
    "A": "#aec6cf",
    "B": "#90ee90",
    "C": "#fffacd",
    "M": "#ffcc99",
    "X": "#ff9999",
}
CLASS_ORDER = ["A", "B", "C", "M", "X"]


def goes_letter(cls_str):
    if not isinstance(cls_str, str) or len(cls_str) == 0:
        return "?"
    letter = cls_str[0].upper()
    return letter if letter in CLASS_COLORS else "?"


# ─────────────────────────────────────────────────────────────────────────────
# Load and clean HEK CSV
# ─────────────────────────────────────────────────────────────────────────────
def load_hek_csv(csv_path: str) -> pd.DataFrame:
    """
    Fix 1 — lc_date dtype:
        Pandas reads integer-looking columns as int64. Cast to str so
        string comparisons like df["lc_date"] == "20250603" work.

    Fix 2 — timezone-aware timestamps:
        CSV times are stored as naive strings. utc=True makes them
        timezone-aware so axvspan comparisons with LC datetimes work.

    Fix 3 — rogue long-duration entries:
        Some HEK entries have end times days/weeks after start
        (automated algorithms sometimes leave end-time unclosed).
        These produce a single axvspan covering the entire x-axis,
        making the background a solid colour with no LC detail visible.
        Drop any entry whose duration exceeds MAX_FLARE_DURATION hours.
    """
    df = pd.read_csv(csv_path, dtype={"lc_date": str})          # Fix 1

    for col in ["event_starttime", "event_endtime", "event_peaktime"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")   # Fix 2

    df = df.dropna(subset=["event_starttime", "event_endtime"])

    # Fix 3 — drop rogue long-duration entries
    df["_duration_h"] = (
        (df["event_endtime"] - df["event_starttime"])
        .dt.total_seconds() / 3600
    )
    n_before  = len(df)
    df        = df[df["_duration_h"] <= MAX_FLARE_DURATION].copy()
    n_removed = n_before - len(df)
    df.drop(columns=["_duration_h"], inplace=True)

    df["goes_letter"] = df["fl_goescls"].apply(goes_letter)
    df = df[df["goes_letter"] != "?"]

    print(f"  Loaded {len(df)} flare events "
          f"({n_removed} rogue long-duration entries removed) "
          f"| classes: {df['goes_letter'].value_counts().to_dict()}")
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Extract date string from filename
# ─────────────────────────────────────────────────────────────────────────────
_LC_PATTERN = re.compile(r"AL1_SOLEXS_(\d{8})_SDD\d_L1\.lc", re.IGNORECASE)

def date_from_lc(filename: Path):
    m = _LC_PATTERN.search(filename.name)
    return m.group(1) if m else None


# ─────────────────────────────────────────────────────────────────────────────
# Read .lc file  (TIME column is Unix seconds UTC)
# ─────────────────────────────────────────────────────────────────────────────
def read_lc(filepath: Path):
    with fits.open(filepath) as hdul:
        data     = hdul[1].data
        time_raw = data["TIME"].astype(float)

        time_dt = [
            datetime.fromtimestamp(t, tz=timezone.utc)
            for t in time_raw
        ]

        if "RATE" in data.columns.names:
            counts = data["RATE"].astype(float)
            ylabel = "Count Rate (cts/s)"
        elif "COUNTS" in data.columns.names:
            counts = data["COUNTS"].astype(float)
            ylabel = "Counts / s"
        else:
            raise KeyError("No RATE or COUNTS column found.")

    return time_dt, counts, ylabel


def compute_global_ylim(lc_files: list, percentile: float = 100.0) -> tuple:
    """
    First pass over all LC files to find the global count range.
    Uses nanpercentile to handle NaN values and suppress single spikes
    if percentile < 100.

    Returns (y_min, y_max) to use as shared ylim across all plots.
    """
    import numpy as np
    all_maxes = []
    all_mins  = []
    print(f"  Computing global y-axis range (percentile={percentile}) ...")

    for f in lc_files:
        try:
            with fits.open(f) as hdul:
                data = hdul[1].data
                col  = "RATE" if "RATE" in data.columns.names else "COUNTS"
                vals = data[col].astype(float)
                all_maxes.append(np.nanmax(vals))
                all_mins.append(np.nanmin(vals))
        except Exception:
            pass

    if not all_maxes:
        return (0, 1000)

    global_max = float(np.percentile(all_maxes, percentile))
    global_min = float(min(0, np.nanmin(all_mins)))   # always start at 0 or below
    y_max      = global_max * 1.05                    # 5% headroom above max

    print(f"  Global count range: {global_min:.1f} → {global_max:.1f}  "
          f"(ylim will be {global_min:.1f} → {y_max:.1f})")
    return (global_min, y_max)


# ─────────────────────────────────────────────────────────────────────────────
# Get flares for a date with day-boundary clamping (Fix 4)
# ─────────────────────────────────────────────────────────────────────────────
def flares_for_date(hek_df: pd.DataFrame, date_str: str) -> pd.DataFrame:
    """
    Fix 4 - midnight-crossing flares:
        Some flares start just before midnight (23:58) and end just after
        (00:05 next day). Clamp start/end to the 24-hour window of lc_date.

    Fix 5 - peak time outside LC day window:
        Some HEK entries have peak time on the PREVIOUS day (e.g. flare
        peaks at 23:57 on Oct 4 but end crosses into Oct 5, so HEK assigns
        it to Oct 5's lc_date). ax.text() given a datetime outside the axes
        x-range maps it to x=0 via get_xaxis_transform, printing the class
        letter jammed at the top-left corner of the plot.
        Fix: if peak is outside [day_start, day_end], replace it with the
        midpoint of the clamped (start, end) span.
    """
    rows = hek_df[hek_df["lc_date"] == date_str].copy()
    if rows.empty:
        return rows

    day_start = pd.Timestamp(
        f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} 00:00:00", tz="UTC"
    )
    day_end = pd.Timestamp(
        f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} 23:59:59", tz="UTC"
    )

    # Clamp span to LC day window
    rows["event_starttime"] = rows["event_starttime"].clip(lower=day_start, upper=day_end)
    rows["event_endtime"]   = rows["event_endtime"].clip(lower=day_start,   upper=day_end)

    # Drop entries that collapsed to zero width after clamping
    rows = rows[rows["event_endtime"] > rows["event_starttime"]]

    # Fix 5: replace out-of-day peak times with midpoint of clamped span
    def safe_peak(row):
        pk = row["event_peaktime"]
        if pd.isna(pk) or pk < day_start or pk > day_end:
            return row["event_starttime"] + (
                row["event_endtime"] - row["event_starttime"]
            ) / 2
        return pk

    rows["event_peaktime"] = rows.apply(safe_peak, axis=1)

    return rows.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Shade flare regions
# ─────────────────────────────────────────────────────────────────────────────
def shade_flares(ax, flares: pd.DataFrame):
    for _, row in flares.iterrows():
        letter = row["goes_letter"]
        color  = CLASS_COLORS.get(letter, "#dddddd")
        t0     = row["event_starttime"].to_pydatetime()
        t1     = row["event_endtime"].to_pydatetime()

        ax.axvspan(t0, t1, color=color, alpha=0.35, zorder=0, linewidth=0)
        ax.axvline(t0, color=color, lw=0.6, alpha=0.7, zorder=1)
        ax.axvline(t1, color=color, lw=0.6, alpha=0.7, zorder=1)


# ─────────────────────────────────────────────────────────────────────────────
# Time axis formatting
# ─────────────────────────────────────────────────────────────────────────────
def apply_time_axis(ax, time_dt: list):
    duration_s = (time_dt[-1] - time_dt[0]).total_seconds()

    if duration_s <= 600:
        fmt    = mdates.DateFormatter("%H:%M:%S")
        xlabel = f"Time (UTC)  [{time_dt[0].strftime('%Y-%m-%d')}]"
    elif duration_s <= 86400:
        fmt    = mdates.DateFormatter("%H:%M")
        xlabel = f"Time (UTC)  [{time_dt[0].strftime('%Y-%m-%d')}]"
    elif duration_s <= 86400 * 7:
        fmt    = mdates.DateFormatter("%b %d %H:%M")
        xlabel = "Date & Time (UTC)"
    else:
        fmt    = mdates.DateFormatter("%Y-%m-%d")
        xlabel = "Date (UTC)"

    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(fmt)
    ax.set_xlabel(xlabel, fontsize=7)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)


# ─────────────────────────────────────────────────────────────────────────────
# Legend
# ─────────────────────────────────────────────────────────────────────────────
def make_legend_handles(classes_present: set) -> list:
    handles = []
    for cls in CLASS_ORDER:
        if cls in classes_present:
            handles.append(mpatches.Patch(
                facecolor=CLASS_COLORS[cls], edgecolor="grey",
                alpha=0.6, label=f"{cls}-class flare"
            ))
    handles.append(Line2D([0], [0], color="#0077b6", lw=0.8, label="SoLEXS LC"))
    return handles


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 60)
    print("  SoLEXS LC Plotter with Flare Shading")
    print("=" * 60)

    print(f"\n[1/3] Loading HEK CSV: {HEK_CSV}")
    hek_df = load_hek_csv(HEK_CSV)

    print(f"\n[2/3] Scanning LC files: {LC_DIR}")
    lc_files = sorted(Path(LC_DIR).rglob("*.lc"))
    if not lc_files:
        print(f"  ERROR: No .lc files found in {LC_DIR}")
        return
    print(f"  Found {len(lc_files)} .lc files")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    n_pages = (len(lc_files) + FILES_PER_PAGE - 1) // FILES_PER_PAGE

    # Compute shared y-axis range (first pass over all files)
    if GLOBAL_YLIM is not None:
        ylim = GLOBAL_YLIM
        print(f"\n  Using fixed ylim: {ylim}")
    else:
        ylim = compute_global_ylim(lc_files, YLIM_PERCENTILE)

    print(f"\n[3/3] Generating plots -> {OUTPUT_DIR}\n")

    for page_idx, start_idx in enumerate(range(0, len(lc_files), FILES_PER_PAGE)):
        batch     = lc_files[start_idx : start_idx + FILES_PER_PAGE]
        fig, axes = plt.subplots(2, 2, figsize=(16, 9))
        axes      = axes.flatten()
        classes_in_batch = set()

        for ax, filepath in zip(axes, batch):
            date_str = date_from_lc(filepath)

            try:
                time_dt, counts, ylabel = read_lc(filepath)
                flares = (
                    flares_for_date(hek_df, date_str)
                    if date_str else pd.DataFrame()
                )

                if not flares.empty:
                    classes_in_batch.update(flares["goes_letter"].unique())

                # LC curve on top
                ax.plot(time_dt, counts, lw=0.7, color="#0077b6", zorder=2)

                # Shaded flare regions behind curve
                if not flares.empty:
                    shade_flares(ax, flares)

                    # Class letter at peak time
                    for _, row in flares.iterrows():
                        if pd.isna(row.get("event_peaktime")):
                            continue
                        t_peak = row["event_peaktime"].to_pydatetime()
                        ax.text(
                            t_peak, 0.97,
                            row["goes_letter"],
                            transform=ax.get_xaxis_transform(),
                            ha="center", va="top",
                            fontsize=7, fontweight="bold",
                            color="black", alpha=0.8, zorder=3,
                        )

                n_fl = len(flares)
                ax.set_title(
                    f"{filepath.name}\n({n_fl} flare(s) annotated)",
                    fontsize=7.5, pad=3
                )
                ax.set_ylabel(ylabel, fontsize=8)
                ax.grid(True, alpha=0.25, zorder=1)
                apply_time_axis(ax, time_dt)

                # Lock x-axis to LC data range (prevents axvspan from stretching it)
                ax.set_xlim(time_dt[0], time_dt[-1])

                # Apply shared y-axis scale for cross-file comparability
                ax.set_ylim(ylim)

            except Exception as e:
                ax.text(0.5, 0.5, f"Error:\n{e}",
                        ha="center", va="center", fontsize=8,
                        transform=ax.transAxes)
                ax.set_title(filepath.name if filepath else "?", fontsize=7.5)

        for ax in axes[len(batch):]:
            ax.axis("off")

        legend_handles = make_legend_handles(classes_in_batch)
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=len(legend_handles),
            fontsize=8,
            framealpha=0.8,
            bbox_to_anchor=(0.5, -0.01),
        )

        page_num = page_idx + 1
        fig.suptitle(
            f"SoLEXS Light Curves with HEK Flare Annotations  "
            f"(Page {page_num}/{n_pages})",
            fontsize=10, fontweight="bold", y=1.01
        )
        plt.tight_layout(rect=[0, 0.04, 1, 1])

        outfile = OUTPUT_DIR / f"lc_annotated_{page_num:03d}.png"
        plt.savefig(outfile, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  Page {page_num:>3}/{n_pages} -> {outfile.name}  "
              f"({len(batch)} plots, classes shaded: {sorted(classes_in_batch)})")

    print(f"\nDone! Saved to: {OUTPUT_DIR}\n")


if __name__ == "__main__":
    main()