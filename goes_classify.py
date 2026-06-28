import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from astropy.io import fits

from scipy.signal import find_peaks

from sunpy.net import Fido, attrs as a
import sunpy.timeseries as ts

# GOES FLARE CLASSIFICATION  (1–8 Å band, W/m²)
GOES_CLASSES = [
    ("X", 1e-4),
    ("M", 1e-5),
    ("C", 1e-6),
    ("B", 1e-7),
    ("A", 0.0),
]

def goes_class(flux_wm2: float) -> str:
    """Return GOES flare class string (e.g. 'M2.3') for a given 1-8 Å flux."""
    for letter, threshold in GOES_CLASSES:
        if flux_wm2 >= threshold:
            sub = flux_wm2 / threshold
            return f"{letter}{sub:.1f}"
    return "sub-A"


# STEP 1 – PARSE THE .lc FILENAME TO EXTRACT DATE AND DETECTOR
def parse_lc_filename(lc_path: Path):
    pattern = r"AL1_SOLEXS_(\d{8})_(SDD\d)_L1"
    match = re.search(pattern, lc_path.name, re.IGNORECASE)
    if not match:
        raise ValueError(
            f"Filename '{lc_path.name}' does not match expected pattern "
            "AL1_SOLEXS_YYYYMMDD_SDDX_L1.lc"
        )
    date_str, detector = match.group(1), match.group(2).upper()
    obs_date = datetime.strptime(date_str, "%Y%m%d").replace(tzinfo=timezone.utc)
    return obs_date, detector


# STEP 2 – READ THE SoLEXS LIGHT CURVE
def read_lc(lc_path: Path):
    with fits.open(lc_path) as hdul:
        data = hdul[1].data
        unix_time = data["TIME"].astype(float)
        if "COUNTS" in data.columns.names:
            counts = data["COUNTS"].astype(float)
            ylabel = "Counts / s"
        elif "RATE" in data.columns.names:
            counts = data["RATE"].astype(float)
            ylabel = "Count Rate (cts/s)"
        else:
            raise KeyError("Neither COUNTS nor RATE column found in LC file.")

    # Convert Unix → UTC datetime
    times_utc = np.array([
        datetime.fromtimestamp(t, tz=timezone.utc) for t in unix_time
    ])
    return times_utc, counts, ylabel


# STEP 3 – FETCH GOES-18 XRS DATA
def fetch_goes_xrs(obs_date: datetime):
    t_start = obs_date.strftime("%Y-%m-%d")
    t_end   = (obs_date + timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"  Fetching GOES-18 XRS data for {t_start} …")
    for sat_num in [18, 17, 16]:
        try:
            result = Fido.search(
                a.Time(t_start, t_end),
                a.Instrument("XRS"),
                a.goes.SatelliteNumber(sat_num),
            )
            if len(result["xrs"]) == 0:
                print(f"    GOES-{sat_num}: no results, trying next …")
                continue
            files = Fido.fetch(result)
            goes_ts = ts.TimeSeries(files, source="XRS", concatenate=True)
            print(f"    ✓ GOES-{sat_num} data fetched.")
            return goes_ts, sat_num
        except Exception as exc:
            print(f"    GOES-{sat_num} failed: {exc}")

    raise RuntimeError("Could not fetch XRS data from GOES-18, 17, or 16.")


# STEP 4a – FLARE DETECTION: SIMPLE THRESHOLD METHOD
def detect_flares_threshold(times, counts, sigma=5, min_duration_sec=60):
    bg   = np.nanpercentile(counts, 20)
    std  = np.nanstd(counts[counts <= np.nanpercentile(counts, 80)])
    threshold = bg + sigma * std

    in_flare  = False
    flares    = []
    f_start   = None
    f_peak_idx = None

    for i, (t, c) in enumerate(zip(times, counts)):
        if c > threshold:
            if not in_flare:
                in_flare   = True
                f_start    = i
                f_peak_idx = i
            elif c > counts[f_peak_idx]:
                f_peak_idx = i
        else:
            if in_flare:
                # Check minimum duration
                duration = (times[i - 1] - times[f_start]).total_seconds()
                if duration >= min_duration_sec:
                    flares.append({
                        "start":       times[f_start],
                        "peak":        times[f_peak_idx],
                        "end":         times[i - 1],
                        "peak_counts": counts[f_peak_idx],
                        "start_idx":   f_start,
                        "peak_idx":    f_peak_idx,
                        "end_idx":     i - 1,
                    })
                in_flare = False

    # Close an open flare at end of data
    if in_flare:
        duration = (times[-1] - times[f_start]).total_seconds()
        if duration >= min_duration_sec:
            flares.append({
                "start":       times[f_start],
                "peak":        times[f_peak_idx],
                "end":         times[-1],
                "peak_counts": counts[f_peak_idx],
                "start_idx":   f_start,
                "peak_idx":    f_peak_idx,
                "end_idx":     len(times) - 1,
            })

    return flares, threshold, bg


# STEP 4b – FLARE DETECTION: SCIPY PEAK-FINDING METHOD
def detect_flares_peaks(times, counts, prominence_factor=5, width_sec=60):

    bg         = np.nanpercentile(counts, 20)
    std        = np.nanstd(counts[counts <= np.nanpercentile(counts, 80)])
    prominence = prominence_factor * std

    # Estimate median cadence in seconds
    dt_sec = np.median(np.diff([t.timestamp() for t in times]))
    min_width_samples = max(1, int(width_sec / dt_sec))

    peaks_idx, props = find_peaks(
        counts,
        prominence=prominence,
        width=min_width_samples,
    )

    flares = []
    for idx in peaks_idx:
        # Define flare window as half-prominence drop on each side
        half_prom = props["prominences"][list(peaks_idx).index(idx)] / 2
        local_thresh = counts[idx] - half_prom

        # Walk left for start
        s = idx
        while s > 0 and counts[s] > local_thresh:
            s -= 1

        # Walk right for end
        e = idx
        while e < len(counts) - 1 and counts[e] > local_thresh:
            e += 1

        flares.append({
            "start":       times[s],
            "peak":        times[idx],
            "end":         times[e],
            "peak_counts": counts[idx],
            "start_idx":   s,
            "peak_idx":    idx,
            "end_idx":     e,
        })

    return flares, prominence, bg


# STEP 5 – CLASSIFY FLARES USING GOES XRS
def classify_flares_with_goes(flares, goes_ts):

    goes_df   = goes_ts.to_dataframe()
    # SunPy XRS column names vary; try common ones
    long_col  = None
    for candidate in ["xrsb", "xrsb_flux", "b_flux", "xrsa", "long"]:
        if candidate in goes_df.columns:
            long_col = candidate
            break

    if long_col is None:
        print(f"  Available GOES columns: {list(goes_df.columns)}")
        print("  WARNING: Could not find 1-8 Å (long) channel. "
              "Using first available column for classification.")
        long_col = goes_df.columns[0]

    # Ensure index is timezone-aware UTC
    if goes_df.index.tz is None:
        goes_df.index = goes_df.index.tz_localize("UTC")

    goes_times  = goes_df.index.to_pydatetime()
    goes_flux   = goes_df[long_col].values.astype(float)

    for flare in flares:
        peak_t  = flare["peak"]
        # Find nearest GOES timestamp
        deltas  = np.abs([(peak_t - gt).total_seconds() for gt in goes_times])
        nearest = np.argmin(deltas)
        flux    = goes_flux[nearest]

        # Also find peak GOES flux in the flare window ±5 min
        mask = np.array([
            abs((t - peak_t).total_seconds()) <= 300
            for t in goes_times
        ])
        if mask.any():
            flux = np.nanmax(goes_flux[mask])

        flare["goes_flux"]  = flux
        flare["goes_class"] = goes_class(flux)

    return flares, long_col


# STEP 6 – PLOTTING
FLARE_COLORS = ["#e63946", "#f4a261", "#2a9d8f", "#8338ec", "#fb5607"]

def _shade_flares(ax, flares, alpha=0.15):
    for i, f in enumerate(flares):
        ax.axvspan(f["start"], f["end"],
                   color=FLARE_COLORS[i % len(FLARE_COLORS)],
                   alpha=alpha, zorder=0)
        ax.axvline(f["peak"], color=FLARE_COLORS[i % len(FLARE_COLORS)],
                   lw=1.2, ls="--", alpha=0.8)


def _flare_legend(flares):
    handles = []
    for i, f in enumerate(flares):
        lbl = (f"Flare {i+1}: {f.get('goes_class','?')}  "
               f"peak {f['peak'].strftime('%H:%M:%S')} UTC")
        handles.append(Patch(color=FLARE_COLORS[i % len(FLARE_COLORS)],
                             alpha=0.5, label=lbl))
    return handles


def plot_combined(lc_times, lc_counts, lc_ylabel,
                  goes_ts, goes_long_col,
                  flares_thresh, flares_peaks,
                  obs_date, detector, sat_num,
                  threshold_val, peaks_prominence, filename):

    #Single figure with 3 stacked panels:
      #Panel 1 – SoLEXS LC + threshold detections
      #Panel 2 – SoLEXS LC + peak-finding detections
      #Panel 3 – GOES XRS 1-8 Å flux with flare class labels

    goes_df   = goes_ts.to_dataframe()
    if goes_df.index.tz is None:
        goes_df.index = goes_df.index.tz_localize("UTC")
    goes_times = goes_df.index.to_pydatetime()
    goes_flux  = goes_df[goes_long_col].values.astype(float)

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=False)
    fig.suptitle(
        f"SoLEXS Flare Analysis  |  {obs_date.strftime('%Y-%m-%d')}  |  "
        f"{detector}  |  GOES-{sat_num} XRS",
        fontsize=13, fontweight="bold", y=0.99
    )

    date_fmt = mdates.DateFormatter("%H:%M")

    # ── Panel 1: Threshold method ──────────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(lc_times, lc_counts, lw=0.7, color="#0077b6", label="SoLEXS LC")
    ax1.axhline(threshold_val, color="red", lw=1, ls=":", label=f"Threshold (5σ)")
    _shade_flares(ax1, flares_thresh)
    ax1.set_ylabel(lc_ylabel, fontsize=9)
    ax1.set_title("Method 1 — Threshold (background + 5σ)", fontsize=10)
    ax1.legend(handles=ax1.get_legend_handles_labels()[0] +
               _flare_legend(flares_thresh),
               fontsize=7.5, loc="upper right")
    ax1.grid(True, alpha=0.25)
    ax1.xaxis.set_major_formatter(date_fmt)
    ax1.set_xlabel("Time (UTC)", fontsize=8)

    # ── Panel 2: Peak-finding method ──────────────────────────────────────
    ax2 = axes[1]
    ax2.plot(lc_times, lc_counts, lw=0.7, color="#023e8a", label="SoLEXS LC")
    for f in flares_peaks:
        ax2.plot(f["peak"], f["peak_counts"], "v",
                 color="darkorange", ms=8, zorder=5)
    _shade_flares(ax2, flares_peaks)
    ax2.set_ylabel(lc_ylabel, fontsize=9)
    ax2.set_title("Method 2 — SciPy Peak Finder (prominence + width)", fontsize=10)
    ax2.legend(handles=[Line2D([0], [0], color="#023e8a", lw=0.7, label="SoLEXS LC"),
                         Line2D([0], [0], marker="v", color="darkorange",
                                ls="None", ms=8, label="Detected peaks")] +
               _flare_legend(flares_peaks),
               fontsize=7.5, loc="upper right")
    ax2.grid(True, alpha=0.25)
    ax2.xaxis.set_major_formatter(date_fmt)
    ax2.set_xlabel("Time (UTC)", fontsize=8)

    # ── Panel 3: GOES XRS ─────────────────────────────────────────────────
    ax3 = axes[2]
    ax3.semilogy(goes_times, np.clip(goes_flux, 1e-9, None),
                 lw=0.8, color="#6a0572", label=f"GOES-{sat_num} XRS 1–8 Å")

    # Draw GOES class boundaries
    class_bounds = {"X": 1e-4, "M": 1e-5, "C": 1e-6, "B": 1e-7, "A": 1e-8}
    for cls, val in class_bounds.items():
        ax3.axhline(val, color="grey", lw=0.6, ls="--", alpha=0.6)
        ax3.text(goes_times[0], val * 1.15, cls, fontsize=8,
                 color="grey", va="bottom")

    # Mark flare peaks on GOES panel (use threshold detections as reference)
    all_flares = {**{i: f for i, f in enumerate(flares_thresh)}}
    for i, f in enumerate(flares_thresh):
        ax3.axvline(f["peak"], color=FLARE_COLORS[i % len(FLARE_COLORS)],
                    lw=1.2, ls="--", alpha=0.8,
                    label=f"Flare {i+1} peak → {f.get('goes_class','?')}")

    ax3.set_ylabel("Flux (W/m²)", fontsize=9)
    ax3.set_title(f"GOES-{sat_num} XRS 1–8 Å  (official classification reference)",
                  fontsize=10)
    ax3.legend(fontsize=7.5, loc="upper right")
    ax3.grid(True, alpha=0.25, which="both")
    ax3.xaxis.set_major_formatter(date_fmt)
    ax3.set_xlabel("Time (UTC)", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(f"plots/{filename}_combined_plot.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: plots/{filename}_combined_plot.png")


def plot_separate(lc_times, lc_counts, lc_ylabel,
                  goes_ts, goes_long_col,
                  flares_thresh, flares_peaks,
                  obs_date, detector, sat_num,
                  threshold_val, filename):

    goes_df = goes_ts.to_dataframe()
    if goes_df.index.tz is None:
        goes_df.index = goes_df.index.tz_localize("UTC")
    goes_times = goes_df.index.to_pydatetime()
    goes_flux  = goes_df[goes_long_col].values.astype(float)

    date_fmt  = mdates.DateFormatter("%H:%M")
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    fig.suptitle(
        f"SoLEXS vs GOES  |  {obs_date.strftime('%Y-%m-%d')}  |  "
        f"{detector}  |  GOES-{sat_num}",
        fontsize=12, fontweight="bold"
    )

    # ── [0,0] SoLEXS full day ─────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(lc_times, lc_counts, lw=0.7, color="#0077b6")
    ax.axhline(threshold_val, color="red", lw=0.8, ls=":", label="5σ threshold")
    _shade_flares(ax, flares_thresh)
    ax.set_title("SoLEXS LC — Full Day (Threshold method)", fontsize=9)
    ax.set_ylabel(lc_ylabel, fontsize=8)
    ax.set_xlabel("Time (UTC)", fontsize=8)
    ax.xaxis.set_major_formatter(date_fmt)
    ax.legend(handles=[Line2D([0], [0], color="red", ls=":", lw=0.8,
                               label="5σ threshold")] +
              _flare_legend(flares_thresh), fontsize=6.5, loc="upper right")
    ax.grid(True, alpha=0.25)

    # ── [0,1] GOES full day ───────────────────────────────────────────────
    ax = axes[0, 1]
    ax.semilogy(goes_times, np.clip(goes_flux, 1e-9, None),
                lw=0.8, color="#6a0572")
    for cls, val in {"X": 1e-4, "M": 1e-5, "C": 1e-6, "B": 1e-7}.items():
        ax.axhline(val, color="grey", lw=0.5, ls="--", alpha=0.6)
        ax.text(goes_times[0], val * 1.2, cls, fontsize=7, color="grey")
    for i, f in enumerate(flares_thresh):
        ax.axvline(f["peak"], color=FLARE_COLORS[i % len(FLARE_COLORS)],
                   lw=1, ls="--", alpha=0.8,
                   label=f"Flare {i+1} → {f.get('goes_class','?')}")
    ax.set_title(f"GOES-{sat_num} XRS 1–8 Å — Full Day", fontsize=9)
    ax.set_ylabel("Flux (W/m²)", fontsize=8)
    ax.set_xlabel("Time (UTC)", fontsize=8)
    ax.xaxis.set_major_formatter(date_fmt)
    ax.legend(fontsize=6.5, loc="upper right")
    ax.grid(True, alpha=0.25, which="both")

    # ── Find largest flare for zoom panels ────────────────────────────────
    zoom_flares = flares_thresh if flares_thresh else flares_peaks
    if zoom_flares:
        biggest = max(zoom_flares, key=lambda f: f["peak_counts"])
        zoom_start = biggest["start"] - timedelta(minutes=10)
        zoom_end   = biggest["end"]   + timedelta(minutes=10)

        # ── [1,0] SoLEXS zoom ─────────────────────────────────────────────
        ax = axes[1, 0]
        lc_mask = np.array([(zoom_start <= t <= zoom_end) for t in lc_times])
        ax.plot(lc_times[lc_mask], lc_counts[lc_mask], lw=1.0, color="#0077b6")
        ax.axvspan(biggest["start"], biggest["end"], alpha=0.15,
                   color=FLARE_COLORS[0])
        ax.axvline(biggest["peak"], color=FLARE_COLORS[0], lw=1.5, ls="--",
                   label=f"Peak  {biggest['peak'].strftime('%H:%M:%S')} UTC")
        ax.set_title(
            f"SoLEXS — Zoom: Largest Flare  "
            f"({biggest.get('goes_class','?')})", fontsize=9
        )
        ax.set_ylabel(lc_ylabel, fontsize=8)
        ax.set_xlabel("Time (UTC)", fontsize=8)
        ax.xaxis.set_major_formatter(date_fmt)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.25)

        # ── [1,1] GOES zoom ───────────────────────────────────────────────
        ax = axes[1, 1]
        goes_mask = np.array([
            (zoom_start <= t <= zoom_end) for t in goes_times
        ])
        if goes_mask.any():
            ax.semilogy(goes_times[goes_mask],
                        np.clip(goes_flux[goes_mask], 1e-9, None),
                        lw=1.0, color="#6a0572")
            for cls, val in {"X": 1e-4, "M": 1e-5, "C": 1e-6, "B": 1e-7}.items():
                ax.axhline(val, color="grey", lw=0.5, ls="--", alpha=0.6)
                ax.text(goes_times[goes_mask][0], val * 1.2, cls,
                        fontsize=7, color="grey")
            ax.axvline(biggest["peak"], color=FLARE_COLORS[0], lw=1.5, ls="--",
                       label=f"Peak  → {biggest.get('goes_class','?')}")
            ax.axvspan(biggest["start"], biggest["end"], alpha=0.1,
                       color=FLARE_COLORS[0])
        ax.set_title(
            f"GOES-{sat_num} XRS — Zoom: Largest Flare  "
            f"({biggest.get('goes_class','?')})", fontsize=9
        )
        ax.set_ylabel("Flux (W/m²)", fontsize=8)
        ax.set_xlabel("Time (UTC)", fontsize=8)
        ax.xaxis.set_major_formatter(date_fmt)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.25, which="both")
    else:
        axes[1, 0].text(0.5, 0.5, "No flares detected", ha="center", va="center")
        axes[1, 1].text(0.5, 0.5, "No flares detected", ha="center", va="center")

    plt.tight_layout()
    plt.savefig(f"plots/{filename}_separate_plots.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: plots/{filename}_separate_plots.png")


def print_summary(flares_thresh, flares_peaks, detector, obs_date):
    print("\n" + "═" * 65)
    print(f"  FLARE DETECTION SUMMARY  |  {obs_date.strftime('%Y-%m-%d')}  |  {detector}")
    print("═" * 65)

    def show(flares, method):
        print(f"\n  ── {method} ({len(flares)} flare(s) detected) ──")
        if not flares:
            print("    No flares detected.")
            return
        for i, f in enumerate(flares, 1):
            cls   = f.get("goes_class", "N/A")
            flux  = f.get("goes_flux",  float("nan"))
            dur   = (f["end"] - f["start"]).total_seconds()
            print(f"    Flare {i}:")
            print(f"      Start  : {f['start'].strftime('%Y-%m-%d %H:%M:%S')} UTC")
            print(f"      Peak   : {f['peak'].strftime('%Y-%m-%d %H:%M:%S')} UTC")
            print(f"      End    : {f['end'].strftime('%Y-%m-%d %H:%M:%S')} UTC")
            print(f"      Duration: {dur:.0f} s  ({dur/60:.1f} min)")
            print(f"      SoLEXS peak counts : {f['peak_counts']:.1f} cts/s")
            print(f"      GOES 1-8Å flux     : {flux:.3e} W/m²")
            print(f"      ► GOES class       : {cls}")

    show(flares_thresh, "Threshold Method (5σ above background)")
    show(flares_peaks,  "Peak-Finder Method (scipy prominence)")


# MAIN
def analyse(lc_file: str):
    lc_path = Path(lc_file)
    if not lc_path.exists():
        raise FileNotFoundError(f"File not found: {lc_path}")

    print(f"\n{'═'*65}")
    print(f"  SoLEXS Flare Analysis Tool")
    print(f"{'═'*65}")
    print(f"  Input file : {lc_path.name}")


    obs_date, detector = parse_lc_filename(lc_path)
    print(f"  Date       : {obs_date.strftime('%Y-%m-%d')}")
    print(f"  Detector   : {detector}")

    print("\n[1/5] Reading SoLEXS light curve …")
    lc_times, lc_counts, lc_ylabel = read_lc(lc_path)
    print(f"      {len(lc_times)} time bins  |  "
          f"range {lc_times[0].strftime('%H:%M')}–{lc_times[-1].strftime('%H:%M')} UTC")


    print("\n[2/5] Fetching GOES XRS data …")
    goes_ts, sat_num = fetch_goes_xrs(obs_date)


    print("\n[3/5] Detecting flares …")
    flares_thresh, threshold_val, bg = detect_flares_threshold(lc_times, lc_counts)
    flares_peaks,  prominence, _     = detect_flares_peaks(lc_times, lc_counts)
    print(f"      Threshold method : {len(flares_thresh)} flare(s)  "
          f"[threshold = {threshold_val:.1f} cts/s]")
    print(f"      Peak-finder      : {len(flares_peaks)} flare(s)  "
          f"[prominence = {prominence:.1f} cts/s]")

    print("\n[4/5] Classifying with GOES XRS …")
    flares_thresh, goes_long_col = classify_flares_with_goes(flares_thresh, goes_ts)
    flares_peaks,  _             = classify_flares_with_goes(flares_peaks,  goes_ts)


    print_summary(flares_thresh, flares_peaks, detector, obs_date)

    print("[5/5] Generating plots …\n")
    plot_combined(
        lc_times, lc_counts, lc_ylabel,
        goes_ts, goes_long_col,
        flares_thresh, flares_peaks,
        obs_date, detector, sat_num,
        threshold_val, prominence, lc_path.name
    )
    # plot_separate(
    #     lc_times, lc_counts, lc_ylabel,
    #     goes_ts, goes_long_col,
    #     flares_thresh, flares_peaks,
    #     obs_date, detector, sat_num,
    #     threshold_val, lc_file.name
    # )

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Using Default file")
        analyse(r"data\lcfiles\AL1_SOLEXS_20240203_SDD2_L1.lc")
    else: analyse(sys.argv[1])