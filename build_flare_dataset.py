#!/usr/bin/env python3
"""
build_flare_dataset.py

Turns raw SoLEXS .lc (FITS light-curve) files + a HEK flare-event CSV into a
windowed, ML-ready feature table for solar-flare DETECTION (binary) and
CLASSIFICATION (A/B/C/M/X).

Design notes (why it's built this way)
---------------------------------------
1. TWO-STAGE FEATURE PIPELINE (1 Hz -> windowed):
   Per-second causal features (trailing background median/MAD, excess,
   sigma) are computed first on the raw 1 Hz stream, exactly the way a
   real-time system would see them (no look-ahead). Fixed-size windows are
   then aggregated from that stream. This means the SAME feature code can
   be reused verbatim in a real-time inference loop later.

2. BACKGROUND IS LOCAL & ADAPTIVE, NOT GLOBAL:
   Background counts drift 30-70 c/s over the solar cycle, so a fixed
   threshold is wrong. Background is a trailing rolling median (robust to
   short flare contamination) over BG_WINDOW_SEC (default 30 min), computed
   causally (only past data). A long-term (24 h) trailing median is added
   too, so the model has both "local" and "solar-cycle-scale" context for
   non-flare regions.

3. HEK IS LABEL-ONLY:
   HEK columns are used exclusively to build `label_binary` / `label_class`.
   No HEK-derived column (flux_estimate_wm2, goes class, timings) is ever
   written into the feature matrix. This file is also the place validation
   labels come from -- inference/deployment code should never import HEK.

4. THE lc_date ROLLOVER PROBLEM SOLVES ITSELF:
   Because we stitch all files into one continuous absolute-UTC-time
   series and label using `event_starttime`/`event_endtime` (real
   timestamps) instead of joining on `lc_date`, a flare HEK mis-filed as
   lc_date = N+1 still lands in the correct place on the real timeline.
   `lc_date` is only used for a QC cross-check (logged, not used for
   placement).

5. CONTIGUITY:
   Files are only concatenated into the same "segment" if their dates are
   exactly consecutive. Rolling background/feature windows never bridge a
   gap of missing days, so you don't get a phantom background jump/flare
   across a data hole. Small intra-day NaN gaps (2-4/day) are linearly
   interpolated up to a short limit; larger gaps are left NaN and flagged.

Usage
-----
    python build_flare_dataset.py \
        --lc-dir /path/to/lc_files \
        --hek-csv /path/to/hek_flares.csv \
        --out-dir /path/to/output \
        --window-sec 60 --step-sec 15 --bg-window-sec 1800

Outputs (in --out-dir)
-----------------------
    windows_features.parquet   -> the ML-ready table (features + labels)
    qc_report.txt              -> coverage / rollover / mismatch summary
"""

import argparse
import glob
import logging
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

try:
    from astropy.io import fits
except ImportError:
    sys.exit("This script needs astropy: pip install astropy --break-system-packages")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

EPOCH = datetime(1970, 1, 1)  # MJD 40587 == 1970-01-01 UTC, matches TIME column definition
FNAME_RE = re.compile(r"AL1_SOLEXS_(\d{8})_SDD2_L1\.lc$")

# ------------------------- config defaults -------------------------
DEFAULT_WINDOW_SEC = 60          # feature-aggregation window length
DEFAULT_STEP_SEC = 15            # stride between windows (overlapping)
DEFAULT_BG_WINDOW_SEC = 1800     # 30 min trailing background window
DEFAULT_LONG_BG_WINDOW_SEC = 86400  # 24 h trailing "solar-cycle-scale" background
DEFAULT_NAN_INTERP_LIMIT = 5     # seconds; only bridge short gaps
DEFAULT_OVERLAP_THRESHOLD = 0.0  # any overlap with a HEK interval -> positive window


# ============================== I/O ==============================

def discover_lc_files(lc_dir: str) -> pd.DataFrame:
    """Find *.lc files, parse their date from the filename, sort chronologically."""
    paths = glob.glob(os.path.join(lc_dir, "*.lc"))
    rows = []
    for p in paths:
        m = FNAME_RE.search(os.path.basename(p))
        if not m:
            log.warning("Skipping file with unexpected name pattern: %s", p)
            continue
        date = datetime.strptime(m.group(1), "%Y%m%d").date()
        rows.append({"path": p, "date": date})
    if not rows:
        raise FileNotFoundError(f"No AL1_SOLEXS_*_SDD2_L1.lc files found in {lc_dir}")
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    log.info("Discovered %d .lc files spanning %s to %s", len(df), df.date.min(), df.date.max())
    return df


def read_lc_file(path: str) -> pd.DataFrame:
    """Read one .lc FITS file -> DataFrame[datetime, counts, source_file]."""
    with fits.open(path) as hdul:
        table_hdu = None
        for hdu in hdul:
            if hasattr(hdu, "columns") and hdu.columns is not None:
                names = [c.upper() for c in hdu.columns.names]
                if "COUNTS" in names and "TIME" in names:
                    table_hdu = hdu
                    break
        if table_hdu is None:
            raise ValueError(f"Could not find TIME/COUNTS table in {path}")
        data = table_hdu.data
        time_col = [c for c in data.columns.names if c.upper() == "TIME"][0]
        counts_col = [c for c in data.columns.names if c.upper() == "COUNTS"][0]
        time_s = np.asarray(data[time_col], dtype="float64")
        counts = np.asarray(data[counts_col], dtype="float64")

    dt = pd.to_datetime(EPOCH) + pd.to_timedelta(time_s, unit="s")
    df = pd.DataFrame({"datetime": dt, "counts": counts})
    df["source_file"] = os.path.basename(path)
    return df


def build_segments(file_index: pd.DataFrame) -> list:
    """
    Group consecutive-day files into contiguous "segments". Rolling
    background/feature windows will only ever be computed within a segment,
    never bridging a missing-day gap.
    Returns list of DataFrames, each a stitched contiguous segment.
    """
    segments = []
    current_files = []
    prev_date = None

    for _, row in file_index.iterrows():
        if prev_date is not None and row["date"] != prev_date + timedelta(days=1):
            segments.append(current_files)
            current_files = []
        current_files.append(row["path"])
        prev_date = row["date"]
    if current_files:
        segments.append(current_files)

    log.info("Files grouped into %d contiguous segment(s)", len(segments))

    seg_dfs = []
    for i, files in enumerate(segments):
        parts = [read_lc_file(f) for f in files]
        seg = pd.concat(parts, ignore_index=True).sort_values("datetime").reset_index(drop=True)
        seg["segment_id"] = i
        seg_dfs.append(seg)
        log.info("  segment %d: %d files, %s -> %s (%d rows)",
                  i, len(files), seg.datetime.min(), seg.datetime.max(), len(seg))
    return seg_dfs


def clean_gaps(seg: pd.DataFrame, interp_limit_s: int = DEFAULT_NAN_INTERP_LIMIT) -> pd.DataFrame:
    """Linearly interpolate short NaN gaps (2-4/day expected); flag longer gaps."""
    seg = seg.set_index("datetime")
    n_nan_before = seg["counts"].isna().sum()
    seg["counts"] = seg["counts"].interpolate(method="time", limit=interp_limit_s, limit_area="inside")
    n_nan_after = seg["counts"].isna().sum()
    if n_nan_after > 0:
        log.warning("Segment %s: %d NaNs remain after interpolation (gap longer than %ds) "
                     "-- left as NaN, will propagate to NaN features",
                     seg["segment_id"].iloc[0], n_nan_after, interp_limit_s)
    log.info("Segment %s: interpolated %d/%d NaN samples",
              seg["segment_id"].iloc[0], n_nan_before - n_nan_after, n_nan_before)
    return seg.reset_index()


# ========================= HEK label prep =========================

def load_hek_flares(hek_csv: str) -> pd.DataFrame:
    df = pd.read_csv(hek_csv)
    df = df.drop(columns=["flux_estimate_wm2"], errors="ignore")  # never used as a feature/label input beyond class

    for col in ["event_starttime", "event_peaktime", "event_endtime"]:
        # strip explicit "+00:00"/tz info -> naive UTC, matching the .lc native timezone
        df[col] = pd.to_datetime(df[col], utc=True).dt.tz_localize(None)

    df["fl_class_letter"] = df["fl_goescls"].str[0].str.upper()
    df["fl_class_mult"] = pd.to_numeric(df["fl_goescls"].str[1:], errors="coerce")

    # QC: flag the lc_date rollover cases (flare start-date != lc_date it was filed under).
    # We don't need to "fix" lc_date because absolute timestamps are used for labeling,
    # but we log it so you can sanity-check the join assumption documented in the CSV.
    df["start_date_int"] = df["event_starttime"].dt.strftime("%Y%m%d").astype(int)
    df["lc_date_rollover_flag"] = df["start_date_int"] != df["lc_date"]

    n_roll = df["lc_date_rollover_flag"].sum()
    log.info("HEK: %d flare rows, %d flagged as lc_date rollover cases (handled automatically "
              "since labeling uses absolute timestamps, not lc_date joins)", len(df), n_roll)

    return df.sort_values("event_starttime").reset_index(drop=True)


# ===================== per-second feature engineering =====================

def add_causal_background_features(seg: pd.DataFrame,
                                     bg_window_s: int,
                                     long_bg_window_s: int) -> pd.DataFrame:
    """
    Add 1 Hz causal (trailing-only, no look-ahead) background features.
    Safe to reuse identically in a real-time stream.
    """
    seg = seg.set_index("datetime")
    counts = seg["counts"]

    bg_win = f"{bg_window_s}s"
    long_win = f"{long_bg_window_s}s"

    roll_bg = counts.rolling(bg_win, min_periods=max(30, bg_window_s // 20))
    seg["bg_median"] = roll_bg.median()
    mad = counts.rolling(bg_win, min_periods=max(30, bg_window_s // 20)).apply(
        lambda x: np.median(np.abs(x - np.median(x))), raw=True
    )
    seg["bg_mad_std"] = 1.4826 * mad  # MAD -> robust std equivalent
    seg["bg_mad_std"] = seg["bg_mad_std"].replace(0, np.nan)  # avoid div-by-zero

    seg["bg_long_median"] = counts.rolling(long_win, min_periods=long_bg_window_s // 10).median()

    seg["excess"] = counts - seg["bg_median"]
    seg["sigma_excess"] = seg["excess"] / seg["bg_mad_std"]
    seg["ratio_to_bg"] = counts / seg["bg_median"]
    seg["ratio_to_long_bg"] = seg["bg_median"] / seg["bg_long_median"]  # local vs solar-cycle-scale drift

    return seg.reset_index()


# ===================== HEK-derived per-second labels =====================

def add_flare_labels(seg: pd.DataFrame, hek: pd.DataFrame) -> pd.DataFrame:
    """Add is_flare_second (bool) and flare_class_second (str/None) via interval membership."""
    t = seg["datetime"].values.astype("datetime64[ns]")
    is_flare = np.zeros(len(seg), dtype=bool)
    flare_class = np.full(len(seg), None, dtype=object)
    flare_id = np.full(len(seg), -1, dtype=int)

    seg_start, seg_end = seg["datetime"].iloc[0], seg["datetime"].iloc[-1]
    relevant = hek[(hek["event_endtime"] >= seg_start) & (hek["event_starttime"] <= seg_end)]

    for idx, row in relevant.iterrows():
        mask = (t >= np.datetime64(row["event_starttime"])) & (t <= np.datetime64(row["event_endtime"]))
        is_flare |= mask
        flare_class[mask] = row["fl_class_letter"]
        flare_id[mask] = idx

    seg["is_flare_second"] = is_flare
    seg["flare_class_second"] = flare_class
    seg["flare_id_second"] = flare_id
    return seg


# ===================== windowing / aggregation =====================

def _slope(x: np.ndarray) -> float:
    valid = ~np.isnan(x)
    if valid.sum() < 3:
        return np.nan
    idx = np.arange(len(x))[valid]
    return np.polyfit(idx, x[valid], 1)[0]


def _first_second_half_delta(x: np.ndarray) -> float:
    n = len(x)
    if n < 4:
        return np.nan
    h = n // 2
    a, b = x[:h], x[h:]
    if np.all(np.isnan(a)) or np.all(np.isnan(b)):
        return np.nan
    return np.nanmean(b) - np.nanmean(a)


def build_windows(seg: pd.DataFrame, window_sec: int, step_sec: int) -> pd.DataFrame:
    """Aggregate the 1 Hz enriched stream into fixed windows, strided by step_sec."""
    seg = seg.reset_index(drop=True)
    n = len(seg)
    starts = np.arange(0, n - window_sec + 1, step_sec)

    counts = seg["counts"].values
    bg_median = seg["bg_median"].values
    bg_mad = seg["bg_mad_std"].values
    excess = seg["excess"].values
    sigma_excess = seg["sigma_excess"].values
    ratio_bg = seg["ratio_to_bg"].values
    ratio_long_bg = seg["ratio_to_long_bg"].values
    is_flare = seg["is_flare_second"].values
    flare_class = seg["flare_class_second"].values
    dt_arr = seg["datetime"].values

    rows = []
    for s in starts:
        e = s + window_sec
        c = counts[s:e]
        if np.all(np.isnan(c)):
            continue  # entirely missing window, skip

        row = {
            "window_start_time": dt_arr[s],
            "window_end_time": dt_arr[e - 1],
            "segment_id": seg["segment_id"].iloc[s],
            "source_file": seg["source_file"].iloc[e - 1],  # causal: label with last file touched
            "n_valid": np.sum(~np.isnan(c)),
            "frac_valid": np.mean(~np.isnan(c)),
            "mean": np.nanmean(c),
            "median": np.nanmedian(c),
            "std": np.nanstd(c),
            "min": np.nanmin(c),
            "max": np.nanmax(c),
            "ptp": np.nanmax(c) - np.nanmin(c),
            "p25": np.nanpercentile(c, 25),
            "p75": np.nanpercentile(c, 75),
            "p90": np.nanpercentile(c, 90),
            "skew": pd.Series(c).skew(),
            "kurtosis": pd.Series(c).kurt(),
            "slope": _slope(c),
            "rise_delta": _first_second_half_delta(c),
            # background-relative features (causal, same at inference time)
            "bg_median_at_end": bg_median[e - 1],
            "bg_mad_std_at_end": bg_mad[e - 1],
            "mean_excess": np.nanmean(excess[s:e]),
            "max_sigma_excess": np.nanmax(sigma_excess[s:e]) if not np.all(np.isnan(sigma_excess[s:e])) else np.nan,
            "mean_ratio_to_bg": np.nanmean(ratio_bg[s:e]),
            "max_ratio_to_bg": np.nanmax(ratio_bg[s:e]) if not np.all(np.isnan(ratio_bg[s:e])) else np.nan,
            "frac_above_3sigma": np.mean(np.nan_to_num(sigma_excess[s:e], nan=-99) > 3),
            "frac_above_5sigma": np.mean(np.nan_to_num(sigma_excess[s:e], nan=-99) > 5),
            "ratio_local_to_longterm_bg": ratio_long_bg[e - 1],  # solar-cycle-scale context
            # time-of-day cyclic features (help distinguish scheduled/instrumental artifacts if any)
            "hour_sin": np.sin(2 * np.pi * pd.Timestamp(dt_arr[e - 1]).hour / 24),
            "hour_cos": np.cos(2 * np.pi * pd.Timestamp(dt_arr[e - 1]).hour / 24),
        }

        # ---- labels (HEK-derived; label-only, never fed back as a feature) ----
        window_flare_frac = np.mean(is_flare[s:e])
        row["label_binary"] = int(window_flare_frac > DEFAULT_OVERLAP_THRESHOLD)
        if row["label_binary"]:
            classes, counts_c = np.unique(flare_class[s:e][is_flare[s:e]], return_counts=True)
            row["label_class"] = classes[np.argmax(counts_c)]
        else:
            row["label_class"] = "NONE"
        row["flare_frac_in_window"] = window_flare_frac

        rows.append(row)

    return pd.DataFrame(rows)


# ============================== main ==============================

@dataclass
class Config:
    lc_dir: str
    hek_csv: str
    out_dir: str
    window_sec: int
    step_sec: int
    bg_window_sec: int
    long_bg_window_sec: int


def run(cfg: Config):
    os.makedirs(cfg.out_dir, exist_ok=True)

    hek = load_hek_flares(cfg.hek_csv)
    file_index = discover_lc_files(cfg.lc_dir)
    segments = build_segments(file_index)

    all_windows = []
    qc_lines = []
    covered_start, covered_end = None, None

    for seg in segments:
        seg = clean_gaps(seg)
        seg = add_causal_background_features(seg, cfg.bg_window_sec, cfg.long_bg_window_sec)
        seg = add_flare_labels(seg, hek)
        w = build_windows(seg, cfg.window_sec, cfg.step_sec)
        all_windows.append(w)

        s0, s1 = seg["datetime"].iloc[0], seg["datetime"].iloc[-1]
        covered_start = s0 if covered_start is None else min(covered_start, s0)
        covered_end = s1 if covered_end is None else max(covered_end, s1)
        qc_lines.append(f"segment {seg['segment_id'].iloc[0]}: {s0} -> {s1}, "
                         f"{w['label_binary'].sum()} positive windows / {len(w)} total")

    windows_df = pd.concat(all_windows, ignore_index=True)

    # HEK flares that fall entirely outside covered data -> can't be used, report them
    missing = hek[(hek["event_endtime"] < covered_start) | (hek["event_starttime"] > covered_end)]
    matched = hek[~hek.index.isin(missing.index)]

    out_path = os.path.join(cfg.out_dir, "windows_features.parquet")
    windows_df.to_parquet(out_path, index=False)

    qc_path = os.path.join(cfg.out_dir, "qc_report.txt")
    with open(qc_path, "w") as f:
        f.write("=== Coverage ===\n")
        f.write(f"LC data coverage: {covered_start} -> {covered_end}\n")
        f.write("\n".join(qc_lines) + "\n\n")
        f.write("=== HEK matching ===\n")
        f.write(f"Total HEK flares: {len(hek)}\n")
        f.write(f"Matched within LC coverage: {len(matched)}\n")
        f.write(f"Outside LC coverage (unused): {len(missing)}\n")
        f.write(f"lc_date rollover cases (auto-resolved via timestamp labeling): "
                f"{hek['lc_date_rollover_flag'].sum()}\n\n")
        f.write("=== Window dataset ===\n")
        f.write(f"Total windows: {len(windows_df)}\n")
        f.write(f"Positive (flare) windows: {windows_df['label_binary'].sum()} "
                f"({windows_df['label_binary'].mean():.3%})\n")
        f.write("Class balance (label_class):\n")
        f.write(windows_df["label_class"].value_counts().to_string() + "\n")

    log.info("Wrote %d windows to %s", len(windows_df), out_path)
    log.info("QC report: %s", qc_path)
    log.info("Positive window rate: %.3f%%", 100 * windows_df["label_binary"].mean())

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lc-dir", default=r"data\lcfiles", help="Directory containing AL1_SOLEXS_*_SDD2_L1.lc files")
    p.add_argument("--hek-csv", default=r"data\hek_flares.csv", help="Path to hek_flares.csv")
    p.add_argument("--out-dir", default=r"data\out_builddata", help="Directory to write outputs into")
    p.add_argument("--window-sec", type=int, default=DEFAULT_WINDOW_SEC)
    p.add_argument("--step-sec", type=int, default=DEFAULT_STEP_SEC)
    p.add_argument("--bg-window-sec", type=int, default=DEFAULT_BG_WINDOW_SEC)
    p.add_argument("--long-bg-window-sec", type=int, default=DEFAULT_LONG_BG_WINDOW_SEC)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(Config(
        lc_dir=args.lc_dir,
        hek_csv=args.hek_csv,
        out_dir=args.out_dir,
        window_sec=args.window_sec,
        step_sec=args.step_sec,
        bg_window_sec=args.bg_window_sec,
        long_bg_window_sec=args.long_bg_window_sec,
    ))
