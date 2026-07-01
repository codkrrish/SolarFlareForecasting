"""
SOLEXS Solar Flare Detection — Threshold Analysis
===================================================
Objective:
  1. Derive per-class COUNTS thresholds (B/C/M/X + flare/non-flare) from the
     training set using HEK ground-truth labels.
  2. Repeat the same analysis on the first-difference gradient of COUNTS.
  3. Add a configurable soft-boundary (error margin) around every threshold.
  4. Report findings to console and write a JSON summary + CSV of per-day stats.

Global configuration — edit the three paths below before running.
"""

# ──────────────────────────────────────────────────────────────────────────────
# GLOBAL CONFIGURATION  ← edit these three variables
# ──────────────────────────────────────────────────────────────────────────────
DATA_DIR   = "data/lcfiles"          # directory containing *.lc files
HEK_CSV    = "data/hek_flares.csv"    # ground-truth CSV
OUTPUT_DIR = "data/analysis"            # where results are written

# Soft-boundary margin (fraction of the threshold value).
# 0.10 = ±10 %.  The script reports both the hard threshold and the
# [threshold*(1-MARGIN), threshold*(1+MARGIN)] soft boundary.
SOFT_MARGIN = 0.10
# ──────────────────────────────────────────────────────────────────────────────

import os
import glob
import json
import warnings
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy import stats
import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def read_lc(path: str) -> pd.DataFrame:
    """Read a SOLEXS .lc FITS file and return a tidy DataFrame."""
    with fits.open(path) as hdul:
        data  = hdul["RATE"].data
        times = data["TIME"].astype(float)    # Unix seconds (UTC)
        cnts  = data["COUNTS"].astype(float)  # integrated counts
    df = pd.DataFrame({"unix_time": times, "counts": cnts})
    df["datetime"] = pd.to_datetime(df["unix_time"], unit="s", utc=True)
    return df


def lc_date_from_path(path: str) -> int:
    """Extract integer date like 20240202 from filename."""
    basename = os.path.basename(path)              # AL1_SOLEXS_20240202_SDD2_L1.lc
    parts    = basename.split("_")
    return int(parts[2])                           # index 2 = YYYYMMDD


def load_all_lc(data_dir: str):
    """
    Return sorted list of (lc_date, DataFrame) for every .lc file found.
    Files with two consecutive dates are NOT pre-merged here — merging is
    done on-the-fly in the window builder so we honour the continuity rule.
    """
    pattern = os.path.join(data_dir, "AL1_SOLEXS_*_SDD2_L1.lc")
    paths   = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No .lc files found in {data_dir!r}")
    print(f"[INFO] Found {len(paths)} .lc files.")
    records = []
    for p in paths:
        date = lc_date_from_path(p)
        df   = read_lc(p)
        records.append((date, df))
    return records


def load_hek(csv_path: str) -> pd.DataFrame:
    """Load and clean HEK ground-truth CSV."""
    df = pd.read_csv(csv_path, usecols=[
        "lc_date", "event_starttime", "event_peaktime",
        "event_endtime", "fl_goescls"
    ])
    # Parse timestamps — strip tz suffix if present, treat as naive UTC
    for col in ["event_starttime", "event_peaktime", "event_endtime"]:
        df[col] = (pd.to_datetime(df[col], utc=False)
                     .dt.tz_localize(None))
    df["flare_class"]     = df["fl_goescls"].str[0].str.upper()
    df["flare_subclass"]  = pd.to_numeric(
        df["fl_goescls"].str[1:], errors="coerce"
    )
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  BUILD LABELLED WINDOWS
# ═══════════════════════════════════════════════════════════════════════════════

CLASS_ORDER = ["B", "C", "M", "X"]


def build_windows(lc_records, hek_df, date_set):
    """
    For every HEK flare whose lc_date is in date_set:
      - Pull the matching LC day (and optionally the next day if the event
        spans midnight, i.e. the next date is also in lc_records).
      - Slice out the [start, end] time window.
      - Collect those COUNTS samples and label them with the flare class.

    Also collect a random 10-minute quiet window from the same day if no
    flare overlaps, labelled "quiet" (non-flare background).

    Returns two DataFrames:
      flare_samples : columns [counts, flare_class, lc_date]
      quiet_samples : columns [counts, lc_date]
    """
    # index by date for O(1) lookup
    lc_by_date = {date: df for date, df in lc_records if date in date_set}

    flare_rows = []
    quiet_rows = []

    dates_in_set = sorted(lc_by_date.keys())
    date_set_lookup = set(dates_in_set)

    for date in dates_in_set:
        lc_today = lc_by_date[date].copy()

        # Check if next calendar date is also available (for continuity concat)
        next_date_int = _next_date_int(date)
        if next_date_int in lc_by_date:
            lc_combined = pd.concat(
                [lc_today, lc_by_date[next_date_int]], ignore_index=True
            )
        else:
            lc_combined = lc_today

        # Flare windows for this lc_date
        day_flares = hek_df[hek_df["lc_date"] == date]
        flare_intervals = []

        for _, row in day_flares.iterrows():
            t_start = row["event_starttime"]
            t_end   = row["event_endtime"]
            mask    = (
                (lc_combined["datetime"].dt.tz_localize(None) >= t_start) &
                (lc_combined["datetime"].dt.tz_localize(None) <= t_end)
            )
            window = lc_combined.loc[mask, "counts"].dropna()
            if len(window) == 0:
                continue
            for v in window.values:
                flare_rows.append({
                    "counts":      v,
                    "flare_class": row["flare_class"],
                    "lc_date":     date,
                })
            flare_intervals.append((t_start, t_end))

        # Quiet window: pick a 600-s stretch that doesn't overlap any flare
        lc_day_only  = lc_today.copy()
        lc_day_only["dt_naive"] = lc_day_only["datetime"].dt.tz_localize(None)
        lc_day_only  = lc_day_only.dropna(subset=["counts"])

        quiet_mask = np.ones(len(lc_day_only), dtype=bool)
        for t_start, t_end in flare_intervals:
            quiet_mask &= ~(
                (lc_day_only["dt_naive"] >= t_start) &
                (lc_day_only["dt_naive"] <= t_end)
            )

        quiet_pool = lc_day_only.loc[quiet_mask, "counts"]
        if len(quiet_pool) >= 600:
            sampled = quiet_pool.sample(n=600, random_state=42)
            for v in sampled.values:
                quiet_rows.append({"counts": v, "lc_date": date})

    flare_df = pd.DataFrame(flare_rows)
    quiet_df = pd.DataFrame(quiet_rows)
    return flare_df, quiet_df


def _next_date_int(date_int: int) -> int:
    """Return the next calendar day as YYYYMMDD integer."""
    d = pd.Timestamp(str(date_int))
    return int((d + pd.Timedelta(days=1)).strftime("%Y%m%d"))


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  GRADIENT FEATURE
# ═══════════════════════════════════════════════════════════════════════════════

def add_gradient_column(df_lc: pd.DataFrame) -> pd.DataFrame:
    """Add a first-difference gradient column (counts/s per second)."""
    df = df_lc.copy().sort_values("unix_time")
    df["gradient"] = df["counts"].diff()   # Δcounts between consecutive seconds
    return df


def build_gradient_windows(lc_records, hek_df, date_set):
    """Same as build_windows but returns gradient values instead of counts."""
    lc_by_date = {}
    for date, df in lc_records:
        if date in date_set:
            lc_by_date[date] = add_gradient_column(df)

    flare_rows = []
    quiet_rows = []
    dates_in_set = sorted(lc_by_date.keys())

    for date in dates_in_set:
        lc_today = lc_by_date[date].copy()
        next_date_int = _next_date_int(date)
        if next_date_int in lc_by_date:
            lc_combined = pd.concat(
                [lc_today, lc_by_date[next_date_int]], ignore_index=True
            )
        else:
            lc_combined = lc_today

        day_flares = hek_df[hek_df["lc_date"] == date]
        flare_intervals = []

        for _, row in day_flares.iterrows():
            t_start = row["event_starttime"]
            t_end   = row["event_endtime"]
            mask = (
                (lc_combined["datetime"].dt.tz_localize(None) >= t_start) &
                (lc_combined["datetime"].dt.tz_localize(None) <= t_end)
            )
            window = lc_combined.loc[mask, "gradient"].dropna()
            if len(window) == 0:
                continue
            for v in window.values:
                flare_rows.append({
                    "gradient":    v,
                    "flare_class": row["flare_class"],
                    "lc_date":     date,
                })
            flare_intervals.append((t_start, t_end))

        # Quiet gradient
        lc_day_only = lc_today.copy()
        lc_day_only["dt_naive"] = lc_day_only["datetime"].dt.tz_localize(None)
        lc_day_only = lc_day_only.dropna(subset=["gradient"])
        quiet_mask = np.ones(len(lc_day_only), dtype=bool)
        for t_start, t_end in flare_intervals:
            quiet_mask &= ~(
                (lc_day_only["dt_naive"] >= t_start) &
                (lc_day_only["dt_naive"] <= t_end)
            )
        quiet_pool = lc_day_only.loc[quiet_mask, "gradient"]
        if len(quiet_pool) >= 600:
            sampled = quiet_pool.sample(n=600, random_state=42)
            for v in sampled.values:
                quiet_rows.append({"gradient": v, "lc_date": date})

    return pd.DataFrame(flare_rows), pd.DataFrame(quiet_rows)


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  THRESHOLD DERIVATION
# ═══════════════════════════════════════════════════════════════════════════════

def derive_thresholds(flare_df: pd.DataFrame,
                      quiet_df:  pd.DataFrame,
                      feature:   str = "counts") -> dict:
    """
    Derive thresholds for:
      A. Flare / non-flare binary classification
      B. Per-class (B, C, M, X) classification

    Strategy:
      - Flare/non-flare : find the value that best separates the quiet
        distribution (mean + k*std) from all flare samples.  We use the
        point midway between (quiet 99th pct) and (all-flare 1st pct) if
        the two distributions don't overlap much; otherwise we optimise a
        simple ROC to maximise Youden's J statistic.
      - Per-class : for each ordered class pair (B↔C, C↔M, M↔X) compute
        the midpoint between the lower class's 95th percentile and the
        upper class's 5th percentile.  If they overlap we fall back to the
        mean of the two medians.

    Returns a dict with keys:
      "flare_nonflare_threshold"    : float
      "class_thresholds"            : {pair_label: float}
      "class_stats"                 : {class: {mean, median, p05, p95, n}}
      "quiet_stats"                 : {mean, std, p99, n}
    """
    result = {}

    # ── Quiet stats ──────────────────────────────────────────────────────────
    q_vals = quiet_df[feature].dropna().values
    result["quiet_stats"] = {
        "mean":   float(np.mean(q_vals)),
        "std":    float(np.std(q_vals)),
        "p99":    float(np.percentile(q_vals, 99)),
        "n":      int(len(q_vals)),
    }

    # ── Per-class stats ───────────────────────────────────────────────────────
    class_stats = {}
    for cls in CLASS_ORDER:
        vals = flare_df.loc[flare_df["flare_class"] == cls, feature].dropna().values
        if len(vals) == 0:
            class_stats[cls] = None
            continue
        class_stats[cls] = {
            "mean":   float(np.mean(vals)),
            "median": float(np.median(vals)),
            "std":    float(np.std(vals)),
            "p05":    float(np.percentile(vals,  5)),
            "p25":    float(np.percentile(vals, 25)),
            "p75":    float(np.percentile(vals, 75)),
            "p95":    float(np.percentile(vals, 95)),
            "min":    float(np.min(vals)),
            "max":    float(np.max(vals)),
            "n":      int(len(vals)),
        }
    result["class_stats"] = class_stats

    # ── Flare / non-flare threshold ───────────────────────────────────────────
    all_flare_vals = flare_df[feature].dropna().values
    quiet_p99      = result["quiet_stats"]["p99"]
    flare_p01      = float(np.percentile(all_flare_vals, 1)) if len(all_flare_vals) else quiet_p99

    if flare_p01 > quiet_p99:
        # Clean separation → midpoint
        fn_thresh = (quiet_p99 + flare_p01) / 2.0
        fn_method = "midpoint"
    else:
        # Overlap → maximise Youden J on a grid
        grid  = np.linspace(
            min(q_vals.min(), all_flare_vals.min()),
            max(q_vals.max(), all_flare_vals.max()),
            500,
        )
        best_j, best_t = -1, grid[0]
        for t in grid:
            tpr = np.mean(all_flare_vals >= t)   # sensitivity
            tnr = np.mean(q_vals         <  t)   # specificity
            j   = tpr + tnr - 1
            if j > best_j:
                best_j, best_t = j, t
        fn_thresh = float(best_t)
        fn_method = f"Youden-J (J={best_j:.3f})"

    result["flare_nonflare_threshold"] = fn_thresh
    result["flare_nonflare_method"]    = fn_method

    # ── Per-class boundary thresholds ─────────────────────────────────────────
    class_boundaries = {}
    for i in range(len(CLASS_ORDER) - 1):
        lo_cls = CLASS_ORDER[i]
        hi_cls = CLASS_ORDER[i + 1]
        label  = f"{lo_cls}_vs_{hi_cls}"

        lo = class_stats.get(lo_cls)
        hi = class_stats.get(hi_cls)
        if lo is None or hi is None:
            class_boundaries[label] = None
            continue

        if hi["p05"] > lo["p95"]:
            # No overlap → midpoint between p95(low) and p05(high)
            thresh  = (lo["p95"] + hi["p05"]) / 2.0
            method  = "midpoint p95/p05"
        else:
            # Overlap → mean of the two medians
            thresh  = (lo["median"] + hi["median"]) / 2.0
            method  = "mean of medians (overlapping distributions)"

        class_boundaries[label] = {
            "threshold": float(thresh),
            "method":    method,
        }
    result["class_thresholds"] = class_boundaries
    return result


def apply_soft_boundary(thresholds: dict, margin: float) -> dict:
    """
    Augment the threshold dict with soft-boundary (low, high) tuples.
    margin = 0.10 means ±10 % around the hard threshold.
    """
    soft = {}

    fn_t = thresholds["flare_nonflare_threshold"]
    soft["flare_nonflare"] = {
        "hard":  fn_t,
        "low":   fn_t * (1 - margin),
        "high":  fn_t * (1 + margin),
    }

    soft["class_thresholds"] = {}
    for label, info in thresholds["class_thresholds"].items():
        if info is None:
            soft["class_thresholds"][label] = None
            continue
        t = info["threshold"]
        soft["class_thresholds"][label] = {
            "hard":   t,
            "low":    t * (1 - margin),
            "high":   t * (1 + margin),
            "method": info["method"],
        }
    return soft


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  EVALUATION ON TEST SET
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_thresholds(lc_records, hek_df, date_set,
                        thresholds_counts: dict, thresholds_grad: dict,
                        feature_pair=("counts", "gradient")):
    """
    For each day in date_set:
      - Classify every second using the COUNTS threshold → predicted label.
      - Classify every second using the GRADIENT threshold → predicted label.
      - Compare against HEK ground truth.
    Returns a per-day summary DataFrame.
    """
    lc_by_date = {date: df for date, df in lc_records if date in date_set}
    rows = []

    fn_thresh_c = thresholds_counts["flare_nonflare_threshold"]
    fn_thresh_g = thresholds_grad["flare_nonflare_threshold"]

    for date in sorted(lc_by_date.keys()):
        lc   = lc_by_date[date].copy()
        lcg  = add_gradient_column(lc)

        # Ground truth flare seconds for this date
        day_flares  = hek_df[hek_df["lc_date"] == date]
        flare_secs  = set()
        for _, row in day_flares.iterrows():
            mask = (
                (lc["datetime"].dt.tz_localize(None) >= row["event_starttime"]) &
                (lc["datetime"].dt.tz_localize(None) <= row["event_endtime"])
            )
            flare_secs.update(lc.index[mask].tolist())

        total_secs  = len(lc)
        valid_mask  = lc["counts"].notna()
        valid_idx   = lc.index[valid_mask]

        true_labels = np.array([1 if i in flare_secs else 0 for i in valid_idx])

        # Counts predictions
        pred_counts = (lc.loc[valid_idx, "counts"].values >= fn_thresh_c).astype(int)
        tp_c = int(np.sum((true_labels == 1) & (pred_counts == 1)))
        fp_c = int(np.sum((true_labels == 0) & (pred_counts == 1)))
        tn_c = int(np.sum((true_labels == 0) & (pred_counts == 0)))
        fn_c = int(np.sum((true_labels == 1) & (pred_counts == 0)))

        # Gradient predictions
        grad_valid_mask = lcg["gradient"].notna() & valid_mask
        grad_valid_idx  = lcg.index[grad_valid_mask]
        if len(grad_valid_idx) > 0:
            true_g = np.array([1 if i in flare_secs else 0 for i in grad_valid_idx])
            pred_g = (lcg.loc[grad_valid_idx, "gradient"].values >= fn_thresh_g).astype(int)
            tp_g = int(np.sum((true_g == 1) & (pred_g == 1)))
            fp_g = int(np.sum((true_g == 0) & (pred_g == 1)))
            tn_g = int(np.sum((true_g == 0) & (pred_g == 0)))
            fn_g = int(np.sum((true_g == 1) & (pred_g == 0)))
        else:
            tp_g = fp_g = tn_g = fn_g = 0

        rows.append({
            "lc_date":      date,
            "n_flare_events": len(day_flares),
            "n_flare_secs": len(flare_secs),
            # Counts
            "c_tp": tp_c, "c_fp": fp_c, "c_tn": tn_c, "c_fn": fn_c,
            "c_precision": _safe_div(tp_c, tp_c + fp_c),
            "c_recall":    _safe_div(tp_c, tp_c + fn_c),
            "c_f1":        _safe_f1(tp_c, fp_c, fn_c),
            # Gradient
            "g_tp": tp_g, "g_fp": fp_g, "g_tn": tn_g, "g_fn": fn_g,
            "g_precision": _safe_div(tp_g, tp_g + fp_g),
            "g_recall":    _safe_div(tp_g, tp_g + + fn_g),
            "g_f1":        _safe_f1(tp_g, fp_g, fn_g),
        })

    return pd.DataFrame(rows)


def _safe_div(n, d):
    return float(n / d) if d > 0 else float("nan")

def _safe_f1(tp, fp, fn):
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom > 0 else float("nan")


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  VISUALISATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _class_color(cls):
    return {"B": "#4daf4a", "C": "#377eb8", "M": "#ff7f00", "X": "#e41a1c"}.get(cls, "grey")


def plot_distributions(flare_df, quiet_df, thresholds, soft, feature,
                       out_dir, tag):
    """Violin + box plot of feature distributions by class."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f"Distribution of {feature} by flare class ({tag})", fontsize=14)

    ax = axes[0]
    data_groups = [quiet_df[feature].dropna().values]
    labels      = ["quiet"]
    colors      = ["#999999"]
    for cls in CLASS_ORDER:
        vals = flare_df.loc[flare_df["flare_class"] == cls, feature].dropna().values
        if len(vals):
            data_groups.append(vals)
            labels.append(cls)
            colors.append(_class_color(cls))

    parts = ax.violinplot(data_groups, showmedians=True)
    for i, (patch, col) in enumerate(zip(parts["bodies"], colors)):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel(feature)
    ax.set_title("Violin plot")
    ax.set_yscale("symlog" if feature == "gradient" else "log")

    # Mark thresholds
    fn_hard = soft["flare_nonflare"]["hard"]
    ax.axhline(fn_hard, color="black", ls="--", lw=1.5, label="F/NF threshold")
    ax.fill_between(
        [0.5, len(labels) + 0.5],
        soft["flare_nonflare"]["low"],
        soft["flare_nonflare"]["high"],
        color="black", alpha=0.1
    )
    ax.legend(fontsize=8)

    ax2 = axes[1]
    for cls, col in zip(CLASS_ORDER, [_class_color(c) for c in CLASS_ORDER]):
        vals = flare_df.loc[flare_df["flare_class"] == cls, feature].dropna().values
        if len(vals) == 0:
            continue
        bins = np.linspace(np.percentile(vals, 1), np.percentile(vals, 99), 60)
        ax2.hist(vals, bins=bins, alpha=0.5, color=col, label=cls, density=True)

    for label, info in soft["class_thresholds"].items():
        if info is None:
            continue
        ax2.axvline(info["hard"], ls="-", lw=1.5, label=f'{label} boundary')
        ax2.axvspan(info["low"], info["high"], alpha=0.1)

    ax2.set_xlabel(feature)
    ax2.set_ylabel("Density")
    ax2.set_title("Histogram by class + class boundaries")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"dist_{tag}.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [plot] saved → {out_path}")


def plot_test_metrics(eval_df, out_dir):
    """Bar chart of per-day F1 for counts vs gradient on test set."""
    fig, ax = plt.subplots(figsize=(14, 5))
    x  = np.arange(len(eval_df))
    w  = 0.35
    ax.bar(x - w/2, eval_df["c_f1"].fillna(0), w, label="Counts F1",   color="#377eb8", alpha=0.8)
    ax.bar(x + w/2, eval_df["g_f1"].fillna(0), w, label="Gradient F1", color="#e41a1c", alpha=0.8)
    ax.set_xticks(x[::max(1, len(x)//20)])
    ax.set_xticklabels(
        [str(eval_df["lc_date"].iloc[i]) for i in range(0, len(eval_df), max(1, len(eval_df)//20))],
        rotation=45, ha="right", fontsize=7
    )
    ax.set_ylabel("F1 score")
    ax.set_title("Per-day binary flare F1 — Counts vs Gradient (test set)")
    ax.legend()
    plt.tight_layout()
    out_path = os.path.join(out_dir, "test_f1_comparison.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [plot] saved → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def print_report(thresh_c, soft_c, thresh_g, soft_g, eval_df, margin):
    sep = "═" * 70

    print(f"\n{sep}")
    print("  SOLEXS FLARE DETECTION — THRESHOLD ANALYSIS REPORT")
    print(sep)
    print(f"  Soft-boundary margin : ±{margin*100:.0f}%\n")

    for tag, thresh, soft in [("COUNTS", thresh_c, soft_c),
                               ("GRADIENT (Δcounts/s)", thresh_g, soft_g)]:
        print(f"\n{'─'*70}")
        print(f"  FEATURE: {tag}")
        print(f"{'─'*70}")

        qs = thresh["quiet_stats"]
        print(f"  Quiet (non-flare) background")
        print(f"    N={qs['n']:,}  mean={qs['mean']:.2f}  std={qs['std']:.2f}  p99={qs['p99']:.2f}")

        print(f"\n  Per-class statistics")
        print(f"  {'Class':>5}  {'N':>7}  {'mean':>10}  {'median':>10}  {'p05':>10}  {'p95':>10}")
        for cls in CLASS_ORDER:
            cs = thresh["class_stats"].get(cls)
            if cs is None:
                print(f"  {cls:>5}  {'—':>7}")
                continue
            print(f"  {cls:>5}  {cs['n']:>7,}  {cs['mean']:>10.2f}  {cs['median']:>10.2f}"
                  f"  {cs['p05']:>10.2f}  {cs['p95']:>10.2f}")

        fn = soft["flare_nonflare"]
        print(f"\n  ► Binary flare/non-flare threshold")
        print(f"    Method  : {thresh['flare_nonflare_method']}")
        print(f"    Hard    : {fn['hard']:.4f}")
        print(f"    Soft ±  : [{fn['low']:.4f}, {fn['high']:.4f}]")

        print(f"\n  ► Class boundary thresholds")
        for label, info in soft["class_thresholds"].items():
            if info is None:
                print(f"    {label:<10}  insufficient data")
                continue
            print(f"    {label:<10}  hard={info['hard']:.4f}  "
                  f"soft=[{info['low']:.4f}, {info['high']:.4f}]  ({info['method']})")

    if eval_df is not None and len(eval_df):
        print(f"\n{'─'*70}")
        print(f"  TEST-SET EVALUATION (binary flare/non-flare, per-second)")
        print(f"{'─'*70}")
        for feat, tp_col, prec_col, rec_col, f1_col in [
            ("Counts",   "c_tp", "c_precision", "c_recall", "c_f1"),
            ("Gradient", "g_tp", "g_precision", "g_recall", "g_f1"),
        ]:
            sub = eval_df.dropna(subset=[f1_col])
            agg_tp = eval_df["c_tp" if feat == "Counts" else "g_tp"].sum()
            agg_fp = eval_df["c_fp" if feat == "Counts" else "g_fp"].sum()
            agg_tn = eval_df["c_tn" if feat == "Counts" else "g_tn"].sum()
            agg_fn = eval_df["c_fn" if feat == "Counts" else "g_fn"].sum()
            agg_prec = _safe_div(agg_tp, agg_tp + agg_fp)
            agg_rec  = _safe_div(agg_tp, agg_tp + agg_fn)
            agg_f1   = _safe_f1(agg_tp, agg_fp, agg_fn)
            print(f"\n  [{feat}]")
            print(f"    Aggregate  precision={agg_prec:.3f}  recall={agg_rec:.3f}  F1={agg_f1:.3f}")
            print(f"    Confusion  TP={agg_tp:,}  FP={agg_fp:,}  TN={agg_tn:,}  FN={agg_fn:,}")
            print(f"    Median daily F1 : {sub[f1_col].median():.3f}")
            print(f"    Mean daily F1   : {sub[f1_col].mean():.3f}")

        better = (
            "GRADIENT" if
            eval_df["g_f1"].mean() > eval_df["c_f1"].mean()
            else "COUNTS"
        )
        print(f"\n  ► Gradient vs Counts: '{better}' yields higher mean daily F1 "
              f"on the test set.")

    print(f"\n{sep}\n")



# Insert this function right above your main() execution block

def test_collective_performance(lc_records, hek_df, train_dates, test_dates, thresholds_counts: dict, thresholds_grad: dict):
    """
    Evaluates binary flare detection performance collectively over entire datasets
    (Train and Test) treated as a single continuous observation stream.
    
    Computes global Accuracy, Precision, Recall, and F1-score.
    """
    fn_thresh_c = thresholds_counts["flare_nonflare_threshold"]
    fn_thresh_g = thresholds_grad["flare_nonflare_threshold"]
    
    datasets = {
        "TRAINING SET": train_dates,
        "TEST SET": test_dates
    }
    
    lc_by_date = {date: df for date, df in lc_records}
    
    print("\n" + "═" * 70)
    print("  COLLECTIVE PERFORMANCE METRICS (GLOBAL STREAM EVALUATION)")
    print("═" * 70)
    
    for name, date_set in datasets.items():
        if not date_set:
            print(f"\n  [ {name} ] - No dates available to evaluate.")
            continue
            
        # Global Counters for Counts feature
        c_tp, c_fp, c_tn, c_fn = 0, 0, 0, 0
        # Global Counters for Gradient feature
        g_tp, g_fp, g_tn, g_fn = 0, 0, 0, 0
        
        for date in sorted(date_set):
            if date not in lc_by_date:
                continue
                
            lc = lc_by_date[date].copy()
            lcg = add_gradient_column(lc)
            
            # Identify ground truth flare timestamps for this date
            day_flares = hek_df[hek_df["lc_date"] == date]
            flare_secs = set()
            for _, row in day_flares.iterrows():
                mask = (
                    (lc["datetime"].dt.tz_localize(None) >= row["event_starttime"]) &
                    (lc["datetime"].dt.tz_localize(None) <= row["event_endtime"])
                )
                flare_secs.update(lc.index[mask].tolist())
                
            valid_mask = lc["counts"].notna()
            valid_idx = lc.index[valid_mask]
            true_labels = np.array([1 if i in flare_secs else 0 for i in valid_idx])
            
            # ── Counts Stream Evaluation ──────────────────────────────────────
            pred_counts = (lc.loc[valid_idx, "counts"].values >= fn_thresh_c).astype(int)
            c_tp += int(np.sum((true_labels == 1) & (pred_counts == 1)))
            c_fp += int(np.sum((true_labels == 0) & (pred_counts == 1)))
            c_tn += int(np.sum((true_labels == 0) & (pred_counts == 0)))
            c_fn += int(np.sum((true_labels == 1) & (pred_counts == 0)))
            
            # ── Gradient Stream Evaluation ────────────────────────────────────
            grad_valid_mask = lcg["gradient"].notna() & valid_mask
            grad_valid_idx = lcg.index[grad_valid_mask]
            if len(grad_valid_idx) > 0:
                true_g = np.array([1 if i in flare_secs else 0 for i in grad_valid_idx])
                pred_g = (lcg.loc[grad_valid_idx, "gradient"].values >= fn_thresh_g).astype(int)
                g_tp += int(np.sum((true_g == 1) & (pred_g == 1)))
                g_fp += int(np.sum((true_g == 0) & (pred_g == 1)))
                g_tn += int(np.sum((true_g == 0) & (pred_g == 0)))
                g_fn += int(np.sum((true_g == 1) & (pred_g == 0)))

        # ── Calculate Global Metrics ──────────────────────────────────────────
        print(f"\n  ┌──────────────────────────────────────────────────────────────┐")
        print(f"  │  DATASET: {name:<50} │")
        print(f"  └──────────────────────────────────────────────────────────────┘")
        
        for feat_name, tp, fp, tn, fn in [("Counts Thresholding", c_tp, c_fp, c_tn, c_fn),
                                           ("Gradient Thresholding", g_tp, g_fp, g_tn, g_fn)]:
            total = tp + fp + tn + fn
            if total == 0:
                print(f"    [{feat_name}] No valid sequences processed.")
                continue
                
            accuracy  = _safe_div((tp + tn), total)
            precision = _safe_div(tp, (tp + fp))
            recall    = _safe_div(tp, (tp + fn))
            f1_score  = _safe_f1(tp, fp, fn)
            
            print(f"    ► Feature Engine: {feat_name}")
            print(f"      Confusion Matrix : TP={tp:,} | FP={fp:,} | TN={tn:,} | FN={fn:,}")
            print(f"      Accuracy         : {accuracy:.4%}")
            print(f"      Precision        : {precision:.4f}")
            print(f"      Recall           : {recall:.4f}")
            print(f"      F1-Score         : {f1_score:.4f}")
            print(f"      ──────────────────────────────────────────────────────────")
            
    print("═" * 70 + "\n")

# ═══════════════════════════════════════════════════════════════════════════════
# 8.  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("[1/7] Loading LC files …")
    lc_records = load_all_lc(DATA_DIR)
    all_dates  = sorted(d for d, _ in lc_records)
    print(f"      Date range: {all_dates[0]} – {all_dates[-1]}")

    print("[2/7] Loading HEK ground truth …")
    hek_df = load_hek(HEK_CSV)
    print(f"      {len(hek_df)} flare events; classes: "
          f"{sorted(hek_df['flare_class'].unique())}")

    # Train/test split (first 270 days = train)
    N_TRAIN    = 270
    train_dates = set(all_dates[:N_TRAIN])
    test_dates  = set(all_dates[N_TRAIN:])
    print(f"      Train: {len(train_dates)} days  |  Test: {len(test_dates)} days")

    # ── Build labelled windows — COUNTS ───────────────────────────────────────
    print("[3/7] Building labelled windows (COUNTS) …")
    flare_c_train, quiet_c_train = build_windows(lc_records, hek_df, train_dates)
    print(f"      Flare seconds: {len(flare_c_train):,}  "
          f"Quiet seconds sampled: {len(quiet_c_train):,}")
    if len(flare_c_train) == 0:
        raise RuntimeError(
            "No flare windows matched in the training set. "
            "Check that DATA_DIR contains files whose dates appear in HEK_CSV."
        )

    # ── Build labelled windows — GRADIENT ─────────────────────────────────────
    print("[4/7] Building labelled windows (GRADIENT) …")
    flare_g_train, quiet_g_train = build_gradient_windows(
        lc_records, hek_df, train_dates
    )
    print(f"      Flare gradient samples: {len(flare_g_train):,}  "
          f"Quiet gradient samples: {len(quiet_g_train):,}")

    # ── Derive thresholds ─────────────────────────────────────────────────────
    print("[5/7] Deriving thresholds …")
    thresh_c = derive_thresholds(flare_c_train, quiet_c_train, feature="counts")
    thresh_g = derive_thresholds(flare_g_train, quiet_g_train, feature="gradient")

    soft_c = apply_soft_boundary(thresh_c, SOFT_MARGIN)
    soft_g = apply_soft_boundary(thresh_g, SOFT_MARGIN)

    # ── Evaluate on test set ──────────────────────────────────────────────────
    print("[6/7] Evaluating on test set …")
    eval_df = None
    if test_dates:
        eval_df = evaluate_thresholds(
            lc_records, hek_df, test_dates,
            thresh_c, thresh_g
        )
        eval_path = os.path.join(OUTPUT_DIR, "test_evaluation_per_day.csv")
        eval_df.to_csv(eval_path, index=False)
        print(f"      Test evaluation written → {eval_path}")
    else:
        print("      (no test dates, skipping evaluation)")

    # ... [Inside main() after Step 5 / Step 6] ...

    # ── Derive thresholds ─────────────────────────────────────────────────────
    print("[5/7] Deriving thresholds …")
    thresh_c = derive_thresholds(flare_c_train, quiet_c_train, feature="counts")
    thresh_g = derive_thresholds(flare_g_train, quiet_g_train, feature="gradient")

    soft_c = apply_soft_boundary(thresh_c, SOFT_MARGIN)
    soft_g = apply_soft_boundary(thresh_g, SOFT_MARGIN)

    # ── Collective Performance Test (New Step) ───────────────────────────────
    print("[5.5/7] Running real-time collective sequence classification test …")
    test_collective_performance(lc_records, hek_df, train_dates, test_dates, thresh_c, thresh_g)

    # ── Evaluate on test set ──────────────────────────────────────────────────
    print("[6/7] Evaluating on test set …")
    # ... [rest of your original main file code handles plotting and saving json] ...

    # ── Report & plots ────────────────────────────────────────────────────────
    print("[7/7] Generating report and plots …")
    print_report(thresh_c, soft_c, thresh_g, soft_g, eval_df, SOFT_MARGIN)

    plot_distributions(flare_c_train, quiet_c_train, thresh_c, soft_c,
                       "counts", OUTPUT_DIR, "counts_train")
    plot_distributions(flare_g_train, quiet_g_train, thresh_g, soft_g,
                       "gradient", OUTPUT_DIR, "gradient_train")
    if eval_df is not None:
        plot_test_metrics(eval_df, OUTPUT_DIR)

    # ── Save summary JSON ─────────────────────────────────────────────────────
    summary = {
        "config": {
            "data_dir":    DATA_DIR,
            "hek_csv":     HEK_CSV,
            "output_dir":  OUTPUT_DIR,
            "soft_margin": SOFT_MARGIN,
            "n_train_days": len(train_dates),
            "n_test_days":  len(test_dates),
        },
        "counts_thresholds": {
            "flare_nonflare": soft_c["flare_nonflare"],
            "class_boundaries": soft_c["class_thresholds"],
            "method": thresh_c["flare_nonflare_method"],
            "quiet_stats": thresh_c["quiet_stats"],
            "class_stats": thresh_c["class_stats"],
        },
        "gradient_thresholds": {
            "flare_nonflare": soft_g["flare_nonflare"],
            "class_boundaries": soft_g["class_thresholds"],
            "method": thresh_g["flare_nonflare_method"],
            "quiet_stats": thresh_g["quiet_stats"],
            "class_stats": thresh_g["class_stats"],
        },
    }
    json_path = os.path.join(OUTPUT_DIR, "threshold_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  [JSON] threshold summary → {json_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
