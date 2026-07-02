"""
Real-time solar flare detector for AL1 SoLEXS .lc (FITS light-curve) data.

Pipeline
--------
1. load_lc_file()          - read a single day's .lc FITS file -> (unix_time, counts)
2. load_ground_truth()     - read hek_flares.csv, resolve the lc_date off-by-one bug,
                              return one row per physical flare with a *canonical* date.
3. FlareDetector            - causal (real-time-safe), two-signal probabilistic detector:
      (a) Welch's t-test between a short "foreground" window and a long
          "background" window whose statistics FREEZE while a flare is suspected
          (so the rising flare doesn't poison its own baseline).
      (b) A trained Bayesian log-likelihood-ratio (LLR) on the foreground/background
          ratio, fitted on TRAIN data using BOTH the flare seconds and the quiet
          seconds (this is the piece your mean/median/z-only attempts were missing -
          you were characterizing flares but never explicitly characterizing "quiet").
   The two signals are combined into a single posterior confidence in [0,1].
   Entry into "flare" state requires confidence > 1-alpha for >= persist_s seconds.
   Exit requires confidence < 1-beta for >= exit_persist_s seconds (beta > alpha,
   hysteresis so the state doesn't flicker at the boundary).
4. tune()                  - grid search alpha/beta/window sizes on the TRAIN split only,
                              optimizing event-level F1 against HEK ground truth.
5. evaluate()               - event-level (and point-level) precision/recall/F1,
                              broken down by GOES class, plus detection latency.

Everything that touches the ground-truth CSV happens only in fit()/tune()/evaluate().
predict() takes nothing but a .lc file path, as required for real-time deployment.
"""

from __future__ import annotations

import glob
import itertools
import os
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from astropy.io import fits
# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

# MJD 40587 = 1970-01-01T00:00:00 UTC, i.e. the .lc TIME column IS a Unix/POSIX timestamp.
MJD_EPOCH_UNIX_OFFSET = 0.0

SECONDS_PER_DAY = 86400


# ========================================================================================
# 1. Loading the .lc FITS files
# ========================================================================================

def load_lc_file(path: str, max_gap_fill_s: int = 5) -> pd.DataFrame:
    """
    Read one AL1_SOLEXS_YYYYMMDD_SDD2_L1.lc file.

    Returns a DataFrame indexed by second-of-day [0, 86399] with columns:
        unix_time : float, POSIX seconds (TIME column, MJD40587 epoch == Unix epoch)
        counts    : float64, NaN gaps linearly interpolated (only short gaps; a flag
                    column marks which samples were imputed so they can be excluded
                    from baseline statistics if desired)
        imputed   : bool
    """


    with fits.open(path) as hdul:
        data = None
        for hdu in hdul:
            if hdu.data is not None and "COUNTS" in (hdu.columns.names if hasattr(hdu, "columns") else []):
                data = hdu.data
                break
        if data is None:
            raise ValueError(f"No COUNTS column found in {path}")
        time = np.asarray(data["TIME"], dtype=np.float64)
        counts = np.asarray(data["COUNTS"], dtype=np.float64)

    df = pd.DataFrame({"unix_time": time, "counts": counts}).sort_values("unix_time")
    df = df.reset_index(drop=True)

    n_nan = df["counts"].isna().sum()
    df["imputed"] = df["counts"].isna()
    if n_nan:
        df["counts"] = df["counts"].interpolate(
            method="linear", limit=max_gap_fill_s, limit_direction="both"
        )
        remaining = df["counts"].isna().sum()
        if remaining:
            warnings.warn(
                f"{path}: {remaining} NaN samples remain after interpolation "
                f"(gap longer than {max_gap_fill_s}s) - forward/back filled."
            )
            df["counts"] = df["counts"].ffill().bfill()

    return df


def date_from_lc_filename(path: str) -> str:
    """Extract YYYYMMDD from 'AL1_SOLEXS_YYYYMMDD_SDD2_L1.lc'."""
    base = os.path.basename(path)
    parts = base.split("_")
    for p in parts:
        if len(p) == 8 and p.isdigit():
            return p
    raise ValueError(f"Could not parse date from filename: {path}")


# ========================================================================================
# 2. Ground truth (HEK) loading + date-boundary resolution
# ========================================================================================

@dataclass
class FlareEvent:
    date: str            # canonical YYYYMMDD (file this flare STARTS in)
    start: pd.Timestamp
    peak: pd.Timestamp
    end: pd.Timestamp
    goes_class: str       # e.g. 'M1.5'

    @property
    def cls_letter(self) -> str:
        return self.goes_class[0]

    @property
    def cls_multiplier(self) -> float:
        try:
            return float(self.goes_class[1:])
        except ValueError:
            return np.nan


def load_ground_truth(csv_path: str) -> list[FlareEvent]:
    """
    Load hek_flares.csv and resolve the lc_date off-by-one issue.

    Resolution strategy (verified against the actual data):
    Some flares appear TWICE - once tagged with lc_date = start_day and once with
    lc_date = start_day + 1 (HEK's next-day rollover bug). A handful of flares only
    have the wrong (+1) row, with no duplicate. In both cases, the fix is the same:
    ignore lc_date entirely, derive the canonical date directly from event_starttime,
    then drop exact duplicates on (start, peak, end, class). This automatically:
      - collapses duplicate rows down to a single physical flare, and
      - re-assigns the lone mis-tagged rows to the day they actually start on.
    `flux_estimate_wm2` is dropped per spec (excluded feature).
    """
    df = pd.read_csv(csv_path)
    for col in ("event_starttime", "event_peaktime", "event_endtime"):
        # Strip an explicit "+00:00" tz suffix if present, then parse as naive UTC.
        df[col] = df[col].astype(str).str.replace(r"\+00:00$", "", regex=True)
        df[col] = pd.to_datetime(df[col])

    df["canonical_date"] = df["event_starttime"].dt.strftime("%Y%m%d")

    before = len(df)
    df = df.drop_duplicates(
        subset=["event_starttime", "event_peaktime", "event_endtime", "fl_goescls"]
    ).reset_index(drop=True)
    removed = before - len(df)
    if removed:
        print(f"[load_ground_truth] Resolved lc_date rollover bug: "
              f"removed {removed} duplicate rows; reassigned lone mis-tagged rows "
              f"to their true start date.")

    events = [
        FlareEvent(
            date=row.canonical_date,
            start=row.event_starttime,
            peak=row.event_peaktime,
            end=row.event_endtime,
            goes_class=row.fl_goescls,
        )
        for row in df.itertuples()
    ]
    events.sort(key=lambda e: e.start)
    return events


def events_overlapping_file(events: list[FlareEvent], date_str: str) -> list[FlareEvent]:
    """
    Flares relevant to a given file's day. Includes flares whose canonical date is
    `date_str` (they start in this file) AND flares from the PREVIOUS day whose end
    time spills into this file (midnight-crossing flares), so labeling covers the
    full physical event even though it's keyed to its start file.
    """
    day = pd.Timestamp(date_str)
    prev_day_str = (day - timedelta(days=1)).strftime("%Y%m%d")
    out = [e for e in events if e.date == date_str]
    out += [e for e in events if e.date == prev_day_str and e.end.normalize() >= day]
    return out


def label_seconds(lc_df: pd.DataFrame, date_str: str, events: list[FlareEvent]) -> np.ndarray:
    """Boolean array, True where second-of-day falls inside any relevant flare [start, end]."""
    t = pd.to_datetime(lc_df["unix_time"], unit="s", utc=True).dt.tz_localize(None)
    label = np.zeros(len(lc_df), dtype=bool)
    for e in events_overlapping_file(events, date_str):
        label |= (t >= e.start).to_numpy() & (t <= e.end).to_numpy()
    return label


# ========================================================================================
# 3. Statistical detector
# ========================================================================================

def _causal_rolling_mean_var(x: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Causal (right-aligned) rolling mean/var/n via cumulative sums. O(n)."""
    n = len(x)
    cs = np.concatenate([[0.0], np.cumsum(x)])
    cs2 = np.concatenate([[0.0], np.cumsum(x * x)])
    idx = np.arange(1, n + 1)
    lo = np.maximum(idx - window, 0)
    count = (idx - lo).astype(np.float64)
    s = cs[idx] - cs[lo]
    s2 = cs2[idx] - cs2[lo]
    mean = s / count
    var = np.maximum(s2 / count - mean ** 2, 1e-6)
    return mean, var, count


def _masked_causal_rolling_mean_var(x: np.ndarray, mask: np.ndarray, window: int
                                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same as above but only samples where mask==True contribute (the 'frozen baseline')."""
    xm = x * mask
    x2m = (x * x) * mask
    m = mask.astype(np.float64)
    n = len(x)
    cs, cs2, csm = (np.concatenate([[0.0], np.cumsum(a)]) for a in (xm, x2m, m))
    idx = np.arange(1, n + 1)
    lo = np.maximum(idx - window, 0)
    n_eff = csm[idx] - csm[lo]
    s = cs[idx] - cs[lo]
    s2 = cs2[idx] - cs2[lo]
    n_eff_safe = np.maximum(n_eff, 1.0)
    mean = s / n_eff_safe
    var = np.maximum(s2 / n_eff_safe - mean ** 2, 1e-6)
    return mean, var, n_eff


def welch_t_pvalue(fg_mean, fg_var, fg_n, bg_mean, bg_var, bg_n) -> tuple[np.ndarray, np.ndarray]:
    """One-sided Welch's t-test: is fg_mean significantly GREATER than bg_mean? Vectorized."""
    bg_n_safe = np.maximum(bg_n, 2.0)
    fg_n_safe = np.maximum(fg_n, 2.0)
    se2_fg = fg_var / fg_n_safe
    se2_bg = bg_var / bg_n_safe
    se = np.sqrt(se2_fg + se2_bg)
    se = np.maximum(se, 1e-9)
    t = (fg_mean - bg_mean) / se
    # Welch-Satterthwaite degrees of freedom
    num = (se2_fg + se2_bg) ** 2
    den = (se2_fg ** 2) / (fg_n_safe - 1) + (se2_bg ** 2) / (bg_n_safe - 1)
    dof = np.clip(num / np.maximum(den, 1e-12), 1.0, 1e6)
    p = stats.t.sf(t, dof)  # one-sided: P(T > t)
    return t, p


@dataclass
class DetectorConfig:
    w_short_s: int = 8          # foreground / "is something happening right now" window
    w_long_s: int = 7200        # background / "quiet sun baseline" window (2 hr default)
    alpha: float = 1e-4         # entry significance level (stricter -> fewer false alarms)
    beta: float = 1e-2          # exit significance level (looser -> requires strong evidence to end)
    persist_s: int = 6          # consecutive seconds confidence must exceed entry thresh
    exit_persist_s: int = 20    # consecutive seconds confidence must fall below exit thresh
    freeze_refine_iters: int = 3  # iterations of baseline-freezing refinement
    use_bayes: bool = True
    bayes_weight: float = 0.5   # blend weight between t-test posterior and Bayes LLR posterior


@dataclass
class TrainedStats:
    """Learned from TRAIN data only: distributions of log-ratio under flare vs quiet seconds."""
    mu_flare: float = 2.0
    sigma_flare: float = 1.0
    mu_quiet: float = 0.0
    sigma_quiet: float = 0.3
    prior_flare: float = 0.02


class FlareDetector:
    def __init__(self, config: Optional[DetectorConfig] = None):
        self.config = config or DetectorConfig()
        self.trained: Optional[TrainedStats] = None

    # ---- feature computation (no ground truth needed -> real-time safe) ----

    def _compute_features(self, counts: np.ndarray) -> dict:
        cfg = self.config
        mask = np.ones(len(counts), dtype=bool)  # unfrozen initially
        fg_mean, fg_var, fg_n = _causal_rolling_mean_var(counts, cfg.w_short_s)

        bg_mean = bg_var = bg_n = None
        conf = None
        for _ in range(cfg.freeze_refine_iters):
            bg_mean, bg_var, bg_n = _masked_causal_rolling_mean_var(counts, mask, cfg.w_long_s)
            t_stat, p_val = welch_t_pvalue(fg_mean, fg_var, fg_n, bg_mean, bg_var, bg_n)
            conf_t = 1.0 - p_val

            if cfg.use_bayes and self.trained is not None:
                ratio = np.clip(fg_mean, 1e-6, None) / np.clip(bg_mean, 1e-6, None)
                logr = np.log(ratio)
                ts = self.trained
                ll_flare = stats.norm.logpdf(logr, ts.mu_flare, ts.sigma_flare)
                ll_quiet = stats.norm.logpdf(logr, ts.mu_quiet, ts.sigma_quiet)
                prior_odds = np.log(ts.prior_flare / max(1 - ts.prior_flare, 1e-9))
                llr = ll_flare - ll_quiet + prior_odds
                conf_bayes = 1.0 / (1.0 + np.exp(-llr))
                conf = cfg.bayes_weight * conf_bayes + (1 - cfg.bayes_weight) * conf_t
            else:
                conf = conf_t

            # refine freeze mask: don't let samples we're now confident are "flare"
            # contribute to the background baseline in the next iteration.
            new_mask = conf < (1 - cfg.alpha)
            if np.array_equal(new_mask, mask):
                break
            mask = new_mask

        return dict(fg_mean=fg_mean, bg_mean=bg_mean, bg_var=bg_var, confidence=conf)

    def _hysteresis_state(self, confidence: np.ndarray) -> np.ndarray:
        """Turn a per-second confidence trace into a boolean flare-state trace with
        entry/exit persistence (a small, cheap sequential pass - real-time compatible)."""
        cfg = self.config
        n = len(confidence)
        state = np.zeros(n, dtype=bool)
        in_flare = False
        above_run = 0
        below_run = 0
        enter_thresh = 1 - cfg.alpha
        exit_thresh = 1 - cfg.beta
        for i in range(n):
            c = confidence[i]
            if not in_flare:
                above_run = above_run + 1 if c > enter_thresh else 0
                if above_run >= cfg.persist_s:
                    in_flare = True
                    below_run = 0
            else:
                below_run = below_run + 1 if c < exit_thresh else 0
                if below_run >= cfg.exit_persist_s:
                    in_flare = False
                    above_run = 0
            state[i] = in_flare
        return state

    def predict(self, lc_path: str) -> pd.DataFrame:
        """
        Real-time-safe: only needs the .lc file. Returns a DataFrame with
        unix_time, counts, confidence, flare_state (bool) per second.
        """
        lc = load_lc_file(lc_path)
        feats = self._compute_features(lc["counts"].to_numpy())
        state = self._hysteresis_state(feats["confidence"])
        return pd.DataFrame({
            "unix_time": lc["unix_time"],
            "counts": lc["counts"],
            "bg_mean": feats["bg_mean"],
            "confidence": feats["confidence"],
            "flare_state": state,
        })

    @staticmethod
    def intervals_from_state(pred_df: pd.DataFrame, date_str: str) -> list[tuple]:
        """Collapse a boolean flare_state trace into (start_ts, end_ts, peak_confidence) tuples."""
        s = pred_df["flare_state"].to_numpy()
        t = pd.to_datetime(pred_df["unix_time"], unit="s", utc=True).dt.tz_localize(None).to_numpy()
        conf = pred_df["confidence"].to_numpy()
        out = []
        i = 0
        n = len(s)
        while i < n:
            if s[i]:
                j = i
                while j < n and s[j]:
                    j += 1
                out.append((pd.Timestamp(t[i]), pd.Timestamp(t[j - 1]), float(conf[i:j].max())))
                i = j
            else:
                i += 1
        return out

    # ---- training (uses ground truth; NOT used inside predict()) ----

    def fit(self, train_files: list[str], events: list[FlareEvent]):
        """Learn the flare-vs-quiet log-ratio distributions from labeled TRAIN data."""
        cfg = self.config
        flare_logr, quiet_logr = [], []
        n_flare_sec = 0
        n_total_sec = 0
        for path in train_files:
            date_str = date_from_lc_filename(path)
            lc = load_lc_file(path)
            counts = lc["counts"].to_numpy()
            label = label_seconds(lc, date_str, events)
            fg_mean, _, _ = _causal_rolling_mean_var(counts, cfg.w_short_s)
            mask = ~label  # background computed from genuinely-quiet seconds only, for fitting
            bg_mean, _, bg_n = _masked_causal_rolling_mean_var(counts, mask, cfg.w_long_s)
            valid = bg_n > cfg.w_long_s * 0.3
            ratio = np.clip(fg_mean, 1e-6, None) / np.clip(bg_mean, 1e-6, None)
            logr = np.log(ratio)
            flare_logr.append(logr[label & valid])
            quiet_logr.append(logr[(~label) & valid])
            n_flare_sec += int(label.sum())
            n_total_sec += len(label)

        flare_logr = np.concatenate(flare_logr) if flare_logr else np.array([1.0])
        quiet_logr = np.concatenate(quiet_logr) if quiet_logr else np.array([0.0])

        self.trained = TrainedStats(
            mu_flare=float(np.mean(flare_logr)),
            sigma_flare=float(max(np.std(flare_logr), 1e-3)),
            mu_quiet=float(np.mean(quiet_logr)),
            sigma_quiet=float(max(np.std(quiet_logr), 1e-3)),
            prior_flare=float(max(n_flare_sec / max(n_total_sec, 1), 1e-4)),
        )
        print(f"[fit] quiet: logr ~ N({self.trained.mu_quiet:.3f}, {self.trained.sigma_quiet:.3f}^2)  "
              f"flare: logr ~ N({self.trained.mu_flare:.3f}, {self.trained.sigma_flare:.3f}^2)  "
              f"prior_flare={self.trained.prior_flare:.4f}")
        return self


# ========================================================================================
# 4. Matching detections to ground truth + evaluation metrics
# ========================================================================================

def match_events(gt_events: list[FlareEvent], detections: list[tuple], tolerance_s: int = 60):
    """
    Overlap-based matching (a detection matches a GT flare if their [start,end]
    intervals overlap, with `tolerance_s` slack on each side).
    Returns (matched_pairs, unmatched_gt, unmatched_det, per_gt_first_detection_latency_s).
    Each GT flare matches at most one detection (greedy, earliest-overlap first) and
    vice versa, avoiding double counting.
    """
    gt_sorted = sorted(gt_events, key=lambda e: e.start)
    det_sorted = sorted(detections, key=lambda d: d[0])
    gt_used = [False] * len(gt_sorted)
    det_used = [False] * len(det_sorted)
    matches = []
    latencies = []

    for gi, e in enumerate(gt_sorted):
        e_start = e.start - timedelta(seconds=tolerance_s)
        e_end = e.end + timedelta(seconds=tolerance_s)
        for di, (d_start, d_end, _peak) in enumerate(det_sorted):
            if det_used[di]:
                continue
            if d_start <= e_end and d_end >= e_start:
                gt_used[gi] = True
                det_used[di] = True
                matches.append((e, det_sorted[di]))
                latencies.append((d_start - e.start).total_seconds())
                break

    unmatched_gt = [e for e, used in zip(gt_sorted, gt_used) if not used]
    unmatched_det = [d for d, used in zip(det_sorted, det_used) if not used]
    return matches, unmatched_gt, unmatched_det, latencies


def evaluate(gt_events: list[FlareEvent], detections: list[tuple], tolerance_s: int = 60) -> dict:
    """Event-level precision/recall/F1 overall and broken down by GOES class letter."""
    matches, missed, false_pos, latencies = match_events(gt_events, detections, tolerance_s)
    tp, fn, fp = len(matches), len(missed), len(false_pos)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    by_class = {}
    for letter in ["B", "C", "M", "X"]:
        gt_c = [e for e in gt_events if e.cls_letter == letter]
        if not gt_c:
            continue
        m_c, miss_c, _, _ = match_events(gt_c, detections, tolerance_s)
        by_class[letter] = {
            "n_gt": len(gt_c),
            "matched": len(m_c),
            "recall": len(m_c) / len(gt_c) if gt_c else 0.0,
        }

    return {
        "n_gt_flares": len(gt_events),
        "n_detections": len(detections),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "median_latency_s": float(np.median(latencies)) if latencies else None,
        "mean_latency_s": float(np.mean(latencies)) if latencies else None,
        "by_class": by_class,
    }


def print_report(name: str, metrics: dict):
    print(f"\n=== {name} ===")
    print(f"Ground-truth flares : {metrics['n_gt_flares']}")
    print(f"Detections raised   : {metrics['n_detections']}")
    print(f"True positives      : {metrics['true_positives']}")
    print(f"False positives     : {metrics['false_positives']}")
    print(f"False negatives     : {metrics['false_negatives']}")
    print(f"Precision           : {metrics['precision']:.3f}")
    print(f"Recall              : {metrics['recall']:.3f}")
    print(f"F1                  : {metrics['f1']:.3f}")
    if metrics["median_latency_s"] is not None:
        print(f"Median detection latency : {metrics['median_latency_s']:.1f} s "
              f"(negative = detected before HEK's listed start)")
    print("Recall by GOES class:")
    for letter, d in metrics["by_class"].items():
        print(f"  {letter}: {d['matched']}/{d['n_gt']}  ({d['recall']:.1%})")


# ========================================================================================
# 5. Tuning (grid search on TRAIN only)
# ========================================================================================

def run_on_files(detector: FlareDetector, files: list[str]) -> list[tuple]:
    """Run predict() over a list of files and return all detected intervals (with dates attached
    via the timestamps themselves, so no extra bookkeeping needed)."""
    all_intervals = []
    for path in files:
        date_str = date_from_lc_filename(path)
        pred = detector.predict(path)
        all_intervals.extend(FlareDetector.intervals_from_state(pred, date_str))
    return all_intervals



def tune(train_files: list[str], events: list[FlareEvent], param_grid: Optional[dict] = None,
         tolerance_s: int = 60, base_config: Optional[DetectorConfig] = None) -> DetectorConfig:
    """
    Grid search alpha/beta/persist_s on the TRAIN split, maximizing event-level F1.
    Windows (w_short_s, w_long_s) are fit once beforehand (Bayes stats) using base_config's
    windows, then held fixed during the alpha/beta search for tractability; widen the grid
    below if you want to search window sizes too.
    """
    param_grid = param_grid or {
        "alpha": [1e-6, 1e-5, 1e-4, 1e-3],
        "beta": [1e-3, 1e-2, 5e-2],
        "persist_s": [3, 6, 10],
        "exit_persist_s": [15, 30, 60],
    }
    base_config = base_config or DetectorConfig()

    gt_train = [e for e in events if any(date_from_lc_filename(f) == e.date or
                                          date_from_lc_filename(f) == (pd.Timestamp(e.date) + timedelta(days=1)).strftime("%Y%m%d")
                                          for f in train_files)]

    # Fit the Bayes component once (window sizes fixed for this call)
    fitter = FlareDetector(base_config)
    fitter.fit(train_files, events)

    best_f1, best_cfg = -1.0, base_config
    keys = list(param_grid.keys())
    for combo in itertools.product(*param_grid.values()):
        cfg = DetectorConfig(**{**base_config.__dict__, **dict(zip(keys, combo))})
        det = FlareDetector(cfg)
        det.trained = fitter.trained
        dets = run_on_files(det, train_files)
        m = evaluate(gt_train, dets, tolerance_s)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_cfg = cfg
            print(f"[tune] new best F1={best_f1:.3f}  cfg={dict(zip(keys, combo))}")

    print(f"\n[tune] BEST on train: F1={best_f1:.3f}  config={best_cfg}")
    return best_cfg

def tune_full(train_files, events, tolerance_s=60,
               window_grid=None, file_subsample=1):
    """
    Stage 1: coarse (w_short_s, w_long_s) search with fixed alpha/beta/persist,
             on every `file_subsample`-th train file (fast).
    Stage 2: full alpha/beta/persist grid search (existing tune()) using the
             winning window sizes, on the FULL train set.
    Set file_subsample=2 or 3 if Stage 1 is still too slow.
    """
    window_grid = window_grid or {
        "w_short_s": [8, 30, 60, 120, 180],
        "w_long_s": [1800, 3600, 7200, 10800],
    }
    probe_kwargs = dict(alpha=1e-4, beta=1e-2, persist_s=5, exit_persist_s=20)
    probe_files = train_files[::file_subsample] if file_subsample > 1 else train_files

    best_f1, best_windows = -1.0, None
    for w_short, w_long in itertools.product(window_grid["w_short_s"], window_grid["w_long_s"]):
        if w_short >= w_long:
            continue
        cfg = DetectorConfig(w_short_s=w_short, w_long_s=w_long, **probe_kwargs)
        det = FlareDetector(cfg)
        det.fit(probe_files, events)
        dets = run_on_files(det, probe_files)
        gt_probe = [e for e in events if any(
            date_from_lc_filename(f) == e.date or
            date_from_lc_filename(f) == (pd.Timestamp(e.date) + timedelta(days=1)).strftime("%Y%m%d")
            for f in probe_files)]
        m = evaluate(gt_probe, dets, tolerance_s)
        print(f"[tune_windows] w_short={w_short:>4}s w_long={w_long:>5}s -> "
              f"F1={m['f1']:.3f} P={m['precision']:.3f} R={m['recall']:.3f}")
        if m["f1"] > best_f1:
            best_f1, best_windows = m["f1"], (w_short, w_long)

    w_short, w_long = best_windows
    print(f"\n[tune_windows] BEST windows: w_short={w_short}s w_long={w_long}s (F1={best_f1:.3f})")

    base_cfg = DetectorConfig(w_short_s=w_short, w_long_s=w_long)
    return tune(train_files, events, tolerance_s=tolerance_s, base_config=base_cfg)
# ========================================================================================
# 6. Convenience: end-to-end pipeline
# ========================================================================================

def chronological_split(files: list[str], test_frac: float = 0.2) -> tuple[list[str], list[str]]:
    files_sorted = sorted(files, key=date_from_lc_filename)
    n_test = max(1, int(len(files_sorted) * test_frac))
    return files_sorted[:-n_test], files_sorted[-n_test:]


def run_pipeline(lc_dir: str, hek_csv: str, test_frac: float = 0.2,
                  tolerance_s: int = 60, do_tune: bool = True):
    files = sorted(glob.glob(os.path.join(lc_dir, "*_SDD2_L1.lc")))
    if not files:
        raise FileNotFoundError(f"No .lc files found in {lc_dir}")
    train_files, test_files = chronological_split(files, test_frac)
    print(f"Found {len(files)} files -> {len(train_files)} train / {len(test_files)} test")

    events = load_ground_truth(hek_csv)

    if do_tune:
        cfg = tune_full(train_files, events, tolerance_s=tolerance_s)
    else:
        cfg = DetectorConfig()

    detector = FlareDetector(cfg)
    detector.fit(train_files, events)

    for name, fset in [("TRAIN", train_files), ("TEST", test_files)]:
        dates_in_split = {date_from_lc_filename(f) for f in fset}
        # a flare belongs to this split if the file it starts in (or the midnight-spillover
        # next file) is inside the split
        gt_split = [e for e in events if e.date in dates_in_split or
                    (pd.Timestamp(e.date) + timedelta(days=1)).strftime("%Y%m%d") in dates_in_split]
        dets = run_on_files(detector, fset)
        # --- diagnostic: does recall improve a lot with a looser match tolerance? ---
        for tol in (60, 120, 300, 600):
            m = evaluate(gt_split, dets, tolerance_s=tol)
            print(f"[{name}] tolerance={tol:>4}s  precision={m['precision']:.3f}  "
                f"recall={m['recall']:.3f}  f1={m['f1']:.3f}  "
                f"TP={m['true_positives']} FP={m['false_positives']} FN={m['false_negatives']}")
        metrics = evaluate(gt_split, dets, tolerance_s)
        print_report(name, metrics)

    return detector, cfg


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SoLEXS real-time flare detector")
    parser.add_argument("--lc-dir", type=str, default="data/solex", help="Directory containing .lc files")
    parser.add_argument("--hek-csv", type=str, default="data/hek_flares.csv")
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--no-tune", action="store_true")
    args = parser.parse_args()

    if args.lc_dir is None:
        print("No --lc-dir given; run demo_synthetic.py for a self-test with synthetic data,")
        print("or pass --lc-dir /path/to/your/lc/files --hek-csv hek_flares.csv")
    else:
        run_pipeline(args.lc_dir, args.hek_csv, test_frac=args.test_frac, do_tune=not args.no_tune)
