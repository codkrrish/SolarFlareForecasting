import re
import time
import pandas as pd
from pathlib import Path
from sunpy.net import Fido, attrs as a



# ── Configuration ─────────────────────────────────────────────────────────────
LC_DIR      = "data/lcfiles"
OUTPUT_CSV  = "data/hek_flares_dedup.csv"
MIN_CLASS   = "B"       # minimum flare class to keep (A/B/C/M/X)
RETRY_LIMIT = 3         # number of retries on network failure
RETRY_WAIT  = 10        # seconds to wait between retries


WANTED_COLS = [
    "event_starttime",
    "event_peaktime",
    "event_endtime",
    "fl_goescls"
]

CLASS_FLOOR = {"A": 1e-8, "B": 1e-7, "C": 1e-6, "M": 1e-5, "X": 1e-4}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Scan directory and extract unique dates
# ─────────────────────────────────────────────────────────────────────────────
def extract_dates_from_lc_files(lc_dir: str) -> list[str]:
    pattern   = re.compile(r"AL1_SOLEXS_(\d{8})_SDD\d_L1\.lc", re.IGNORECASE)
    lc_path   = Path(lc_dir)
    all_files = sorted(lc_path.rglob("*.lc"))

    if not all_files:
        raise FileNotFoundError(f"No .lc files found in: {lc_dir}")

    dates = set()
    unmatched = []
    for f in all_files:
        m = pattern.search(f.name)
        if m:
            dates.add(m.group(1))
        else:
            unmatched.append(f.name)

    if unmatched:
        print(f"  Warning: {len(unmatched)} files didn't match naming pattern:")
        for name in unmatched[:5]:
            print(f"    {name}")
        if len(unmatched) > 5:
            print(f"    ... and {len(unmatched) - 5} more")

    sorted_dates = sorted(dates)
    print(f"\n  Found {len(all_files)} .lc files → {len(sorted_dates)} unique dates")
    return sorted_dates


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Fetch HEK FL events for one date (full 24-hour window)
# ─────────────────────────────────────────────────────────────────────────────
def hek_table_to_dataframe(hek_result) -> pd.DataFrame:
    # Tier 1: direct to_pandas (most SunPy versions)
    try:
        return hek_result.to_pandas()
    except AttributeError:
        pass

    # Tier 2: wrap in astropy Table first
    try:
        from astropy.table import Table
        return Table(hek_result).to_pandas()
    except Exception:
        pass

    # Tier 3: manual column-by-column extraction (guaranteed fallback)
    cols = {}
    for col in hek_result.colnames:
        try:
            cols[col] = list(hek_result[col])
        except Exception:
            pass
    return pd.DataFrame(cols)


def fetch_one_date(date_str: str) -> pd.DataFrame | None:
    # Build full-day time window
    t_start = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}T00:00:00"
    t_end   = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}T23:59:59"

    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            result = Fido.search(
                a.Time(t_start, t_end),
                a.hek.EventType("FL"),
            )
            hek_result = result["hek"]

            if len(hek_result) == 0:
                return None

            df = hek_table_to_dataframe(hek_result)

            # Keep only wanted columns that exist
            available = [c for c in WANTED_COLS if c in df.columns]
            df = df[available].copy()

            # Tag with source date for traceability
            df.insert(0, "lc_date", date_str)
            return df

        except Exception as e:
            print(f"\n    Attempt {attempt}/{RETRY_LIMIT} failed: {e}")
            if attempt < RETRY_LIMIT:
                print(f"    Retrying in {RETRY_WAIT}s …")
                time.sleep(RETRY_WAIT)

    return None   # all retries exhausted


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Filter by GOES class
# ─────────────────────────────────────────────────────────────────────────────
def cls_to_flux(cls: str) -> float:
    """Convert GOES class string (e.g. 'M1.2') to approximate flux in W/m²."""
    if not isinstance(cls, str) or len(cls) < 2:
        return 0.0
    letter = cls[0].upper()
    try:
        return CLASS_FLOOR.get(letter, 0.0) * float(cls[1:])
    except ValueError:
        return 0.0


def filter_dataframe(df: pd.DataFrame, min_class: str = "B") -> pd.DataFrame:
    """
    Keep only GOES-detected flares at or above min_class.
    Adds a numeric flux_estimate column.
    """
    if df.empty:
        return df

    # Add numeric flux column
    if "fl_goescls" in df.columns:
        df["flux_estimate_wm2"] = df["fl_goescls"].apply(cls_to_flux)
        floor_val = CLASS_FLOOR.get(min_class.upper(), 0.0)
        df = df[df["flux_estimate_wm2"] >= floor_val]

    # Keep only GOES observatory detections
    if "obs_observatory" in df.columns:
        df = df[df["obs_observatory"].str.contains("GOES", case=False, na=False)]

    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Print final summary
# ─────────────────────────────────────────────────────────────────────────────
def print_summary(df: pd.DataFrame, n_dates: int, failed: list[str]):
    print("\n" + "═" * 60)
    print("  BATCH FETCH SUMMARY")
    print("═" * 60)
    print(f"  Unique dates processed : {n_dates}")
    print(f"  Failed / no data dates : {len(failed)}")
    if failed:
        print(f"  Failed dates           : {', '.join(failed[:10])}"
              + (" ..." if len(failed) > 10 else ""))
    print(f"  Total flare events     : {len(df)}")

    if "fl_goescls" in df.columns and not df.empty:
        df["_cls"] = df["fl_goescls"].str[0].str.upper()
        dist = df["_cls"].value_counts().sort_index()
        print(f"  Class distribution     :")
        for cls, count in dist.items():
            print(f"      {cls}-class : {count:>5}")
        df.drop(columns=["_cls"], inplace=True)

    if "lc_date" in df.columns and not df.empty:
        dates_with_data = df["lc_date"].nunique()
        print(f"  Dates with ≥1 flare    : {dates_with_data} / {n_dates}")

    print("═" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "═" * 60)
    print("  SoLEXS Batch HEK Flare Fetcher")
    print("═" * 60)

    # Step 1: get unique dates
    print(f"\n[1/3] Scanning: {LC_DIR}")
    unique_dates = extract_dates_from_lc_files(LC_DIR)

    # Step 2: load existing CSV to support resume
    output_path  = Path(OUTPUT_CSV)
    already_done = set()
    existing_dfs = []

    if output_path.exists():
        existing = pd.read_csv(output_path, dtype=str)
        if "lc_date" in existing.columns:
            already_done = set(existing["lc_date"].unique())
            existing_dfs.append(existing)
            print(f"\n  Resuming — {len(already_done)} dates already in CSV, "
                  f"skipping those.")

    # Dates still to fetch
    pending = [d for d in unique_dates if d not in already_done]
    print(f"  Dates to fetch: {len(pending)} / {len(unique_dates)}")

    if not pending:
        print("\n  All dates already fetched. Nothing to do.")
        print(f"  Output: {OUTPUT_CSV}")
        return

    # Step 3: fetch each date
    print(f"\n[2/3] Fetching HEK FL events …\n")
    new_dfs  = []
    failed   = []

    for i, date_str in enumerate(pending, 1):
        pretty = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        print(f"  [{i:>3}/{len(pending)}] {pretty} … ", end="", flush=True)

        df = fetch_one_date(date_str)

        if df is None:
            print("no events / failed.")
            failed.append(date_str)
            continue

        df_filtered = filter_dataframe(df, MIN_CLASS)
        n_raw       = len(df)
        n_kept      = len(df_filtered)
        print(f"{n_raw} events fetched → {n_kept} kept (≥{MIN_CLASS}-class, GOES only)")

        if not df_filtered.empty:
            new_dfs.append(df_filtered)

        # Polite pause to avoid hammering the HEK server
        time.sleep(1)

    # Step 4: combine and save
    print(f"\n[3/3] Combining and saving …")
    all_dfs = existing_dfs + new_dfs

    if not all_dfs:
        print("  No data to save.")
        return

    combined = pd.concat(all_dfs, ignore_index=True)

    # Sort by peak time for clean ordering
    if "event_peaktime" in combined.columns:
        combined.sort_values("event_peaktime", inplace=True)
        combined.reset_index(drop=True, inplace=True)

    combined.to_csv(OUTPUT_CSV, index=False)
    print(f"  Saved → {OUTPUT_CSV}")

    print_summary(combined, len(unique_dates), failed)

    # Save a list of failed dates for reference
    if failed:
        failed_path = Path(OUTPUT_CSV).stem + "_failed_dates.txt"
        Path(failed_path).write_text("\n".join(failed))
        print(f"  Failed dates saved → {failed_path}")
        print("  Re-run the script to retry failed dates automatically.\n")

def dedup():
    df = pd.read_csv("data/hek_flares.csv", dtype={"lc_date": str})
    df["event_peaktime"] = pd.to_datetime(df["event_peaktime"], utc=True)
    df["event_starttime"] = pd.to_datetime(df["event_starttime"], utc=True)
    df["event_endtime"] = pd.to_datetime(df["event_endtime"], utc=True)

# Sort by peak time
    df = df.sort_values("event_peaktime").reset_index(drop=True)

# Deduplicate: merge events whose peaks are within 5 minutes of each other
# Keep the one with highest flux_estimate (most reliable class)
    df["peak_group"] = (
    df["event_peaktime"]
    .diff()
    .gt(pd.Timedelta("5min"))
    .cumsum()
)

    deduped = (
    df.sort_values("flux_estimate_wm2", ascending=False)
      .groupby(["lc_date", "peak_group"], as_index=False)
      .first()
      .sort_values("event_peaktime")
      .reset_index(drop=True)
)
    # Sort by event_starttime (earliest first) and save to CSV
    deduped = deduped.sort_values("event_starttime").reset_index(drop=True)
    deduped.to_csv("data/hek_flares_dedup.csv", index=False)
    
    print(f"Before dedup: {len(df)}  |  After: {len(deduped)}")


if __name__ == "__main__":
    main()
    #dedup()
    