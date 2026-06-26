# ─────────────────────────────────────────────
#  CONFIGURATION  ← edit these before running
# ─────────────────────────────────────────────

ROOT_DIR         = ""    # folder containing all .zip files
TARGET_DIR       = "/data"       # destination for lc_files/ and pi_files/
REVIEW_DIR       = "/data"       # destination for flare .lc files (manual check)

MIN_GTI_EXPOSURE = 0        # minimum total GTI exposure in seconds to be considered valid
                             # set to 0 to only use row-count check (recommended default)
FLARE_MULTIPLIER = 5        # spike threshold = FLARE_MULTIPLIER × median(COUNTS)
FLARE_GAP_SEC    = 60       # seconds gap between spikes to treat them as separate events

# ─────────────────────────────────────────────
#  END OF CONFIGURATION
# ─────────────────────────────────────────────

import os
import csv
import gzip
import glob
import shutil
import logging
import zipfile
import tempfile
import traceback

import numpy as np
from astropy.io import fits
from astropy.time import Time



LOG_FILE = os.path.join(TARGET_DIR, "pipeline.log")
CSV_FILE = os.path.join(TARGET_DIR, "flare_labels.csv")

CSV_HEADER = [
    "zip_file", "date", "sdd", "event_no",
    "start_utc", "peak_utc", "end_utc",
    "duration_s", "peak_count", "median_count", "multiplier_used"
]


def setup_logging():
    os.makedirs(TARGET_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def init_csv():
    """Write CSV header if the file does not yet exist."""
    if not os.path.isfile(CSV_FILE):
        with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(CSV_HEADER)


def append_csv_rows(rows: list[dict]):
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER)
        for row in rows:
            w.writerow(row)



def load_gti(gti_gz_path: str) -> tuple[int, float]:
    """
    Returns (n_rows, total_exposure_seconds).
    n_rows == 0  →  no valid GTI intervals (unreliable day for this SDD).
    """
    with gzip.open(gti_gz_path, "rb") as gz:
        data = gz.read()

    # Write to a temp file because astropy needs a seekable file-like object
    tmp = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
    tmp.write(data)
    tmp.flush()
    tmp.close()

    try:
        with fits.open(tmp.name) as hdul:
            gti_hdu = hdul["GTI"]
            n_rows = len(gti_hdu.data) if gti_hdu.data is not None else 0
            exposure = float(gti_hdu.header.get("EXPOSURE", 0.0))
            if n_rows > 0 and exposure == 0.0:
                # Compute from START/STOP if header value is missing
                starts = gti_hdu.data["START"].astype(float)
                stops  = gti_hdu.data["STOP"].astype(float)
                exposure = float(np.sum(stops - starts))
    finally:
        os.unlink(tmp.name)

    return n_rows, exposure


def gti_is_valid(n_rows: int, exposure: float) -> bool:
    if n_rows == 0:
        return False
    if MIN_GTI_EXPOSURE > 0 and exposure < MIN_GTI_EXPOSURE:
        return False
    return True



def load_lc(lc_gz_path: str) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Returns (times, counts, date_obs).
    times  — Unix seconds (float64)
    counts — photon counts per second (float64, NaNs already removed)
    date_obs — 'YYYY-MM-DD' string from FITS header
    """
    with gzip.open(lc_gz_path, "rb") as gz:
        data = gz.read()

    tmp = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
    tmp.write(data)
    tmp.flush()
    tmp.close()

    try:
        with fits.open(tmp.name) as hdul:
            rate_hdu = hdul["RATE"]
            raw_times  = rate_hdu.data["TIME"].astype(float)
            raw_counts = rate_hdu.data["COUNTS"].astype(float)
            date_obs   = rate_hdu.header.get("DATE-OBS", "")[:10]
    finally:
        os.unlink(tmp.name)

    valid = ~np.isnan(raw_counts)
    return raw_times[valid], raw_counts[valid], date_obs


def detect_flares(times: np.ndarray, counts: np.ndarray) -> list[dict]:
    """
    Returns a list of flare event dicts.  Empty list → no flare.

    Each dict:
        start_unix, peak_unix, end_unix,
        duration_s, peak_count, median_count
    """
    if len(counts) == 0:
        return []

    median_val = float(np.median(counts))
    if median_val <= 0:
        return []

    threshold = FLARE_MULTIPLIER * median_val
    spike_mask = counts > threshold

    if not spike_mask.any():
        return []

    spike_times  = times[spike_mask]
    spike_counts = counts[spike_mask]

    gaps = np.diff(spike_times)
    boundaries = np.where(gaps > FLARE_GAP_SEC)[0]

    slices = []
    prev = 0
    for b in boundaries:
        slices.append((prev, b + 1))
        prev = b + 1
    slices.append((prev, len(spike_times)))

    events = []
    for s, e in slices:
        et = spike_times[s:e]
        ec = spike_counts[s:e]
        peak_idx = int(np.argmax(ec))
        events.append({
            "start_unix":   float(et[0]),
            "peak_unix":    float(et[peak_idx]),
            "end_unix":     float(et[-1]),
            "duration_s":   float(et[-1] - et[0]),
            "peak_count":   float(ec[peak_idx]),
            "median_count": median_val,
        })

    return events


def unix_to_iso(unix_sec: float) -> str:
    return Time(unix_sec, format="unix").iso  



def decompress_gz(src_gz: str, dst: str):
    """
    Decompress a .gz file directly to dst (the final file path, no .gz extension).
    Creates parent directories as needed.
    """
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with gzip.open(src_gz, "rb") as f_in, open(dst, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)


def gz_stem(gz_path: str) -> str:
    """
    Return the filename inside a .gz archive, i.e. strip the trailing '.gz'.
    e.g. 'AL1_SOLEXS_20240415_SDD2_L1.lc.gz'  →  'AL1_SOLEXS_20240415_SDD2_L1.lc'
    """
    return os.path.basename(gz_path)[:-3]   # remove '.gz'



def process_sdd(
    sdd_name: str,          # 'SDD1' or 'SDD2'
    sdd_dir: str,           # path to extracted SDD folder
    date_str: str,          # 'YYYYMMDD' from zip filename
    zip_basename: str,      # for logging / CSV
    zip_stem: str,          # zip filename without .zip  (used as top folder in output)
) -> bool:
    """
    Process one SDD subfolder.
    Returns True if this SDD was valid (GTI non-empty), False otherwise.

    Output paths follow the pattern:
        TARGET_DIR/lc_files/<zip_stem>/<sdd_name>/<original_lc_filename>
        TARGET_DIR/pi_files/<zip_stem>/<sdd_name>/<original_pi_filename>
        REVIEW_DIR/<zip_stem>/<sdd_name>/<original_lc_filename>
    """
    log = logging.getLogger()

    gti_files = glob.glob(os.path.join(sdd_dir, "*.gti.gz"))
    if not gti_files:
        log.warning(f"[{zip_basename}] [{sdd_name}] No .gti.gz found — skipping SDD.")
        return False

    gti_path = gti_files[0]

    try:
        n_rows, exposure = load_gti(gti_path)
    except Exception:
        log.error(f"[{zip_basename}] [{sdd_name}] Failed to read GTI:\n{traceback.format_exc()}")
        return False

    if not gti_is_valid(n_rows, exposure):
        log.info(
            f"[{zip_basename}] [{sdd_name}] GTI UNRELIABLE "
            f"(rows={n_rows}, exposure={exposure:.0f}s) — skipping SDD."
        )
        return False

    log.info(
        f"[{zip_basename}] [{sdd_name}] GTI VALID "
        f"(rows={n_rows}, exposure={exposure:.0f}s)."
    )


    lc_files = glob.glob(os.path.join(sdd_dir, "*.lc.gz"))
    if not lc_files:
        log.warning(
            f"[{zip_basename}] [{sdd_name}] GTI valid but no .lc.gz found "
            f"(possible saturation without GTI flag) — skipping SDD."
        )
        return True   # GTI was valid; LC absence is logged but SDD counts as visited

    lc_src = lc_files[0]
    lc_filename = gz_stem(lc_src)
    lc_dst = os.path.join(TARGET_DIR, "lc_files", lc_filename)
    decompress_gz(lc_src, lc_dst)
    log.info(f"[{zip_basename}] [{sdd_name}] LC extracted → {lc_dst}")


    try:
        times, counts, date_obs = load_lc(lc_src)
    except Exception:
        log.error(f"[{zip_basename}] [{sdd_name}] Failed to read LC:\n{traceback.format_exc()}")
        return True

    events = detect_flares(times, counts)

    if not events:
        log.info(
            f"[{zip_basename}] [{sdd_name}] No flare detected "
            f"(median={np.median(counts):.1f}, threshold={FLARE_MULTIPLIER}×median)."
        )
        return True


    log.info(
        f"[{zip_basename}] [{sdd_name}] FLARE DETECTED — "
        f"{len(events)} event(s) on {date_obs}."
    )


    csv_rows = []
    for i, ev in enumerate(events, start=1):
        csv_rows.append({
            "zip_file":       zip_basename,
            "date":           date_obs,
            "sdd":            sdd_name,
            "event_no":       i,
            "start_utc":      unix_to_iso(ev["start_unix"]),
            "peak_utc":       unix_to_iso(ev["peak_unix"]),
            "end_utc":        unix_to_iso(ev["end_unix"]),
            "duration_s":     f"{ev['duration_s']:.0f}",
            "peak_count":     f"{ev['peak_count']:.1f}",
            "median_count":   f"{ev['median_count']:.2f}",
            "multiplier_used": FLARE_MULTIPLIER,
        })
        log.info(
            f"  Event {i}: start={unix_to_iso(ev['start_unix'])}  "
            f"peak={unix_to_iso(ev['peak_unix'])}  "
            f"peak_count={ev['peak_count']:.0f}  duration={ev['duration_s']:.0f}s"
        )
    append_csv_rows(csv_rows)


    pi_files = glob.glob(os.path.join(sdd_dir, "*.pi.gz"))
    if pi_files:
        pi_src = pi_files[0]
        pi_filename = gz_stem(pi_src) 
        pi_dst = os.path.join(TARGET_DIR, "pi_files", pi_filename)
        decompress_gz(pi_src, pi_dst)
        log.info(f"[{zip_basename}] [{sdd_name}] PI extracted → {pi_dst}")
    else:
        log.warning(f"[{zip_basename}] [{sdd_name}] Flare detected but no .pi.gz found.")

    review_dst = os.path.join(REVIEW_DIR, lc_filename)
    decompress_gz(lc_src, review_dst)
    log.info(f"[{zip_basename}] [{sdd_name}] LC extracted to review → {review_dst}")

    return True



def process_zip(zip_path: str):
    zip_basename = os.path.basename(zip_path)
    log = logging.getLogger()
    log.info(f"{'─'*60}")
    log.info(f"Processing: {zip_basename}")

    # Extract date string from filename  (AL1_SLX_L1_YYYYMMDD_v*)
    try:
        date_str = zip_basename.split("_")[3]   # 'YYYYMMDD'
        int(date_str)                            # basic sanity check
    except (IndexError, ValueError):
        log.error(f"[{zip_basename}] Cannot parse date from filename — skipping zip.")
        return

    tmp_dir = tempfile.mkdtemp(prefix="solex_")
    try:
        # Extract zip
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_dir)


        top_dirs = [
            d for d in os.listdir(tmp_dir)
            if os.path.isdir(os.path.join(tmp_dir, d))
        ]
        if not top_dirs:
            log.error(f"[{zip_basename}] Zip extracted but no top-level directory found.")
            return

        root = os.path.join(tmp_dir, top_dirs[0])

        # Process each SDD in order: SDD1 first, then SDD2
        zip_stem = os.path.splitext(zip_basename)[0]   # strip '.zip'
        any_valid = False
        for sdd_name in ["SDD1", "SDD2"]:
            sdd_dir = os.path.join(root, sdd_name)
            if not os.path.isdir(sdd_dir):
                log.warning(f"[{zip_basename}] [{sdd_name}] Directory not found in zip — skipping.")
                continue
            valid = process_sdd(sdd_name, sdd_dir, date_str, zip_basename, zip_stem)
            any_valid = any_valid or valid

        if not any_valid:
            log.info(f"[{zip_basename}] Both SDDs unreliable — no data extracted.")

    except zipfile.BadZipFile:
        log.error(f"[{zip_basename}] Bad zip file — skipping.")
    except Exception:
        log.error(f"[{zip_basename}] Unexpected error:\n{traceback.format_exc()}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)




def main():
    setup_logging()
    init_csv()

    os.makedirs(REVIEW_DIR, exist_ok=True)

    log = logging.getLogger()
    log.info("=" * 60)
    log.info("SoLEXS Dataset Pipeline — START")
    log.info(f"  ROOT_DIR         : {ROOT_DIR}")
    log.info(f"  TARGET_DIR       : {TARGET_DIR}")
    log.info(f"  REVIEW_DIR       : {REVIEW_DIR}")
    log.info(f"  MIN_GTI_EXPOSURE : {MIN_GTI_EXPOSURE}s")
    log.info(f"  FLARE_MULTIPLIER : {FLARE_MULTIPLIER}×median")
    log.info(f"  FLARE_GAP_SEC    : {FLARE_GAP_SEC}s")
    log.info("=" * 60)

    zip_paths = sorted(glob.glob(os.path.join(ROOT_DIR, "AL1_SLX_L1_*.zip")))

    if not zip_paths:
        log.warning(f"No zip files matching 'AL1_SLX_L1_*.zip' found in {ROOT_DIR}")
        return

    log.info(f"Found {len(zip_paths)} zip file(s) to process.\n")

    total      = len(zip_paths)
    n_reliable = 0    # at least one SDD valid
    n_flare    = 0    # at least one SDD had a flare  (approximate; tracked via CSV)
    n_error    = 0

    flare_dates_before = _count_csv_rows()

    for i, zp in enumerate(zip_paths, start=1):
        log.info(f"[{i}/{total}]")
        try:
            process_zip(zp)
            # Count as reliable if lc_files directory for this zip was created
            zip_stem = os.path.splitext(os.path.basename(zp))[0]
            lc_zip_dir = os.path.join(TARGET_DIR, "lc_files", zip_stem)
            if os.path.isdir(lc_zip_dir):
                n_reliable += 1
        except Exception:
            log.error(f"Top-level error on {zp}:\n{traceback.format_exc()}")
            n_error += 1

    flare_events_total = _count_csv_rows() - flare_dates_before

    log.info("=" * 60)
    log.info("SoLEXS Dataset Pipeline — COMPLETE")
    log.info(f"  Total zips processed  : {total}")
    log.info(f"  Days with ≥1 valid SDD: {n_reliable}")
    log.info(f"  Flare events logged   : {flare_events_total}")
    log.info(f"  Errors                : {n_error}")
    log.info(f"  Label CSV             : {CSV_FILE}")
    log.info(f"  Log file              : {LOG_FILE}")
    log.info("=" * 60)


def _count_csv_rows() -> int:
    """Count data rows currently in the flare CSV (excluding header)."""
    if not os.path.isfile(CSV_FILE):
        return 0
    with open(CSV_FILE, encoding="utf-8") as f:
        return max(0, sum(1 for _ in f) - 1)


if __name__ == "__main__":
    main()