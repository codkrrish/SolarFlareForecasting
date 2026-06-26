from pathlib import Path
from astropy.io import fits
from astropy.time import Time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

root_dir = "data/lcfiles"
output_dir = Path("data/lc_figs")
output_dir.mkdir(parents=True, exist_ok=True)

lc_files = sorted(Path(root_dir).rglob("*.lc"))
if not lc_files:
    print(f"No .lc files found in {root_dir}")
    raise SystemExit

print(f"Found {len(lc_files)} .lc files")

files_per_page = 4
for start_idx in range(0, len(lc_files), files_per_page):
    batch = lc_files[start_idx:start_idx + files_per_page]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()

    for ax, filename in zip(axes, batch):
        try:
            with fits.open(filename) as hdul:
                hdr = hdul[1].header
                data = hdul[1].data
                time_raw = data["TIME"]

                # --- Convert raw TIME to UTC datetimes ---
                mjdref = hdr.get("MJDREFI", 40587) + hdr.get("MJDREFF", 0.0)
                timesys = hdr.get("TIMESYS", "UTC")
                time_mjd = mjdref + time_raw / 86400.0  # seconds → days
                time_utc = Time(time_mjd, format="mjd", scale=timesys.lower())
                time_dt = time_utc.to_datetime()          # Python datetime objects

                if "RATE" in data.columns.names:
                    y = data["RATE"]
                    ylabel = "Count Rate"
                elif "COUNTS" in data.columns.names:
                    y = data["COUNTS"]
                    ylabel = "Counts"
                else:
                    ax.text(
                        0.5, 0.5,
                        "No RATE/COUNTS column",
                        ha="center", va="center"
                    )
                    ax.set_title(filename.name, fontsize=8)
                    continue

            ax.plot(time_dt, y, lw=0.8)
            ax.set_title(filename.name, fontsize=8)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)

            # --- Smart time axis formatting ---
            duration_s = (time_dt[-1] - time_dt[0]).total_seconds()
            if duration_s <= 600:
                fmt = mdates.DateFormatter("%H:%M:%S")
                locator = mdates.AutoDateLocator()
                xlabel = f"Time (UTC)  [{time_dt[0].strftime('%Y-%m-%d')}]"
            elif duration_s <= 86400:
                fmt = mdates.DateFormatter("%H:%M")
                locator = mdates.AutoDateLocator()
                xlabel = f"Time (UTC)  [{time_dt[0].strftime('%Y-%m-%d')}]"
            elif duration_s <= 86400 * 7:
                fmt = mdates.DateFormatter("%b %d %H:%M")
                locator = mdates.AutoDateLocator()
                xlabel = "Date & Time (UTC)"
            else:
                fmt = mdates.DateFormatter("%Y-%m-%d")
                locator = mdates.AutoDateLocator()
                xlabel = "Date (UTC)"

            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(fmt)
            ax.set_xlabel(xlabel, fontsize=7)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)

        except Exception as e:
            ax.text(
                0.5, 0.5,
                f"Error:\n{str(e)}",
                ha="center", va="center"
            )
            ax.set_title(filename.name, fontsize=8)

    for ax in axes[len(batch):]:
        ax.axis("off")

    plt.tight_layout()
    page_num = start_idx // files_per_page + 1
    outfile = output_dir / f"lc_batch_{page_num:03d}.png"
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outfile}")

print("Done!")