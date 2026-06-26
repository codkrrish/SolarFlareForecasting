from pathlib import Path
from astropy.io import fits
import matplotlib.pyplot as plt

# Root directory containing .lc files
root_dir = r"data/lcfiles"

# Find all .lc files recursively
lc_files = sorted(Path(root_dir).rglob("*.lc"))

if not lc_files:
    print(f"No .lc files found in {root_dir}")
    raise SystemExit

print(f"Found {len(lc_files)} .lc files")

# Number of plots per page
files_per_page = 4

for start_idx in range(0, len(lc_files), files_per_page):

    batch = lc_files[start_idx:start_idx + files_per_page]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()

    for ax, filename in zip(axes, batch):

        try:
            # Read-only access
            with fits.open(filename) as hdul:

                data = hdul[1].data

                time = data["TIME"]

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

            ax.plot(time, y, lw=0.8)
            ax.set_title(filename.name, fontsize=8)
            ax.set_xlabel("Time")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)

        except Exception as e:
            ax.text(
                0.5, 0.5,
                f"Error:\n{str(e)}",
                ha="center", va="center"
            )
            ax.set_title(filename.name, fontsize=8)

    # Hide unused subplots on the last page
    for ax in axes[len(batch):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    # Optional: wait before showing next 4 files
    if start_idx + files_per_page < len(lc_files):
        input(
            f"Showing files {start_idx + 1}-{start_idx + len(batch)} "
            f"of {len(lc_files)}. Press Enter for next 4..."
        )