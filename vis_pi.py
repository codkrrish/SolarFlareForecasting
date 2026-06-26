from astropy.io import fits
import matplotlib.pyplot as plt

filename = "data/pi_files/AL1_SOLEXS_20240202_SDD2_L1.pi"

with fits.open(filename) as hdul:

    hdul.info()

    # Find first binary table extension
    table_hdu = None
    for hdu in hdul:
        if hasattr(hdu, "data") and hdu.data is not None:
            if hasattr(hdu.data, "columns"):
                table_hdu = hdu
                break

    if table_hdu is None:
        raise ValueError("No table found in file")

    data = table_hdu.data
    cols = data.columns.names

    print("\nColumns:")
    print(cols)

    # X axis
    for candidate in ["CHANNEL", "PI"]:
        if candidate in cols:
            x = data[candidate]
            xlabel = candidate
            break
    else:
        raise ValueError("No CHANNEL/PI column found")

    # Y axis
    for candidate in ["COUNTS", "RATE"]:
        if candidate in cols:
            y = data[candidate]
            ylabel = candidate
            break
    else:
        raise ValueError("No COUNTS/RATE column found")

plt.figure(figsize=(10, 5))
plt.step(x, y, where="mid", lw=1)
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(filename.split("/")[-1])
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()