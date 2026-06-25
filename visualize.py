import matplotlib.pyplot as plt
from astropy.io import fits

with fits.open("/home/k/Downloads/AL1_SLX_L1_20240407_v1.0/SDD2/AL1_SOLEXS_20240407_SDD2_L1.lc") as hdul:
    gti = hdul["GTI"].data

start = gti["START"]
stop = gti["STOP"]

fig, ax = plt.subplots(figsize=(10, 2))

for s, e in zip(start, stop):
    ax.plot([s, e], [1, 1], lw=8)

ax.set_xlabel("Time")
ax.set_yticks([])
ax.set_title("Good Time Intervals (GTI)")

plt.show()