"""Plot the measured micromodel flooding results held in `Results/`.

`Results/` contains the per-run output of the segmentation pipeline for the nine
micromodel flooding experiments analysed for the Journal of Molecular Liquids
study: for each run, the oil area ratio and the derived recovery rate at hourly
intervals over the ten-hour acquisition.

Writes `figures/measured-recovery.png`.
"""

import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = "Results"
FIG_DIR = "figures"

# Distinct colours so nine curves stay separable, with the run label alongside.
PALETTE = [
    "#1f4e79", "#2e75b6", "#7f7f7f", "#548235", "#a9d18e",
    "#c00000", "#ed7d31", "#7030a0", "#bf9000",
]


def read_series(path):
    """Parse lines of the form `3h: 0.2131` or `3h: 43.2751%`."""
    values = []
    for line in open(path):
        match = re.match(r"\s*(\d+)h:\s*([-\d.]+)%?", line)
        if match:
            values.append((int(match.group(1)), float(match.group(2))))
    values.sort()
    return np.array([h for h, _ in values]), np.array([v for _, v in values])


def load_runs():
    runs = {}
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "ratios_*.txt"))):
        run = os.path.basename(path)[len("ratios_"):-len(".txt")]
        hours, ratios = read_series(path)
        rec_path = os.path.join(RESULTS_DIR, f"recovery_rates_{run}.txt")
        _, recovery = read_series(rec_path)
        runs[run] = (hours, ratios, recovery)
    return runs


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    runs = load_runs()
    if not runs:
        raise SystemExit(f"no result files found in {RESULTS_DIR}/")

    fig, (left, right) = plt.subplots(1, 2, figsize=(13.5, 5.2))

    for (run, (hours, ratios, recovery)), colour in zip(runs.items(), PALETTE):
        left.plot(hours, ratios, "o-", color=colour, lw=1.8, ms=4.5, label=f"run {run}")
        right.plot(hours, recovery, "s-", color=colour, lw=1.8, ms=4.5, label=f"run {run}")

    left.set_title("Oil area ratio over time", fontweight="bold")
    left.set_ylabel("Oil area / total area")
    right.set_title("Tertiary oil recovery over time", fontweight="bold")
    right.set_ylabel("Recovery (%)")

    for ax in (left, right):
        ax.set_xlabel("Time (hours)")
        ax.set_xticks(range(1, 11))
        ax.grid(True, alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    right.legend(ncol=3, fontsize=9, frameon=False, loc="lower right")

    fig.suptitle(
        "Measured micromodel flooding results, nine runs",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "measured-recovery.png")
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)

    print("\nfinal recovery by run:")
    for run, (_, _, recovery) in runs.items():
        print(f"  run {run:>2}: {recovery[-1]:6.2f} %")


if __name__ == "__main__":
    main()
