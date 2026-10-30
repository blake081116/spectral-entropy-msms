"""Create English-labeled figures for the Thomas spectral-entropy pilot."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "results/runs/thomas_entropy_pilot/matplotlib-cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LABELS = {
    "same_connectivity_same_adduct": "Same connectivity, same adduct",
    "different_connectivity_random": "Different connectivity, random",
    "different_connectivity_mass_matched": "Different connectivity, within 10 ppm",
}
COLORS = {"same_connectivity_same_adduct": "#147d64", "different_connectivity_random": "#5470a6",
          "different_connectivity_mass_matched": "#be6b27"}


def plot_pilot(spectra, pair_rows, example_rows, directory):
    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    upper_entropy = max(6.0, np.ceil(max(s.spectral_entropy for s in spectra if s.usable)*2)/2)
    entropy_bins = np.linspace(0, upper_entropy, 34)
    for name, color in [("GNPS-COLLECTIONS-PESTICIDES-POSITIVE.mgf", "#147d64"),
                        ("GNPS-NIH-CLINICALCOLLECTION1.mgf", "#5470a6")]:
        values = [s.spectral_entropy for s in spectra if s.usable and s.source_file == name]
        assert np.histogram(values, bins=entropy_bins)[0].sum() == len(values)
        axes[0].hist(values, bins=entropy_bins, alpha=0.65,
                     label="Pesticides" if "PESTICIDES" in name else "NIH clinical", color=color)
    axes[0].set(xlabel="Spectral entropy after cleaning (nats)", ylabel="Number of spectra",
                title=f"{sum(s.usable for s in spectra):,} usable spectra from two GNPS libraries")
    axes[0].axvline(3, color="#555555", ls="--", lw=1, label="Weighting cutoff S = 3")
    axes[0].legend(frameon=False, fontsize=8)
    for kind in LABELS:
        selected = [r for r in pair_rows if r["pair_type"] == kind]
        axes[1].scatter([r["entropy_unweighted"] for r in selected],
                        [r["entropy_weighted"] for r in selected],
                        s=12, alpha=0.45, color=COLORS[kind], label=f"{LABELS[kind]} (n={len(selected)})")
    axes[1].plot([0, 1], [0, 1], ls="--", color="#888888", lw=1)
    axes[1].set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02), xlabel="Unweighted entropy similarity",
                ylabel="Weighted entropy similarity", title="Same peaks, two entropy variants")
    axes[1].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(directory / f"entropy_pilot_overview.{ext}", dpi=180)
    plt.close(fig)

    by_id = {s.spectrum_id:s for s in spectra}
    fig, axes = plt.subplots(len(example_rows), 1, figsize=(12, 3.2*len(example_rows)), squeeze=False)
    for axis, row in zip(axes[:, 0], example_rows, strict=True):
        query, reference = by_id[row["query_id"]], by_id[row["reference_id"]]
        q, r = query.peaks, reference.peaks
        axis.vlines(q[:, 0], 0, q[:, 1]/q[:, 1].max(), color="#147d64", lw=1.2, label=f"Query: {query.spectrum_id}")
        axis.vlines(r[:, 0], 0, -r[:, 1]/r[:, 1].max(), color="#b65a45", lw=1.2, label=f"Reference: {reference.spectrum_id}")
        axis.axhline(0, color="#555555", lw=0.7)
        axis.set(xlabel="m/z", ylabel="Base-peak-scaled intensity",
                 title=f"{LABELS[row['pair_type']]} | weighted={row['entropy_weighted']:.3f}; unweighted={row['entropy_unweighted']:.3f}",
                 ylim=(-1.12, 1.12))
        axis.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(directory / f"example_spectral_pairs.{ext}", dpi=180)
    plt.close(fig)
