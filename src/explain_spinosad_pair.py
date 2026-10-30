"""Walk through one real Spinosyn A pair using the existing pilot functions.

Run from the repository root: python src/explain_spinosad_pair.py
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.special import xlogy
from scipy.stats import entropy
from spectral_entropy import tools

try:
    from .pilot_data import read_mgf, prepare_spectrum
    from .run_entropy_pilot import score_pair, write_csv
except ImportError:
    from pilot_data import read_mgf, prepare_spectrum
    from run_entropy_pilot import score_pair, write_csv

ROOT = Path(__file__).resolve().parents[1]
QUERY_ID = "CCMSLIB00001058649"
REFERENCE_ID = "CCMSLIB00001058908"
METRICS = ROOT / "results/metrics/spinosad_walkthrough"
FIGURES = ROOT / "results/figures/spinosad_walkthrough"
LOCAL = ROOT / "results/runs/spinosad_walkthrough"
os.environ.setdefault("MPLCONFIGDIR", str(LOCAL / "matplotlib-cache"))


def draw_figure(query, reference, probe):
    """Show the raw peak, normalized peaks, and an actual cross-spectrum match."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    green, orange = "#147d64", "#c36b35"
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.2), constrained_layout=True)
    a, b, c, d = axes.flat
    a.vlines(query.raw_peaks[:, 0], 0, query.raw_peaks[:, 1], color=green, lw=0.7)
    base = query.raw_peaks[np.argmax(query.raw_peaks[:, 1])]
    a.annotate(f"One raw peak\nm/z = {base[0]:.6f}\nintensity = {base[1]:,.0f}",
               xy=base, xytext=(0.40, 0.68), textcoords="axes fraction",
               arrowprops={"arrowstyle":"->", "color":"#333333"}, fontsize=9)
    a.axvline(query.precursor_mz-1.6, ls="--", color=orange, lw=1, label="Precursor cutoff")
    a.set(title=f"1. Spectrum A: {len(query.raw_peaks):,} raw peak records", xlabel="m/z", ylabel="Raw intensity")
    a.legend(frameon=False, fontsize=8, loc="upper right")
    b.vlines(query.peaks[:, 0], 0, query.peaks[:, 1], color=green, lw=1.3)
    b.annotate(f"p = {probe['query_p']:.6f}\n{100*probe['query_p']:.2f}% of retained intensity",
               xy=(probe["query_mz"], probe["query_p"]), xytext=(0.46, 0.72), textcoords="axes fraction",
               arrowprops={"arrowstyle":"->", "color":"#333333"}, fontsize=9)
    b.set(title=f"2. Spectrum A: {len(query.peaks)} cleaned peaks; sum(p) = 1", xlabel="m/z", ylabel="Normalized intensity p")
    c.vlines(query.peaks[:, 0], 0, query.peaks[:, 1], color=green, lw=1.2, label="A: query")
    c.vlines(reference.peaks[:, 0], 0, -reference.peaks[:, 1], color=orange, lw=1.2, label="B: reference (inverted)")
    c.axhline(0, color="#666666", lw=0.7)
    c.set(title="3. Two cleaned spectra, aligned by m/z", xlabel="m/z", ylabel="Normalized intensity (reference inverted)")
    c.legend(frameon=False, fontsize=8)
    for key, color, label in [("query", green, "A"), ("reference", orange, "B")]:
        d.vlines(probe[key+"_mz"], 0, probe[key+"_p"], color=color, lw=2,
                 label=f"{label}: m/z {probe[key+'_mz']:.6f}; p {probe[key+'_p']:.6f}")
    center = (probe["query_mz"]+probe["reference_mz"])/2
    d.set(xlim=(center-0.005, center+0.005), ylim=(0, 0.9), xlabel="m/z (zoomed)", ylabel="Normalized intensity",
          title=f"4. A real match: delta m/z = {probe['delta_mz']:.6f} < 0.05")
    d.ticklabel_format(axis="x", useOffset=False)
    d.legend(frameon=False, fontsize=8, loc="upper left")
    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Spinosad (Spinosyn A), [M+H]+: two real GNPS spectra", fontsize=15)
    for extension in ("png", "svg"):
        fig.savefig(FIGURES / f"peak_walkthrough.{extension}", dpi=180)
    plt.close(fig)


def main():
    for directory in (METRICS, FIGURES, LOCAL):
        directory.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "configs/thomas_entropy_pilot.json").read_text())

    # Step 1: Read the two named records from the original MGF, checking its SHA-256.
    source = ROOT / config["input_files"][0]
    with source.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    assert checksum == config["expected_sha256"][source.name], "Source checksum mismatch"
    records = {s.spectrum_id:s for s in read_mgf(source) if s.spectrum_id in (QUERY_ID, REFERENCE_ID)}
    query, reference = records[QUERY_ID], records[REFERENCE_ID]

    # Step 2: Reuse the pilot cleaner: precursor cutoff, centroiding, noise, and L1.
    for label, spectrum in (("query", query), ("reference", reference)):
        prepare_spectrum(spectrum, config)
        assert spectrum.label_usable
        for stage, peaks in (("raw", spectrum.raw_peaks), ("clean", spectrum.peaks)):
            write_csv(LOCAL / f"{label}_{stage}_peaks.csv",
                      [{"mz":float(mz), "intensity":float(value)} for mz,value in peaks])
    assert query.compound_key == reference.compound_key
    assert query.fingerprint != reference.fingerprint

    # Step 3: Align the cleaned spectra using the same upstream peak matcher.
    tolerance = config["ms2_tolerance_da"]
    matched = tools.match_peaks_in_spectra(query.peaks.copy(), reference.peaks.copy(), ms2_da=tolerance)
    # Preserve the original m/z coordinates in the explanatory table. This fixed
    # pair has one reference peak per aligned bin; the general matcher can sum more.
    mz_info = np.zeros((len(matched), 4), dtype=np.float64)
    for i, (mz, intensity_a, intensity_b) in enumerate(matched):
        if intensity_a > 0:
            mz_info[i, :2] = (mz, intensity_a)
        if intensity_b > 0:
            nearby = reference.peaks[np.abs(reference.peaks[:, 0]-mz) <= tolerance]
            assert len(nearby) == 1, "This walkthrough requires one reference peak per bin"
            mz_info[i, 2:] = nearby[0]
    np.testing.assert_allclose(mz_info[:, [1, 3]], matched[:, [1, 2]], rtol=1e-6, atol=1e-8)

    # Step 4: Calculate each entropy explicitly; xlogy makes 0*log(0) equal zero.
    p, q = matched[:, 1].astype(np.float64), matched[:, 2].astype(np.float64)
    p, q = p/p.sum(), q/q.sum()
    mixture = (p+q)/2
    terms = [-xlogy(v, v) for v in (p, q, mixture)]
    sa, sb, sm = [float(t.sum()) for t in terms]
    manual_unweighted = 1-(2*sm-sa-sb)/np.log(4)
    weights = [0.25+0.25*s if s<3 else 1.0 for s in (sa, sb)]
    pw, qw = p**weights[0], q**weights[1]
    pw, qw = pw/pw.sum(), qw/qw.sum()
    manual_weighted = float(1-(2*entropy((pw+qw)/2)-entropy(pw)-entropy(qw))/np.log(4))

    # Step 5: Check these calculations against the exact functions used by the pilot.
    weighted, unweighted = score_pair(query.peaks, reference.peaks, tolerance)
    np.testing.assert_allclose([manual_weighted, manual_unweighted], [weighted, unweighted], atol=5e-6, rtol=0)
    rows = []
    for i, info in enumerate(mz_info):
        both = p[i]>0 and q[i]>0
        rows.append({"row":i+1, "query_mz":float(info[0]) if p[i]>0 else "",
                     "reference_mz":float(info[2]) if q[i]>0 else "",
                     "delta_mz":abs(float(info[0]-info[2])) if both else "",
                     "query_p":float(p[i]), "reference_p":float(q[i]), "mixture_m":float(mixture[i]),
                     "minus_p_ln_p":float(terms[0][i]), "minus_q_ln_q":float(terms[1][i]),
                     "minus_m_ln_m":float(terms[2][i]),
                     "status":"matched" if both else "query_only" if p[i]>0 else "reference_only"})
    write_csv(METRICS / "matched_peaks.csv", rows)
    probe = max((r for r in rows if r["status"]=="matched"), key=lambda r:r["query_p"])
    draw_figure(query, reference, probe)
    summary = {"compound":"Spinosad (Spinosyn A)", "adduct":"[M+H]+", "query_id":QUERY_ID,
               "reference_id":REFERENCE_ID, "source_file":config["input_files"][0], "source_sha256":checksum,
               "precursor_mz":query.precursor_mz, "ms2_tolerance_da":tolerance,
               "precursor_cutoff_mz":query.precursor_mz-config["precursor_exclusion_da"],
               "raw_peak_records":{"query":len(query.raw_peaks), "reference":len(reference.raw_peaks)},
               "clean_peak_records":{"query":len(query.peaks), "reference":len(reference.peaks)},
               "aligned_rows":len(rows), "matched_rows":sum(r["status"]=="matched" for r in rows),
               "query_only_rows":sum(r["status"]=="query_only" for r in rows),
               "reference_only_rows":sum(r["status"]=="reference_only" for r in rows),
               "entropy_nats":{"query":sa, "reference":sb, "mixture":sm},
               "intensity_exponents":{"query":weights[0], "reference":weights[1]},
               "manual_similarity":{"weighted":manual_weighted, "unweighted":float(manual_unweighted)},
               "upstream_similarity":{"weighted":weighted, "unweighted":unweighted}, "highlighted_match":probe,
               "scope":"A real pairwise comparison already included in pair_scores.csv; this script does not perform a library-wide identification search.",
               "formula_validation":"Both explicit calculations agree with the original implementation within 5e-6."}
    (METRICS / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    print(json.dumps(summary, indent=2))
    print("Aligned peaks: results/metrics/spinosad_walkthrough/matched_peaks.csv")
    print("Figure: results/figures/spinosad_walkthrough/peak_walkthrough.png")
    print("Full raw/clean peak lists: results/runs/spinosad_walkthrough/")


if __name__ == "__main__":
    main()
