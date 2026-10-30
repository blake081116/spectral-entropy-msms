# Spectral Similarity Comparison

## Goal

The goal of this project is to determine which of the most common algorithms for comparing spectral similarity is the most robust and reliable for predicting chemical similarity between two molecules based on their tandem mass spectrometry outputs. The algorithms we will compare are spectral entropy, modified cosine, and spec2vec. By the end of the project we should have an understanding of how these algorithms work, which one is the most accurate for Dr. Thomas’s work, and which one is most computationally efficient.

Reference paper: [Li et al. (2021), Spectral entropy outperforms MS/MS dot product similarity for small-molecule compound identification](references/li-2021-spectral-entropy.pdf).

## Setup

Requires Conda. Run these commands from the repository root:

```bash
conda env create --file environment.yml --yes
conda activate CSCI558
```

The original paper implementation is cloned at `external/SpectralEntropy/` from [YuanyueLi/SpectralEntropy](https://github.com/YuanyueLi/SpectralEntropy). The local clone is at revision `4795aff2de88942e2865ad32ce0538ebfded3b7f` and retains its upstream license and Git history. This directory is ignored by the project repository.

If the source directory is absent in a fresh checkout of this project:

```bash
git clone https://github.com/YuanyueLi/SpectralEntropy.git external/SpectralEntropy
git -C external/SpectralEntropy checkout 4795aff2de88942e2865ad32ce0538ebfded3b7f
```

Install the local source into the activated environment:

```bash
python -m pip install --no-index --no-build-isolation --no-deps --no-cache-dir ./external/SpectralEntropy
mkdir -p data/raw data/processed results/runs
```

Local datasets: [data/raw/](data/raw/). The three source groups are [provided by Dr. Thomas](data/raw/01-dr-thomas-provided/), [public libraries from week 2](data/raw/02-week-02-public-libraries/), and [datasets cited in Li et al. (2021)](data/raw/03-li-2021-paper-data/). These are local paths relative to this repository; raw data remain excluded from Git.

Local data inventory: [manifest](data/manifest.csv) and [source comparison](data/dataset-comparison.md). The Thomas-provided files are byte-identical to their public GNPS counterparts; the three folders preserve provenance and are not independent benchmark splits.

The pilot environment contains the original `spectral_entropy` implementation, NumPy, SciPy, RDKit for metadata-derived compound labels, Matplotlib, and Pytest. The recorded package versions are saved with the pilot outputs. The project goal above covers three algorithms; this executable pilot evaluates the weighted and unweighted entropy variants.

The named Conda environment is `CSCI558`. An earlier local environment at `.venv/` is retained; the commands here use `CSCI558`.

## Run

Activate the environment in each new terminal:

```bash
conda activate CSCI558
```

Check the installed pilot libraries:

```bash
python -c "import spectral_entropy, numpy, scipy; from rdkit import Chem; print('Pilot environment ready')"
```

Run the complete pilot on the two GNPS files supplied by Dr. Thomas:

```bash
python src/run_entropy_pilot.py --config configs/thomas_entropy_pilot.json
```

Run the validation checks:

```bash
python -m pytest -q tests/test_entropy_pilot.py
```

Walk through one real Spinosyn A pair, with raw peaks, normalized intensities, matched peaks, and explicit entropy calculations:

```bash
python src/explain_spinosad_pair.py
```

This pairwise walkthrough writes [aligned peaks](results/metrics/spinosad_walkthrough/matched_peaks.csv), a [numeric summary](results/metrics/spinosad_walkthrough/summary.json), and an [annotated figure](results/figures/spinosad_walkthrough/peak_walkthrough.png). Complete raw and cleaned peak lists remain local in `results/runs/spinosad_walkthrough/`.

Read the [English Spinosyn A walkthrough](docs/spinosad-walkthrough.md) for the input peaks, cosine and entropy formulas, figures, and code navigation.

Run the original upstream example using the installed, compiled package:

```bash
python -c "import runpy; runpy.run_path('external/SpectralEntropy/example.py', run_name='__main__')"
```

Pilot outputs:

- [Summary and recorded environment](results/metrics/thomas_entropy_pilot/pilot_summary.json).
- [CSCI558 environment validation](results/metrics/thomas_entropy_pilot/csci558_environment_validation.json): ten passing tests and a complete pilot rerun with matching scientific results.
- [Spectrum-level metadata and preprocessing audit](results/metrics/thomas_entropy_pilot/spectra_audit.csv).
- [Real pair scores](results/metrics/thomas_entropy_pilot/pair_scores.csv).
- [Retrieval metrics](results/metrics/thomas_entropy_pilot/retrieval_summary.csv) and [query-level results](results/metrics/thomas_entropy_pilot/retrieval_queries.csv).
- [Entropy overview figure](results/figures/thomas_entropy_pilot/entropy_pilot_overview.png) and [example spectral pairs](results/figures/thomas_entropy_pilot/example_spectral_pairs.png).

Raw data, cleaned-peak arrays, full score matrices, and run logs remain local under the ignored `data/` and `results/runs/` folders. Small metrics and figures are available for version control. The pilot uses leave-one-spectrum-out retrieval with SMILES-derived connectivity labels; it is an exploratory check rather than a held-out benchmark or validation on Dr. Thomas's sediment samples.
