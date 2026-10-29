# Spectral Entropy for MS/MS

**Exploring Spectral Entropy for MS/MS Spectral Matching**

CSCI 558 — Bioinformatics Algorithms · University of Montana · Fall 2026.

## Current goal

Run spectral entropy similarity on a small set of MS/MS spectra first. Understand the required inputs, preprocessing, parameters, and scores before building a comparison with other similarity methods.

## Status

Repository scaffold only. No implementation, dataset, dependency environment, or experiment results are included yet. The immediate plan is an initial spectral entropy run; a broader course-project scope can be confirmed separately.

## Phase 1 — Run spectral entropy

- Set up a Python environment for the published `ms_entropy` implementation.
- Load a few MS/MS spectra and inspect their peak lists and metadata.
- Run spectral entropy similarity on selected pairs of spectra.
- Inspect the scores and document the preprocessing and parameters used.
- Save a small, reproducible example with input provenance, package versions, and the exact run command.

## Phase 2 — Compare methods later

After the initial run works, define a comparison with cosine or another method. Choose the dataset, shared preprocessing, retrieval metrics, runtime measurement, and robustness experiments at that stage.

The initial milestone is a working spectral entropy example; a full multi-method benchmark is a later step.

## Layout

```text
spectral-entropy-msms/
├── README.md
├── .gitignore
├── src/               # Implementation
├── tests/             # Algorithm validation
├── configs/           # Preprocessing and experiment settings
├── notebooks/         # Data exploration and demos
├── data/
│   ├── README.md      # Data provenance conventions
│   ├── raw/           # Local data; ignored by Git
│   └── processed/     # Local derived data; ignored by Git
└── results/
    ├── README.md      # Output conventions
    ├── runs/          # Generated run outputs; ignored by Git
    ├── metrics/       # Selected small tables to keep in Git
    └── figures/       # Selected figures to keep in Git
```

The empty code/config/notebook/test folders contain `.gitkeep` placeholders. Dependencies and run commands will be added when implementation begins.

## References

- [Spectral entropy similarity — Li et al. (2021)](https://www.nature.com/articles/s41592-021-01331-z)
- [matchms CosineGreedy](https://matchms.readthedocs.io/en/latest/api/matchms.similarity.CosineGreedy.html)
- [MS Entropy documentation](https://msentropy.readthedocs.io/en/latest/classical_entropy_similarity.html)
- [MassBank data repository](https://github.com/MassBank/MassBank-data)

## GitHub setup

Repository: [blake081116/spectral-entropy-msms](https://github.com/blake081116/spectral-entropy-msms).

Remote URL: `https://github.com/blake081116/spectral-entropy-msms.git`.

Use this folder as the local repository root. After the initial setup, make further commits from here:

```bash
git status
git add .
git commit -m "Describe the change"
git push
```

To work on another computer, clone the repository into a new or empty directory. The raw-data and generated-output folders are ignored by Git; recreate or populate them when the experiments begin.

This folder is the repository root. Meeting notes, personal documents, and archived course files are stored in sibling workspace folders. This README is self-contained so it also works after cloning the repository elsewhere.
