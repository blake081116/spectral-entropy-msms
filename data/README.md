# Data

No dataset has been downloaded as part of workspace organization.

- `raw/`: original downloaded spectra; ignored by Git.
- `processed/`: filtered or transformed data; ignored by Git.
- `manifest.csv`: create when data are available. Track source URL, release/version, filename, download date, SHA-256, spectrum count, and provenance/filter notes.
- `splits.csv`: create when the benchmark split is defined. Record spectrum ID, compound key, query/reference role, development/test assignment, and random seed.

The manifest and split files are allowed by `.gitignore` so the benchmark can be reproduced without committing the spectra. Paths such as `data/raw/...` are relative to the repository root.
