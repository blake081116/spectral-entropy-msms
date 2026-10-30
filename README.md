# Spectral Similarity Comparison

## Goal

The goal of this project is to determine which of the most common algorithms for comparing spectral similarity is the most robust and reliable for predicting chemical similarity between two molecules based on their tandem mass spectrometry outputs. The algorithms we will compare are spectral entropy, modified cosine, and spec2vec. By the end of the project we should have an understanding of how these algorithms work, which one is the most accurate for Dr. Thomas’s work, and which one is most computationally efficient.

## Setup

Requires Conda. Run these commands from the repository root:

```bash
conda create --prefix ./.venv -c conda-forge -c bioconda python=3.12 pip matchms spec2vec -y
conda activate ./.venv
python -m pip install ms_entropy
mkdir -p data/raw data/processed results/runs
```

This environment includes [MS Entropy](https://pypi.org/project/ms-entropy/), [matchms](https://pypi.org/project/matchms/) for modified cosine, and [Spec2Vec](https://pypi.org/project/spec2vec/).

## Run

Activate the environment in each new terminal:

```bash
conda activate ./.venv
```

Check the installed libraries:

```bash
python -c "import ms_entropy, matchms, spec2vec; print('Environment ready')"
```

Run a standalone spectral entropy similarity example with illustrative peak lists:

```bash
python - <<'PYTHON'
import numpy as np
from ms_entropy import calculate_entropy_similarity

query = np.array([[100.0, 10.0], [150.0, 50.0], [200.0, 100.0]], dtype=np.float32)
reference = np.array([[100.01, 20.0], [150.01, 40.0], [250.0, 80.0]], dtype=np.float32)

score = calculate_entropy_similarity(
    query, reference, ms2_tolerance_in_da=0.02, clean_spectra=True
)
print(f"Entropy similarity: {score:.6f}")
PYTHON
```

The example uses the [MS Entropy API](https://msentropy.readthedocs.io/en/latest/classical_entropy_similarity.html). It runs independently of a dataset or project pipeline. Spec2Vec scoring requires a trained Word2Vec model; this repository does not include one.
