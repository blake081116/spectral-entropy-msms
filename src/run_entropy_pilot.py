"""Run the original spectral-entropy implementation on the Thomas-provided pilot."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import spectral_entropy
from spectral_entropy import tools_fast

try:
    from .pilot_data import Spectrum, prepare_spectrum, read_mgf, reference_candidates, spectrum_row
except ImportError:
    from pilot_data import Spectrum, prepare_spectrum, read_mgf, reference_candidates, spectrum_row

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("entropy_weighted", "entropy_unweighted")
NUMERICS = {"evaluated_scores": 0, "clipped_counts": {}, "raw_minimum": 1.0,
            "raw_maximum": 0.0, "examples": []}


def portable_module_path(filename: str) -> str:
    """Record modules from either a project-local or a named Conda environment."""
    path = Path(filename).resolve()
    for prefix, label in ((ROOT.resolve(), ""), (Path(sys.prefix).resolve(), "$CONDA_PREFIX/")):
        if path.is_relative_to(prefix):
            return label + path.relative_to(prefix).as_posix()
    return str(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(values) -> dict:
    values = np.asarray(list(values), dtype=float)
    if not len(values):
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    return {"count": len(values), "min": float(values.min()), "median": float(np.median(values)),
            "mean": float(values.mean()), "max": float(values.max())}


def normalize_score(value: float, method: str, tolerance: float = 1e-5, context: str = "") -> float:
    """Record and bound only floating-point excursions outside the physical range."""
    value = float(value)
    if not np.isfinite(value) or value < -tolerance or value > 1+tolerance:
        raise RuntimeError(f"Invalid {method} similarity score: {value} ({context})")
    NUMERICS["evaluated_scores"] += 1
    NUMERICS["raw_minimum"] = min(NUMERICS["raw_minimum"], value)
    NUMERICS["raw_maximum"] = max(NUMERICS["raw_maximum"], value)
    bounded = min(1.0, max(0.0, value))
    if bounded != value:
        counts = NUMERICS["clipped_counts"]
        counts[method] = counts.get(method, 0)+1
        if len(NUMERICS["examples"]) < 20:
            NUMERICS["examples"].append({"method": method, "raw_score": value,
                                         "bounded_score": bounded, "context": context})
    return bounded


def score_pair(a: np.ndarray, b: np.ndarray, tolerance: float,
               roundoff_tolerance: float = 1e-5, context: str = "") -> tuple[float, float]:
    """Score the same preprocessed peaks with the two published variants."""
    weighted = spectral_entropy.calculate_entropy_similarity(
        a, b, ms2_da=tolerance, need_clean_spectra=False)
    unweighted = spectral_entropy.similarity(
        a, b, method="unweighted_entropy", ms2_da=tolerance, need_clean_spectra=False)
    return (normalize_score(weighted, METHODS[0], roundoff_tolerance, context),
            normalize_score(unweighted, METHODS[1], roundoff_tolerance, context))


def rank_compounds(query_key: str, references: list[Spectrum], scores: list[float], tolerance: float) -> dict:
    """Collapse multiple reference spectra per compound and rank ties pessimistically."""
    best = {}
    for reference, score in zip(references, scores, strict=True):
        best[reference.compound_key] = max(best.get(reference.compound_key, -np.inf), float(score))
    if query_key not in best:
        raise ValueError("A retrieval query has no eligible true reference")
    true_score = best[query_key]
    rank = 1 + sum(value >= true_score - tolerance for key, value in best.items() if key != query_key)
    maximum = max(best.values())
    tied = sorted(key for key, value in best.items() if abs(value - maximum) <= tolerance)
    return {"rank": int(rank), "true_compound_score": float(true_score),
            "predicted_compound_key": tied[0], "top_tie_compounds": len(tied),
            "candidate_compounds": len(best)}


def synthetic_example(config: dict) -> dict:
    query = np.array([[69.071, 7.917962], [86.066, 1.021589], [86.0969, 100]], dtype=np.float32)
    reference = np.array([[41.04, 37.16], [69.07, 66.83], [86.1, 999]], dtype=np.float32)
    tolerance = config["ms2_tolerance_da"]
    cleaned_q = spectral_entropy.clean_spectrum(query.copy(), ms2_da=tolerance, noise_removal=0.01)
    cleaned_r = spectral_entropy.clean_spectrum(reference.copy(), ms2_da=tolerance, noise_removal=0.01)
    weighted, unweighted = score_pair(cleaned_q, cleaned_r, tolerance)
    return {"source": "YuanyueLi/SpectralEntropy/example.py", "kind": "illustrative_synthetic_peak_lists",
            "raw_query_peaks": query.tolist(), "raw_reference_peaks": reference.tolist(),
            "clean_query_peaks": cleaned_q.tolist(), "clean_reference_peaks": cleaned_r.tolist(),
            "ms2_tolerance_da": tolerance, "noise_removal_fraction": 0.01,
            "entropy_weighted": weighted, "entropy_unweighted": unweighted,
            "note": "The illustrative example has no precursor exclusion because no precursor metadata are supplied."}


def pair_samples(spectra: list[Spectrum], config: dict, rng: np.random.Generator) -> list[tuple[int, int, str]]:
    groups = defaultdict(list)
    pool = []
    for index, spectrum in enumerate(spectra):
        if spectrum.label_usable:
            groups[(spectrum.compound_key, spectrum.adduct, spectrum.ion_mode)].append(index)
            pool.append(index)
    same = sorted({(i, j) for group in groups.values() for i, j in itertools.combinations(group, 2)
                   if spectra[i].fingerprint != spectra[j].fingerprint})
    limit = config["maximum_same_compound_pairs"]
    if len(same) > limit:
        same = [same[i] for i in sorted(rng.choice(len(same), size=limit, replace=False))]
    different = set()
    limit = config["maximum_different_compound_pairs"]
    for _ in range(limit * 100):
        i, j = sorted(rng.choice(pool, size=2, replace=False).tolist())
        a, b = spectra[i], spectra[j]
        if a.compound_key != b.compound_key and a.adduct == b.adduct and a.ion_mode == b.ion_mode and a.fingerprint != b.fingerprint:
            different.add((i, j))
        if len(different) >= limit:
            break
    hard = set()
    for i in pool:
        for j in reference_candidates(spectra[i], spectra, config["precursor_tolerance_ppm"]):
            if spectra[i].compound_key != spectra[j].compound_key:
                hard.add(tuple(sorted((i, j))))
    hard = sorted(hard)
    if len(hard) > limit:
        hard = [hard[i] for i in sorted(rng.choice(len(hard), size=limit, replace=False))]
    return ([(i, j, "same_connectivity_same_adduct") for i, j in same]
            + [(i, j, "different_connectivity_random") for i, j in sorted(different)]
            + [(i, j, "different_connectivity_mass_matched") for i, j in hard])


def retrieval_metrics(rows: list[dict]) -> list[dict]:
    result = []
    keys = sorted({(row["candidate_mode"], row["method"]) for row in rows})
    for mode, method in keys:
        all_rows = [row for row in rows if row["candidate_mode"] == mode and row["method"] == method]
        for subset in ("all_queries", "multiple_candidate_compounds"):
            selected = all_rows if subset == "all_queries" else [row for row in all_rows if row["candidate_compounds"] > 1]
            n = len(selected)
            result.append({"candidate_mode": mode, "method": method, "subset": subset,
                           "queries": n, "top1_accuracy": sum(row["rank"] == 1 for row in selected)/n if n else None,
                           "recall_at_5": sum(row["rank"] <= 5 for row in selected)/n if n else None,
                           "mrr": sum(1/row["rank"] for row in selected)/n if n else None,
                           "median_candidate_spectra": float(np.median([row["candidate_spectra"] for row in selected])) if n else None,
                           "median_candidate_compounds": float(np.median([row["candidate_compounds"] for row in selected])) if n else None,
                           "total_score_seconds": sum(row["score_seconds"] for row in selected),
                           "mean_score_ms_per_query": 1000*sum(row["score_seconds"] for row in selected)/n if n else None})
    return result


def run(config_path: Path) -> dict:
    start = time.perf_counter()
    NUMERICS.update({"evaluated_scores":0, "clipped_counts":{}, "raw_minimum":1.0,
                     "raw_maximum":0.0, "examples":[]})
    config = json.loads(config_path.read_text())
    metrics = ROOT / config["metrics_directory"]
    figures = ROOT / config["figures_directory"]
    runs = ROOT / config["run_directory"]
    for folder in (metrics, figures, runs):
        folder.mkdir(parents=True, exist_ok=True)
    upstream = ROOT / "external/SpectralEntropy"
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=upstream, text=True).strip()
    if commit != config["upstream_commit"]:
        raise RuntimeError(f"Unexpected upstream revision: {commit}")
    input_rows, spectra = [], []
    preparation_start = time.perf_counter()
    for name in config["input_files"]:
        path = ROOT / name
        with path.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        if checksum != config["expected_sha256"][path.name]:
            raise RuntimeError(f"Input checksum mismatch: {path.name}")
        records = [prepare_spectrum(item, config) for item in read_mgf(path)]
        spectra.extend(records)
        input_rows.append({"source_file": path.name, "relative_path": name, "bytes": path.stat().st_size,
                           "sha256": checksum, "spectra": len(records), "received_from": "Dr. Thomas",
                           "upstream_url": "https://external.gnps2.org/gnpslibrary/" + path.name})
    if len({item.spectrum_id for item in spectra}) != len(spectra):
        raise RuntimeError("Input files contain duplicate spectrum IDs")
    preparation_seconds = time.perf_counter() - preparation_start
    write_csv(metrics / "input_manifest.csv", input_rows)
    write_csv(metrics / "spectra_audit.csv", [spectrum_row(item) for item in spectra])
    usable = [item for item in spectra if item.usable]
    print(json.dumps({"stage": "preprocessed", "input_spectra": len(spectra), "usable_spectra": len(usable),
                      "label_usable_spectra": sum(item.label_usable for item in spectra)}), flush=True)

    example = synthetic_example(config)
    (metrics / "synthetic_example.json").write_text(json.dumps(example, indent=2))
    tolerance = config["ms2_tolerance_da"]
    controls = []
    for item in usable:
        a, b = score_pair(item.peaks, item.peaks, tolerance, config["score_roundoff_tolerance"], item.spectrum_id)
        controls.append({"spectrum_id": item.spectrum_id, "entropy_weighted": a, "entropy_unweighted": b})
    if any(abs(row[method]-1) > 1e-6 for row in controls for method in METHODS):
        raise RuntimeError("Self-similarity control failed")
    write_csv(metrics / "self_similarity_controls.csv", controls)

    rng = np.random.default_rng(config["seed"])
    pairs = []
    pair_start = time.perf_counter()
    for i, j, kind in pair_samples(spectra, config, rng):
        a, b = spectra[i], spectra[j]
        weighted, unweighted = score_pair(a.peaks, b.peaks, tolerance, config["score_roundoff_tolerance"],
                                          f"{a.spectrum_id} versus {b.spectrum_id}")
        pairs.append({"query_id": a.spectrum_id, "reference_id": b.spectrum_id,
                      "pair_type": kind, "query_source": a.source_file, "reference_source": b.source_file,
                      "query_name": a.metadata.get("NAME", ""), "reference_name": b.metadata.get("NAME", ""),
                      "query_compound_key": a.compound_key, "reference_compound_key": b.compound_key,
                      "adduct": a.adduct, "precursor_error_ppm": abs(a.precursor_mz-b.precursor_mz)/a.precursor_mz*1e6,
                      "entropy_weighted": weighted, "entropy_unweighted": unweighted})
    pair_seconds = time.perf_counter() - pair_start
    write_csv(metrics / "pair_scores.csv", pairs)

    eligible_by_compound = defaultdict(list)
    for index, item in enumerate(spectra):
        if item.label_usable:
            references = reference_candidates(item, spectra, config["precursor_tolerance_ppm"])
            if any(spectra[j].compound_key == item.compound_key for j in references):
                eligible_by_compound[item.compound_key].append(index)
    compounds = sorted(eligible_by_compound)
    if not compounds:
        raise RuntimeError("No queries have a distinct same-compound reference within precursor tolerance")
    query_rng = np.random.default_rng(config["seed"])
    n = min(config["maximum_query_spectra"], len(compounds))
    selected_compounds = sorted(query_rng.choice(compounds, size=n, replace=False).tolist())
    queries = [int(query_rng.choice(eligible_by_compound[key])) for key in selected_compounds]
    write_csv(metrics / "query_selection.csv", [{"spectrum_id": spectra[i].spectrum_id,
               "compound_key": spectra[i].compound_key, "source_file": spectra[i].source_file,
               "adduct": spectra[i].adduct} for i in queries])

    retrieval_rows = []
    score_matrix = np.full((len(queries), len(spectra), 2), np.nan, dtype=np.float64)
    modes = (("same_adduct_all_precursors", None), ("same_adduct_precursor_10ppm", config["precursor_tolerance_ppm"]))
    for mode, ppm in modes:
        mode_start = time.perf_counter()
        for qi, index in enumerate(queries):
            query = spectra[index]
            candidate_start = time.perf_counter()
            indices = reference_candidates(query, spectra, ppm)
            selection_seconds = time.perf_counter()-candidate_start
            references = [spectra[j] for j in indices]
            scores_by_method = [[], []]
            times = [0.0, 0.0]
            for j in indices:
                reference = spectra[j]
                t = time.perf_counter()
                weighted = float(spectral_entropy.calculate_entropy_similarity(
                    query.peaks, reference.peaks, ms2_da=tolerance, need_clean_spectra=False))
                times[0] += time.perf_counter()-t
                t = time.perf_counter()
                unweighted = float(spectral_entropy.similarity(
                    query.peaks, reference.peaks, method="unweighted_entropy", ms2_da=tolerance,
                    need_clean_spectra=False))
                times[1] += time.perf_counter()-t
                context = f"{query.spectrum_id} versus {reference.spectrum_id}"
                weighted = normalize_score(weighted, METHODS[0], config["score_roundoff_tolerance"], context)
                unweighted = normalize_score(unweighted, METHODS[1], config["score_roundoff_tolerance"], context)
                scores_by_method[0].append(weighted)
                scores_by_method[1].append(unweighted)
                if ppm is None:
                    score_matrix[qi, j] = (weighted, unweighted)
            for method_index, method in enumerate(METHODS):
                ranking = rank_compounds(query.compound_key, references, scores_by_method[method_index], config["score_tie_tolerance"])
                retrieval_rows.append({"query_id":query.spectrum_id, "query_compound_key":query.compound_key,
                                       "query_source":query.source_file, "candidate_mode":mode, "method":method,
                                       "candidate_spectra":len(indices), **ranking,
                                       "candidate_selection_seconds":selection_seconds, "score_seconds":times[method_index]})
            if (qi+1) % 10 == 0 or qi+1 == len(queries):
                print(json.dumps({"stage":"retrieval", "mode":mode, "completed_queries":qi+1,
                                  "queries":len(queries), "elapsed_seconds":round(time.perf_counter()-mode_start,2)}),flush=True)
    write_csv(metrics / "retrieval_queries.csv", retrieval_rows)
    retrieval_summary = retrieval_metrics(retrieval_rows)
    write_csv(metrics / "retrieval_summary.csv", retrieval_summary)
    np.savez_compressed(runs / "retrieval_scores.npz", scores=score_matrix,
                        query_ids=np.asarray([spectra[i].spectrum_id for i in queries]),
                        reference_ids=np.asarray([s.spectrum_id for s in spectra]), methods=np.asarray(METHODS))
    offsets = np.cumsum([0] + [len(s.peaks) for s in spectra])
    np.savez_compressed(runs / "cleaned_spectra.npz", peaks=np.concatenate([s.peaks for s in spectra]),
                        offsets=offsets, spectrum_ids=np.asarray([s.spectrum_id for s in spectra]))

    per_file = []
    for name in config["input_files"]:
        subset = [s for s in spectra if s.source_file == Path(name).name]
        per_file.append({"source_file":Path(name).name, "spectra":len(subset),
                         "usable_spectra":sum(s.usable for s in subset),
                         "label_usable_spectra":sum(s.label_usable for s in subset),
                         "original_inchikey_missing":sum(not s.metadata.get("INCHIKEY") for s in subset),
                         "derived_inchikey_available":sum(bool(s.derived_inchikey) for s in subset),
                         "conflicting_structure_annotations":sum(s.structure_status == "conflicting_annotations" for s in subset),
                         "missing_collision_energy":sum(not s.metadata.get("COLLISION_ENERGY") for s in subset),
                         "ion_modes":dict(Counter(s.ion_mode for s in subset)),
                         "adducts_inferred_from_name":dict(Counter(s.adduct for s in subset)),
                         "raw_peak_count":summarize(len(s.raw_peaks) for s in subset),
                         "clean_peak_count":summarize(len(s.peaks) for s in subset if s.usable),
                         "spectral_entropy_nats":summarize(s.spectral_entropy for s in subset if s.usable)})

    example_rows = []
    for kind in sorted({row["pair_type"] for row in pairs}):
        subset = sorted([row for row in pairs if row["pair_type"] == kind], key=lambda r:r["entropy_weighted"])
        chosen = subset[-1] if kind == "different_connectivity_mass_matched" else subset[len(subset)//2]
        example_rows.append(chosen)
    (metrics / "example_pairs.json").write_text(json.dumps(example_rows,indent=2))
    try:
        from .pilot_figures import plot_pilot
    except ImportError:
        from pilot_figures import plot_pilot
    plot_pilot(spectra, pairs, example_rows, figures)

    packages = {}
    for package in ("SpectralEntropy", "numpy", "scipy", "Cython", "rdkit", "matplotlib", "pytest"):
        packages[package] = importlib.metadata.version(package)
    environment = {"python":sys.version, "platform":platform.platform(), "machine":platform.machine(),
                   "logical_cpus":os.cpu_count(), "packages":packages,
                   "spectral_entropy_module":portable_module_path(spectral_entropy.__file__),
                   "cython_extension":portable_module_path(tools_fast.__file__)}
    summary = {"run_timestamp_utc":datetime.now(timezone.utc).isoformat(), "config":config,
               "config_sha256":hashlib.sha256(config_path.read_bytes()).hexdigest(), "upstream_commit":commit,
               "command":"python src/run_entropy_pilot.py --config configs/thomas_entropy_pilot.json",
               "environment":environment, "input_files":input_rows,
               "total_input_spectra":len(spectra), "usable_spectra":len(usable),
               "label_usable_spectra":sum(s.label_usable for s in spectra),
               "exclusion_reasons":dict(Counter(s.exclusion_reason for s in spectra if s.exclusion_reason)),
               "structure_status":dict(Counter(s.structure_status for s in spectra)),
               "unique_label_usable_compounds":len({s.compound_key for s in spectra if s.label_usable}),
               "duplicate_clean_peak_vectors":sum(n-1 for n in Counter(s.fingerprint for s in usable).values()),
               "eligible_query_spectra":sum(map(len,eligible_by_compound.values())),
               "eligible_query_compounds":len(compounds), "evaluated_query_spectra":len(queries),
               "query_source_counts":dict(Counter(spectra[i].source_file for i in queries)),
               "query_protocol":"One seeded query per sampled connectivity key; leave-one-spectrum-out; identical cleaned peak vectors excluded; no held-out test set.",
               "label_protocol":"First 14 characters of RDKit InChIKey derived from supplied SMILES; conflicting supplied InChI labels excluded from labeled analysis.",
               "tie_policy":f"Pessimistic compound rank with score tolerance {config['score_tie_tolerance']}; each compound scored by its best reference spectrum.",
               "numerical_diagnostics": NUMERICS,
               "self_similarity_max_abs_error":{method:max(abs(row[method]-1) for row in controls) for method in METHODS},
               "synthetic_example":example, "per_file":per_file,
               "pair_statistics":{kind:{"pairs":sum(row["pair_type"]==kind for row in pairs),
                                         **{method:summarize(row[method] for row in pairs if row["pair_type"]==kind) for method in METHODS}}
                                  for kind in sorted({row["pair_type"] for row in pairs})},
               "retrieval":retrieval_summary,
               "timing_seconds":{"preparation":preparation_seconds, "pair_examples":pair_seconds,
                                 "pipeline_wall":time.perf_counter()-start},
               "scope":"Entropy-only pilot on public GNPS exports supplied by Dr. Thomas; not a three-algorithm comparison or a sediment/lipidomics validation."}
    (metrics / "pilot_summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False))
    (runs / "config_used.json").write_text(json.dumps(config,indent=2))
    print(json.dumps({"stage":"complete", "summary":str((metrics/'pilot_summary.json').relative_to(ROOT)),
                      "wall_seconds":round(summary['timing_seconds']['pipeline_wall'],2)}),flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/thomas_entropy_pilot.json")
    args = parser.parse_args()
    run(args.config.resolve())
