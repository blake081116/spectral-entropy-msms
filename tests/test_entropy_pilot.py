"""Check data integrity, duplicate exclusion, and ranking semantics of the pilot."""

import numpy as np
import pytest

import src.run_entropy_pilot as pilot
from src.pilot_data import Spectrum, derive_structure, extract_adduct, peak_fingerprint, prepare_spectrum, read_mgf, reference_candidates
from src.run_entropy_pilot import normalize_score, rank_compounds, score_pair


def make_spectrum(identifier, key, fingerprint, precursor=200.0, adduct="[M+H]"):
    spectrum = Spectrum(identifier, "fixture.mgf", {}, np.array([[50, 1], [100, 1]], dtype=np.float32))
    spectrum.peaks = spectrum.raw_peaks/np.array([1, 2], dtype=np.float32)
    spectrum.precursor_mz = precursor
    spectrum.ion_mode = "positive"
    spectrum.adduct = adduct
    spectrum.compound_key = key
    spectrum.structure_status = "valid"
    spectrum.fingerprint = fingerprint
    return spectrum


def test_reader_rejects_truncated_record(tmp_path):
    path = tmp_path/"broken.mgf"
    path.write_text("BEGIN IONS\nPEPMASS=200\n50 100\n")
    with pytest.raises(ValueError, match="Unclosed MGF record"):
        list(read_mgf(path))


def test_reader_preserves_metadata_and_peaks(tmp_path):
    path = tmp_path/"one.mgf"
    path.write_text("BEGIN IONS\nSPECTRUMID=example\nSMILES=C=O\nPEPMASS=200 50\n100 3\n150 7\nEND IONS\n")
    spectrum, = read_mgf(path)
    assert spectrum.metadata["SMILES"] == "C=O"
    assert spectrum.spectrum_id == "example"
    np.testing.assert_array_equal(spectrum.raw_peaks, [[100, 3], [150, 7]])


def test_adduct_extraction_supports_both_source_name_styles():
    assert extract_adduct({"NAME":"Pesticide long name M+H"}) == "[M+H]"
    assert extract_adduct({"NAME":"Compound [M+2H]", "CHARGE":"1"}) == "[M+2H]"
    assert extract_adduct({"NAME":"Compound M-H2O+H"}) == "[M-H2O+H]"
    assert extract_adduct({"NAME":"Unknown compound"}) == ""


def test_cleaning_excludes_precursor_and_noise_without_mutating_raw():
    peaks = np.array([[100, 100], [100.01, 50], [150, 0.1], [250, 50], [299, 100], [300, 100]], dtype=np.float32)
    spectrum = Spectrum("fixture", "fixture.mgf", {"PEPMASS":"300", "SMILES":"CCO", "NAME":"Ethanol M+H", "IONMODE":"Positive"}, peaks.copy())
    config = {"precursor_exclusion_da":1.6, "noise_removal_fraction":0.01, "ms2_tolerance_da":0.05, "minimum_clean_peaks":2}
    result = prepare_spectrum(spectrum, config)
    assert result.usable and len(result.peaks) == 2
    assert result.peaks[:,0].max() <= 298.4
    assert result.peaks[:,1].sum() == pytest.approx(1)
    np.testing.assert_array_equal(result.raw_peaks, peaks)


def test_equivalent_smiles_produce_the_same_connectivity_label():
    a = derive_structure({"SMILES":"CCO"})
    b = derive_structure({"SMILES":"OCC"})
    assert a["structure_status"] == b["structure_status"] == "valid"
    assert a["compound_key"] == b["compound_key"]


def test_candidates_remove_self_and_duplicate_peak_vectors():
    query = make_spectrum("q", "A", "same")
    duplicate = make_spectrum("duplicate", "A", "same")
    true = make_spectrum("true", "A", "different", precursor=200.001)
    wrong_adduct = make_spectrum("adduct", "A", "third", adduct="[M+Na]")
    far = make_spectrum("far", "B", "fourth", precursor=300)
    library = [query, duplicate, true, wrong_adduct, far]
    assert reference_candidates(query, library, 10) == [2]
    assert reference_candidates(query, library, None) == [2, 4]


def test_compound_ranking_collapses_replicates_and_penalizes_ties():
    refs = [make_spectrum("a1", "A", "1"), make_spectrum("a2", "A", "2"),
            make_spectrum("b", "B", "3"), make_spectrum("c", "C", "4")]
    ranked = rank_compounds("A", refs, [0.5, 0.8, 0.8, 0.8], 1e-8)
    assert ranked["candidate_compounds"] == 3
    assert ranked["rank"] == 3
    assert ranked["top_tie_compounds"] == 3


def test_entropy_controls_and_intensity_scale_invariance():
    a = np.array([[50, 0.25], [100, 0.75]], dtype=np.float32)
    b = np.array([[150, 0.5], [200, 0.5]], dtype=np.float32)
    assert score_pair(a, a, 0.05) == pytest.approx((1, 1), abs=1e-6)
    assert score_pair(a, b, 0.05) == pytest.approx((0, 0), abs=1e-6)
    assert peak_fingerprint(a) == peak_fingerprint(a*np.array([1, 100], dtype=np.float32))


def test_float32_roundoff_is_bounded_but_invalid_scores_are_rejected():
    assert normalize_score(-1.3e-6, "fixture") == 0
    assert normalize_score(1+1.3e-6, "fixture") == 1
    with pytest.raises(RuntimeError, match="Invalid"):
        normalize_score(-0.01, "fixture")
    with pytest.raises(RuntimeError, match="Invalid"):
        normalize_score(float("nan"), "fixture")


def test_module_paths_support_named_and_project_local_environments(tmp_path, monkeypatch):
    project = tmp_path / "project"
    environment = tmp_path / "envs/CSCI558"
    monkeypatch.setattr(pilot, "ROOT", project)
    monkeypatch.setattr(pilot.sys, "prefix", str(environment))
    assert pilot.portable_module_path(str(environment / "lib/site-packages/spectral_entropy/__init__.py")) == "$CONDA_PREFIX/lib/site-packages/spectral_entropy/__init__.py"
    assert pilot.portable_module_path(str(project / ".venv/lib/spectral_entropy/__init__.py")) == ".venv/lib/spectral_entropy/__init__.py"
