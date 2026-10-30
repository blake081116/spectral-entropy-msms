"""Read, audit, and preprocess the two GNPS files supplied by Dr. Thomas."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import re
from pathlib import Path
from typing import Iterator

import numpy as np
from rdkit import Chem, rdBase
from scipy.stats import entropy
import spectral_entropy


@dataclass
class Spectrum:
    spectrum_id: str
    source_file: str
    metadata: dict[str, str]
    raw_peaks: np.ndarray
    peaks: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.float32))
    precursor_mz: float | None = None
    adduct: str = ""
    ion_mode: str = ""
    compound_key: str = ""
    canonical_smiles: str = ""
    derived_inchikey: str = ""
    structure_status: str = "unprocessed"
    inchi_smiles_agree: bool | None = None
    fingerprint: str = ""
    spectral_entropy: float | None = None
    invalid_peak_count: int = 0
    exclusion_reason: str = ""

    @property
    def usable(self) -> bool:
        return not self.exclusion_reason

    @property
    def label_usable(self) -> bool:
        return self.usable and self.structure_status == "valid" and bool(self.adduct)


def read_mgf(path: Path) -> Iterator[Spectrum]:
    """Read MGF records strictly, preserving source metadata and peak values."""
    metadata = None
    peaks: list[tuple[float, float]] = []
    record_number = 0
    with path.open(encoding="utf-8", errors="strict") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            line = raw_line.strip()
            if not line or line.startswith(("#", ";", "!")):
                continue
            if line.upper() == "BEGIN IONS":
                if metadata is not None:
                    raise ValueError(f"Nested MGF record in {path.name}:{line_number}")
                metadata, peaks = {}, []
                record_number += 1
            elif line.upper() == "END IONS":
                if metadata is None:
                    raise ValueError(f"Unexpected END IONS in {path.name}:{line_number}")
                identifier = metadata.get("SPECTRUMID") or f"{path.stem}:{record_number}"
                yield Spectrum(identifier, path.name, metadata,
                               np.asarray(peaks, dtype=np.float32).reshape(-1, 2))
                metadata = None
            elif metadata is not None:
                if "=" in line:
                    key, value = line.split("=", 1)
                    metadata[key.upper().strip()] = value.strip()
                else:
                    fields = line.split()
                    try:
                        peaks.append((float(fields[0]), float(fields[1])))
                    except (ValueError, IndexError) as exc:
                        raise ValueError(f"Invalid peak in {path.name}:{line_number}") from exc
        if metadata is not None:
            raise ValueError(f"Unclosed MGF record at end of {path.name}")


def extract_adduct(metadata: dict[str, str]) -> str:
    """Extract the adduct expression without trusting inconsistent charge labels."""
    explicit = metadata.get("ADDUCT") or metadata.get("PRECURSOR_TYPE")
    text = explicit or metadata.get("NAME", "")
    bracketed = re.search(r"\[([^\[\]]+)\](?:\d*[+-])?\s*$", text)
    if bracketed:
        return "[" + bracketed.group(1).replace(" ", "") + "]"
    bare = re.search(r"(?:^|\s)(\d*M(?:[+-][A-Za-z0-9]+)+)\s*$", text)
    return "[" + bare.group(1) + "]" if bare else ""


def derive_structure(metadata: dict[str, str]) -> dict:
    """Derive achiral connectivity labels from supplied SMILES, not from peaks."""
    smiles = metadata.get("SMILES", "").strip().strip('"')
    if smiles.upper() in ("", "N/A", "NA", "NONE"):
        return {"structure_status": "missing_smiles"}
    try:
        with rdBase.BlockLogs():
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                return {"structure_status": "invalid_smiles"}
            inchikey = Chem.MolToInchiKey(molecule)
            canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
            if not re.fullmatch(r"[A-Z]{14}-[A-Z]{10}-[A-Z]", inchikey):
                return {"structure_status": "inchikey_generation_failed"}
            source_inchi = metadata.get("INCHI", "").strip().strip('"')
            agreement = None
            if source_inchi.startswith("InChI="):
                source_molecule = Chem.MolFromInchi(source_inchi)
                if source_molecule is not None:
                    source_key = Chem.MolToInchiKey(source_molecule)
                    agreement = bool(source_key) and source_key[:14] == inchikey[:14]
        return {"compound_key": inchikey[:14], "derived_inchikey": inchikey,
                "canonical_smiles": canonical, "inchi_smiles_agree": agreement,
                "structure_status": "conflicting_annotations" if agreement is False else "valid"}
    except Exception as exc:
        return {"structure_status": f"structure_error:{type(exc).__name__}"}


def peak_fingerprint(peaks: np.ndarray) -> str:
    """Hash sorted, normalized peaks at 6 decimal places for duplicate exclusion."""
    values = np.asarray(peaks, dtype=np.float64).copy()
    values = values[np.argsort(values[:, 0], kind="stable")]
    total = values[:, 1].sum()
    if total > 0:
        values[:, 1] /= total
    values = np.round(values, decimals=6).astype("<f8")
    return hashlib.sha256(values.tobytes()).hexdigest()


def prepare_spectrum(spectrum: Spectrum, config: dict) -> Spectrum:
    """Apply the published cleaner once and retain all audit fields."""
    metadata = spectrum.metadata
    spectrum.ion_mode = metadata.get("IONMODE", "").strip().lower()
    spectrum.adduct = extract_adduct(metadata)
    for name, value in derive_structure(metadata).items():
        setattr(spectrum, name, value)
    try:
        spectrum.precursor_mz = float(metadata.get("PEPMASS", "").split()[0])
    except (ValueError, IndexError):
        spectrum.exclusion_reason = "missing_or_invalid_precursor_mz"
        return spectrum
    if not np.isfinite(spectrum.precursor_mz) or spectrum.precursor_mz <= config["precursor_exclusion_da"]:
        spectrum.exclusion_reason = "nonphysical_precursor_mz"
        return spectrum
    if metadata.get("MSLEVEL", "2") != "2":
        spectrum.exclusion_reason = "not_ms2"
        return spectrum
    peaks = spectrum.raw_peaks
    valid = np.isfinite(peaks).all(axis=1) & (peaks[:, 0] > 0) & (peaks[:, 1] > 0)
    spectrum.invalid_peak_count = int((~valid).sum())
    if not valid.any():
        spectrum.exclusion_reason = "no_valid_raw_peaks"
        return spectrum
    cleaned = spectral_entropy.clean_spectrum(
        peaks[valid].copy(),
        max_mz=spectrum.precursor_mz - config["precursor_exclusion_da"],
        noise_removal=config["noise_removal_fraction"],
        ms2_da=config["ms2_tolerance_da"],
    )
    spectrum.peaks = np.asarray(cleaned, dtype=np.float32).reshape(-1, 2)
    if len(cleaned) < config["minimum_clean_peaks"]:
        spectrum.exclusion_reason = "fewer_than_minimum_clean_peaks"
        return spectrum
    if not np.isfinite(cleaned).all() or np.any(cleaned[:, 1] <= 0):
        spectrum.exclusion_reason = "invalid_cleaned_peaks"
        return spectrum
    if not np.isclose(cleaned[:, 1].sum(), 1, atol=1e-6):
        raise RuntimeError(f"Cleaner did not L1-normalize {spectrum.spectrum_id}")
    spectrum.fingerprint = peak_fingerprint(cleaned)
    spectrum.spectral_entropy = float(entropy(cleaned[:, 1]))
    return spectrum


def reference_candidates(query: Spectrum, spectra: list[Spectrum], ppm: float | None) -> list[int]:
    """Exclude the query and all identical cleaned peak vectors before ranking."""
    candidates = []
    for index, reference in enumerate(spectra):
        if not reference.label_usable:
            continue
        if reference.spectrum_id == query.spectrum_id or reference.fingerprint == query.fingerprint:
            continue
        if reference.adduct != query.adduct or reference.ion_mode != query.ion_mode:
            continue
        if ppm is not None:
            error = abs(reference.precursor_mz - query.precursor_mz) / query.precursor_mz * 1e6
            if error > ppm:
                continue
        candidates.append(index)
    return candidates


def spectrum_row(spectrum: Spectrum) -> dict:
    return {
        "spectrum_id": spectrum.spectrum_id, "source_file": spectrum.source_file,
        "name": spectrum.metadata.get("NAME", ""), "precursor_mz": spectrum.precursor_mz,
        "ion_mode": spectrum.ion_mode, "adduct_from_name": spectrum.adduct,
        "reported_charge": spectrum.metadata.get("CHARGE", ""),
        "instrument": spectrum.metadata.get("SOURCE_INSTRUMENT", ""),
        "reported_collision_energy": spectrum.metadata.get("COLLISION_ENERGY", ""),
        "original_inchikey": spectrum.metadata.get("INCHIKEY", ""),
        "derived_inchikey": spectrum.derived_inchikey, "compound_key": spectrum.compound_key,
        "canonical_smiles": spectrum.canonical_smiles, "structure_status": spectrum.structure_status,
        "inchi_smiles_connectivity_agree": spectrum.inchi_smiles_agree,
        "raw_peaks": len(spectrum.raw_peaks), "clean_peaks": len(spectrum.peaks),
        "invalid_raw_peaks": spectrum.invalid_peak_count,
        "spectral_entropy_nats": spectrum.spectral_entropy, "peak_fingerprint": spectrum.fingerprint,
        "exclusion_reason": spectrum.exclusion_reason,
    }
