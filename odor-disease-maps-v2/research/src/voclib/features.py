"""Permutation-invariant representations with full stereochemical identities."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator

PROPERTIES = ("MW", "logP", "TPSA", "HBD", "HBA", "rotatable_bonds", "rings", "fraction_CSP3")
CHANNELS = ("mean_fingerprint_chord", "union_fingerprint_jaccard", "component_set_jaccard", "component_count_difference") + tuple("mean_" + p for p in PROPERTIES) + tuple("std_" + p for p in PROPERTIES)
FEATURE_VERSION = "morgan-r2-2048-chiral-weighted-v1"
_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)


@lru_cache(maxsize=20000)
def molecule(smiles):
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("SMILES must be a nonempty string")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or not mol.GetNumAtoms():
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
    fp = _GENERATOR.GetFingerprintAsNumPy(mol).astype(float)
    props = np.array([Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol), Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol), Descriptors.NumRotatableBonds(mol), Descriptors.RingCount(mol), Descriptors.FractionCSP3(mol)])
    fp.setflags(write=False)
    props.setflags(write=False)
    return canonical, fp, props


@dataclass(frozen=True)
class Mixture:
    identities: tuple
    weights: tuple
    weight_basis: str
    mean_fp: np.ndarray
    union_fp: np.ndarray
    mean_props: np.ndarray
    std_props: np.ndarray

    @property
    def composition_signature(self):
        """Conservative leakage key: also groups different ratios of the same components."""
        return self.identities


def mixture(smiles, weights=None, weight_basis="equal"):
    if not smiles:
        raise ValueError("A mixture needs at least one component")
    if weight_basis not in {"equal", "mole_fraction"}:
        raise ValueError("weight_basis must be equal or mole_fraction; liquid dilution is not gas composition")
    if (weights is None) != (weight_basis == "equal"):
        raise ValueError("Provide all weights only for mole_fraction")
    values = [molecule(s) for s in smiles]
    w = np.ones(len(values)) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (len(values),) or not np.isfinite(w).all() or (w <= 0).any():
        raise ValueError("Component weights must be finite, positive and match components")
    # Duplicate aliases may combine supplied amounts, but must not silently double an equal-weight component.
    merged = {}
    for value, weight in zip(values, w):
        identity = value[0]
        if identity in merged:
            if weight_basis == "equal":
                raise ValueError("Duplicate canonical component in equal-weight mixture")
            merged[identity][0] += float(weight)
        else:
            merged[identity] = [float(weight), value]
    ordered = sorted(merged)
    w = np.array([merged[key][0] for key in ordered]); w /= w.sum()
    fps = np.stack([merged[key][1][1] for key in ordered])
    props = np.stack([merged[key][1][2] for key in ordered])
    mean = w @ props
    arrays = [w @ fps, fps.max(axis=0), mean, np.sqrt(w @ ((props - mean) ** 2))]
    for array in arrays:
        array.setflags(write=False)
    return Mixture(tuple(ordered), tuple(w.tolist()), weight_basis, *arrays)


def pair_channels(a, b):
    if a.weight_basis != b.weight_basis:
        raise ValueError("Cannot compare equal-weight proxies with measured mole fractions in one channel set")
    u = a.mean_fp / np.linalg.norm(a.mean_fp)
    v = b.mean_fp / np.linalg.norm(b.mean_fp)
    union = np.maximum(a.union_fp, b.union_fp).sum()
    component_union = set(a.identities) | set(b.identities)
    return np.r_[np.linalg.norm(u - v), 1 - np.minimum(a.union_fp, b.union_fp).sum() / union,
                 1 - len(set(a.identities) & set(b.identities)) / len(component_union),
                 abs(len(a.identities) - len(b.identities)),
                 np.abs(a.mean_props - b.mean_props), np.abs(a.std_props - b.std_props)]
