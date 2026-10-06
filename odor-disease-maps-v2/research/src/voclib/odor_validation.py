"""Group-safe rating models and diagnostics. Similarity is not confidence."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold


def ridge_model(alpha):
    return make_pipeline(StandardScaler(), Ridge(alpha=float(alpha), solver='lsqr'))


def select_ridge(inputs, y, groups, alphas=(1., 10., 100., 1000., 10000.)):
    """Choose representation/regularization exclusively inside supplied groups."""
    groups = np.asarray(groups)
    folds = list(GroupKFold(n_splits=3).split(y, groups=groups))
    scores = []
    for name, x in inputs.items():
        for alpha in alphas:
            pred = np.empty_like(y, dtype=float)
            for train, val in folds:
                assert not set(groups[train]) & set(groups[val])
                model = ridge_model(alpha).fit(x[train], y[train])
                pred[val] = np.clip(model.predict(x[val]), 0, 1)
            scores.append(dict(representation=name, alpha=alpha,
                               mae=float(np.abs(pred-y).mean())))
    return min(scores, key=lambda r: r['mae']), scores


def tanimoto_matrix(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    intersection = a @ b.T
    union = a.sum(1)[:, None] + b.sum(1)[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def delta_diagnostics(rows, y, pred, fingerprints, folds, min_similarity=.5):
    """Pairs from the SAME held-out fold and dilution; no pair independence claim.

    Fingerprint similarity does not establish a matched molecular pair.
    """
    sim = tanimoto_matrix(fingerprints, fingerprints)
    records = []
    for a in range(len(rows)):
        for b in range(a+1, len(rows)):
            if (folds[a] != folds[b] or rows[a]['key'] == rows[b]['key']
                    or rows[a]['concentration'] != rows[b]['concentration']
                    or sim[a, b] < min_similarity):
                continue
            records.append(dict(a=a, b=b, similarity=float(sim[a, b]),
                                observed_delta=(y[b]-y[a]).tolist(),
                                predicted_delta=(pred[b]-pred[a]).tolist()))
    if not records:
        return {'n_pairs': 0, 'delta_mae': None}, records
    true = np.array([r['observed_delta'] for r in records])
    estimate = np.array([r['predicted_delta'] for r in records])
    return {'n_pairs': len(records), 'delta_mae': np.abs(true-estimate).mean(0).tolist(),
            'zero_change_mae': np.abs(true).mean(0).tolist(),
            'scope': 'Similar structure pairs, not verified substitutions; shared molecules make pairs dependent.'}, records


class RatingRidgePredictor:
    """Load only trusted, locally trained joblib bundles.

    POM/hybrid predictions require embeddings from the exact encoder stored in
    metadata. No mixture, gas-phase or individual-human claim is supported.
    """
    def __init__(self, bundle):
        self.bundle = bundle

    def predict_features(self, inputs, encoder_metadata=None):
        if self.bundle['representation'] in ('pom', 'hybrid'):
            if encoder_metadata is None or encoder_metadata != self.bundle.get('encoder'):
                raise ValueError('POM encoder metadata must match the trained checkpoint/configuration.')
        x = np.asarray(inputs[self.bundle['representation']])
        if x.ndim != 2 or not np.isfinite(x).all():
            raise ValueError('Expected a finite two-dimensional feature matrix.')
        return np.clip(self.bundle['model'].predict(x), 0, 1)

    def predict_smiles(self, smiles, concentration=.001, encoder=None,
                       cached_embeddings=None, encoder_metadata=None):
        from .odor_edit import molecule, features, rating_features
        if not smiles:
            raise ValueError('Provide at least one molecule')
        molecules = [molecule(s) for s in smiles]
        conditions = np.array([rating_features(m, concentration)[-4:] for m in molecules])
        representation = self.bundle['representation']
        blocks = []
        if representation in ('morgan', 'hybrid'):
            blocks.append(np.array([features(m) for m in molecules]))
        if representation in ('pom', 'hybrid'):
            if cached_embeddings is None:
                if encoder is None:
                    raise ValueError('Provide the trained POM encoder or matching cached embeddings')
                if encoder.config != self.bundle['encoder']:
                    raise ValueError('Incompatible POM encoder')
                cached_embeddings, _ = encoder.encode(smiles)
                encoder_metadata = encoder.config
            z = np.asarray(cached_embeddings)
            if z.shape != (len(smiles), 256):
                raise ValueError('Expected one 256-dimensional POM embedding per SMILES')
            blocks.append(z)
        blocks.append(conditions)
        return self.predict_features({representation: np.column_stack(blocks)}, encoder_metadata)
