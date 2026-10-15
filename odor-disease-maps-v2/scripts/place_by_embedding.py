"""Place molecules that are not on the fixed map, using the frozen OpenPOM checkpoint.

Run in the pinned OpenPOM environment (research/docs/pom-*-lock.txt) with DGLBACKEND=pytorch.
Existing coordinates are never changed. Each new molecule is embedded (256-D), then placed at
the cosine-similarity-weighted mean of its k most similar map molecules. The same rule applied
leave-one-out to the map molecules reports the expected placement error.

Example:
  python scripts/place_by_embedding.py --cache private_data/inference_cache.npz \
    --overlay private_data/evidence_overlay.json --molecules data/crc_additions_curated.csv \
    --checkpoint private_data/checkpoint2.pt --out results/crc_placement
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'research' / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from crc_tools import knn_place, loo_errors, unit_rows  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--cache', type=Path, required=True, help='inference_cache.npz with smiles, embeddings, descriptor_scores')
    p.add_argument('--overlay', type=Path, required=True, help='evidence_overlay.json (fixed x, y per point)')
    p.add_argument('--molecules', type=Path, required=True, help='CSV with name and smiles columns')
    p.add_argument('--checkpoint', type=Path, required=True, help='OpenPOM checkpoint2.pt')
    p.add_argument('--config', type=Path, default=ROOT / 'research/examples/openpom_checkpoint_config.json')
    p.add_argument('--k', type=int, default=5)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()

    from rdkit import Chem, RDLogger
    from voclib.pom_backend import POMEncoder
    RDLogger.DisableLog('rdApp.*')
    key = lambda s: Chem.MolToInchiKey(Chem.MolFromSmiles(s))

    labels = json.loads(a.config.read_text(encoding='utf-8'))['labels']
    cache = np.load(a.cache, allow_pickle=False)
    doc = json.loads(a.overlay.read_text(encoding='utf-8-sig'))
    pts = doc['points']
    if [q['smiles'] for q in pts] != cache['smiles'].tolist():
        raise ValueError('Overlay points and cache rows must be in the same order')
    xy = np.array([[q['x'], q['y']] for q in pts], float)
    ref = unit_rows(cache['embeddings'])
    on_map = {q['inchikey']: q for q in pts}

    a.out.mkdir(parents=True, exist_ok=True)
    enc = POMEncoder(a.checkpoint, a.config, a.out / 'model-dir')
    z_chk, s_chk = enc.encode(cache['smiles'][:8].tolist())
    parity = {'max_abs_embedding_diff': float(np.abs(z_chk - cache['embeddings'][:8]).max()),
              'max_abs_score_diff': float(np.abs(s_chk - cache['descriptor_scores'][:8]).max())}
    if parity['max_abs_embedding_diff'] > 1e-3:
        raise ValueError(f'Checkpoint does not reproduce the cached embeddings: {parity}')

    with a.molecules.open(encoding='utf-8-sig') as f:
        mols = list(csv.DictReader(f))
    new = [m for m in mols if key(m['smiles']) not in on_map]
    z, s = enc.encode([m['smiles'] for m in new]) if new else (np.zeros((0, 256)), np.zeros((0, len(labels))))

    rows = []
    for i, m in enumerate(new):
        pos, nn, sim = knn_place(z[i], ref, xy, a.k)
        top = np.argsort(-s[i])[:6]
        rows.append({'name': m['name'], 'smiles': m['smiles'], 'inchikey': key(m['smiles']),
                     'x': round(float(pos[0]), 6), 'y': round(float(pos[1]), 6), 'max_cos': round(float(sim[0]), 4),
                     'neighbors': '; '.join(f"{pts[j].get('name', j)} ({c:.2f})" for j, c in zip(nn, sim)),
                     'pom_top6': '; '.join(f'{labels[j]} {s[i, j]:.2f}' for j in top)})
    with (a.out / 'placement.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=['name', 'smiles', 'inchikey', 'x', 'y', 'max_cos', 'neighbors', 'pom_top6'])
        w.writeheader(); w.writerows(rows)
    np.savez(a.out / 'new_embeddings.npz', smiles=np.array([m['smiles'] for m in new]), embeddings=z, descriptor_scores=s)

    loo = loo_errors(ref, xy, a.k)
    report = {'k': a.k, 'n_input': len(mols), 'n_already_on_map': len(mols) - len(new), 'n_placed': len(new),
              'parity_first8': parity,
              'loo_error_tsne_units': {'median': float(np.median(loo)), 'p90': float(np.percentile(loo, 90)),
                                       'map_span': float(np.ptp(xy, axis=0).max())},
              'note': 'Approximate positions on fixed coordinates; not part of the original t-SNE optimisation.'}
    (a.out / 'placement_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
