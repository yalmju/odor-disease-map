"""Rank disease-discordant molecule pairs in OpenPOM embedding space and compare with structure.

For every (disease-linked, unlinked) pair of map molecules that both carry an observed odor label,
report the embedding cosine similarity and Morgan Tanimoto similarity, the rank of the focus pair,
and how the embedding space agrees with structural and predicted-odor similarity overall.
This is post-hoc support for a hand-picked pair, not a validated selection algorithm.

Example:
  python scripts/pair_ranking.py --cache private_data/inference_cache.npz \
    --overlay private_data/evidence_overlay.json --out results/pair_ranking
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crc_tools import discordant_pairs, neighbor_overlap, unit_rows  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--overlay', type=Path, required=True)
    p.add_argument('--disease', default='Colorectal cancer')
    p.add_argument('--top', type=int, default=200, help='Rows written to discordant_pairs.csv')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()

    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem
    from scipy.stats import spearmanr
    RDLogger.DisableLog('rdApp.*')

    cache = np.load(a.cache, allow_pickle=False)
    pts = json.loads(a.overlay.read_text(encoding='utf-8-sig'))['points']
    smiles = cache['smiles'].tolist()
    if [q['smiles'] for q in pts] != smiles:
        raise ValueError('Overlay points and cache rows must be in the same order')
    names = [q.get('name', str(q['index'])) for q in pts]
    Z = unit_rows(cache['embeddings']); C = Z @ Z.T
    S = unit_rows(cache['descriptor_scores']); D = S @ S.T
    fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, 2048) for s in smiles]
    T = np.array([DataStructs.BulkTanimotoSimilarity(f, fps) for f in fps])

    n = len(smiles); iu = np.triu_indices(n, 1)
    agreement = {
        'spearman_embedding_vs_morgan': float(spearmanr(C[iu], T[iu])[0]),
        'spearman_embedding_vs_predicted_profile': float(spearmanr(C[iu], D[iu])[0]),
        'knn10_overlap_embedding_vs_morgan': neighbor_overlap(C, T, 10),
        'knn10_overlap_embedding_vs_predicted_profile': neighbor_overlap(C, D, 10),
        'knn10_overlap_random_expectation': 10 / (n - 1),
        'note': 'Predicted profiles come from the same network, so that comparison is partly circular.'}

    linked = np.array([a.disease in q['diseases'] for q in pts])
    labeled = np.array([bool(q['odors']) for q in pts])
    ii, jj, ss = discordant_pairs(C, linked, labeled)
    fi, fj = [smiles.index(f['smiles']) for f in json.loads(a.overlay.read_text(encoding='utf-8-sig'))['focus'][:2]]
    hit = np.where(((ii == fi) & (jj == fj)) | ((ii == fj) & (jj == fi)))[0]
    focus = {'pair': [names[fi], names[fj]], 'cosine': float(C[fi, fj]), 'tanimoto': float(T[fi, fj]),
             'rank_among_discordant': int(hit[0] + 1) if len(hit) else None, 'n_discordant_pairs': int(len(ss)),
             'distribution': {'median': float(np.median(ss)), 'p90': float(np.percentile(ss, 90)), 'p98': float(np.percentile(ss, 98))}}

    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / 'discordant_pairs.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['rank', 'linked', 'unlinked', 'embedding_cosine', 'morgan_tanimoto', 'linked_odors', 'unlinked_odors'])
        for r, (i, j, s) in enumerate(zip(ii[:a.top], jj[:a.top], ss[:a.top]), 1):
            w.writerow([r, names[i], names[j], f'{s:.4f}', f'{T[i, j]:.3f}', '|'.join(pts[i]['odors']), '|'.join(pts[j]['odors'])])
    report = {'agreement': agreement, 'focus_pair': focus,
              'interpretation': 'Similarity is relative to this library; not perceptual equivalence, a p-value or diagnostic evidence.'}
    (a.out / 'pair_ranking.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
