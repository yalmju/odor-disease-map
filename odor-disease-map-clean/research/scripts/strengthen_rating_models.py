"""Internal repeated nested molecule-group CV; original test labels are excluded.

Uses cached fixed OpenPOM embeddings. Pretraining overlap remains unresolved.
No NNLS, web UI, GPU or new downloads are needed.
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler

from voclib.odor_edit import molecule, features, rating_features, RATINGS
from voclib.odor_validation import select_ridge, ridge_model, tanimoto_matrix, delta_diagnostics, RatingRidgePredictor
from run_pom_mlp_pilot import train_head, RatingHead


def refit_head(x, y, train, epochs, seed, out):
    """Use selected epoch count, then refit from scratch on all outer train rows."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    scaler = StandardScaler().fit(x[train])
    xx = torch.tensor(scaler.transform(x), dtype=torch.float32)
    target = torch.tensor(y, dtype=torch.float32)
    net = RatingHead(x.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=.001)
    for _ in range(epochs):
        net.train()
        for idx in np.array_split(rng.permutation(train), max(1,int(np.ceil(len(train)/64)))):
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(net(xx[idx]), target[idx])
            loss.backward()
            opt.step()
    net.eval()
    torch.save({'state': net.state_dict(), 'mean': scaler.mean_.tolist(),
                'scale': scaler.scale_.tolist(), 'input_dim': x.shape[1],
                'epochs': epochs, 'seed': seed}, out)
    with torch.no_grad():
        return net(xx).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', type=Path, required=True)
    ap.add_argument('--cache', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError('Choose an empty output directory')
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    manifest = json.loads(args.split.read_text(encoding='utf-8'))
    # Do not use original test outcomes for tuning or evaluation in this run.
    rows = [r for r in manifest['ratings'] if r['split'] != 'test']
    excluded = {r['key'] for r in manifest['ratings'] if r['split'] == 'test'}
    groups = np.array([r['key'] for r in rows])
    assert not set(groups) & excluded
    y = np.array([r['y'] for r in rows], dtype=float)
    cache = np.load(args.cache, allow_pickle=False)
    lookup = {s: i for i, s in enumerate(cache['smiles'])}
    fp = np.array([features(molecule(r['smiles'])) for r in rows])
    conditions = np.array([rating_features(molecule(r['smiles']), r['concentration'])[-4:] for r in rows])
    z = cache['embeddings'][[lookup[r['smiles']] for r in rows]]
    inputs = dict(morgan=np.column_stack([fp, conditions]),
                  pom=np.column_stack([z, conditions]),
                  hybrid=np.column_stack([fp, z, conditions]))
    report = {'scope': 'Internal development CV, not fresh external validation. Frozen POM pretraining overlap unresolved.',
              'ratings': RATINGS, 'n_rows': len(rows), 'n_groups': len(set(groups)),
              'original_test_groups_excluded': len(excluded),
              'split_sha256': hashlib.sha256(args.split.read_bytes()).hexdigest(),
              'embedding_sha256': hashlib.sha256(args.cache.read_bytes()).hexdigest(),
              'protocol': '2 repeats x 3 outer molecule folds; 3 inner molecule folds select ridge alpha/representation by row-mean MAE. MLP inner validation selects epoch count, then full outer-train refit.',
              'environment': {'python': platform.python_version(), 'torch': torch.__version__},
              'repeats': []}
    names = ['condition_mean', 'morgan_mlp', 'pom_mlp', 'morgan_trees',
             'morgan_ridge', 'pom_ridge', 'hybrid_ridge', 'selected_ridge']
    for seed in [42, 137]:
        predictions = {name: np.empty_like(y) for name in names}
        assignments = np.zeros(len(rows), dtype=int)
        nearest = np.zeros(len(rows))
        selection = []
        for fold, (train, test) in enumerate(GroupKFold(3, shuffle=True, random_state=seed).split(y, groups=groups)):
            assert not set(groups[train]) & set(groups[test])
            assignments[test] = fold
            nearest[test] = tanimoto_matrix(fp[test], fp[train]).max(1)
            subinputs = {name: x[train] for name, x in inputs.items()}
            best, scores = select_ridge(subinputs, y[train], groups[train])
            selection.append({'fold': fold, 'best': best, 'inner_scores': scores})
            for name, x in inputs.items():
                chosen = min((s for s in scores if s['representation'] == name), key=lambda s: s['mae'])
                model = ridge_model(chosen['alpha']).fit(x[train], y[train])
                predictions[name+'_ridge'][test] = np.clip(model.predict(x[test]), 0, 1)
            predictions['selected_ridge'][test] = predictions[best['representation']+'_ridge'][test]
            for concentration in sorted({r['concentration'] for r in rows}):
                ti = [i for i in train if rows[i]['concentration'] == concentration]
                vi = [i for i in test if rows[i]['concentration'] == concentration]
                predictions['condition_mean'][vi] = y[ti].mean(0)
            tree = ExtraTreesRegressor(n_estimators=160, min_samples_leaf=3, max_features=.7, random_state=seed, n_jobs=2)
            predictions['morgan_trees'][test] = tree.fit(inputs['morgan'][train], y[train]).predict(inputs['morgan'][test])
            inner_train, inner_val = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed+fold).split(train, groups=groups[train]))
            for name in ['morgan', 'pom']:
                _, _, _, _, head_bundle = train_head(inputs[name], y,
                    {'train': train[inner_train], 'val': train[inner_val]},
                    args.out/f'{name}-seed{seed}-fold{fold}.pt', seed=seed+fold)
                pred = refit_head(inputs[name], y, train, head_bundle['best_epoch'], seed+fold,
                                  args.out/f'{name}-seed{seed}-fold{fold}-refit.pt')
                predictions[name+'_mlp'][test] = pred[test]
            print(f'seed={seed} fold={fold}: {best}', flush=True)
        metrics = {}
        for name, p in predictions.items():
            metrics[name] = {'mae': np.abs(y-p).mean(0).tolist(),
                             'mean_mae': float(np.abs(y-p).mean()),
                             'molecule_macro_mae': np.mean([np.abs(y[groups==g]-p[groups==g]).mean(0) for g in np.unique(groups)],axis=0).tolist()}
        delta, pairs = delta_diagnostics(rows, y, predictions['selected_ridge'], fp, assignments)
        domain = {}
        for label, mask in [('similarity_below_0.4', nearest < .4), ('similarity_at_least_0.4', nearest >= .4)]:
            domain[label] = {'n': int(mask.sum()), 'mae': np.abs(y[mask]-predictions['selected_ridge'][mask]).mean(0).tolist() if mask.any() else None}
        report['repeats'].append({'seed': seed, 'metrics': metrics, 'selection': selection, 'delta': delta, 'domain': domain})
        (args.out/f'pairs-seed{seed}.json').write_text(json.dumps(pairs, indent=2), encoding='utf-8')
        np.savez_compressed(args.out/f'oof-seed{seed}.npz', observed=y, fold=assignments,
                            smiles=np.array([r['smiles'] for r in rows]), group=groups,
                            concentration=np.array([r['concentration'] for r in rows]),
                            nearest_train_similarity=nearest, **predictions)
    # Deployable candidate selection uses internal CV on development rows only.
    best, scores = select_ridge(inputs, y, groups)
    model = ridge_model(best['alpha']).fit(inputs[best['representation']], y)
    metadata = args.cache.parent/'encoder_metadata.json'
    if not metadata.exists():
        raise ValueError('Embedding cache requires encoder_metadata.json provenance')
    bundle = {**best, 'model': model, 'ratings': RATINGS, 'trained_groups': sorted(set(groups)),
              'encoder': json.loads(metadata.read_text(encoding='utf-8')),
              'scope': report['scope'], 'conditions': [1e-3, 1e-5],
              'feature_layout': 'Morgan2048 and/or POM256, then MW/logP/TPSA/log10(dilution)'}
    joblib.dump(bundle, args.out/'rating_ridge_candidate.joblib')
    loaded = RatingRidgePredictor(joblib.load(args.out/'rating_ridge_candidate.joblib'))
    np.testing.assert_allclose(loaded.predict_features(inputs, bundle['encoder']), np.clip(model.predict(inputs[best['representation']]),0,1))
    report['final_candidate'] = best
    report['final_inner_scores'] = scores
    report['checks'] = {'outer_groups_disjoint': True, 'original_test_excluded': True, 'serialization_roundtrip': True}
    report['aggregate'] = {name: {'mae_mean_over_repeats': np.mean([r['metrics'][name]['mae'] for r in report['repeats']], axis=0).tolist(),
                                 'mean_mae': float(np.mean([r['metrics'][name]['mean_mae'] for r in report['repeats']]))} for name in names}
    (args.out/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    lines = ['# 문헌 기반 지각 평가 모델 강화', '', report['scope'], '',
             f"개발 데이터 {len(rows)}행 / {len(set(groups))}분자 그룹. 기존 시험 그룹 {len(excluded)}개 제외.", '',
             '| Model | Pleasantness MAE | Intensity MAE | Familiarity MAE | Mean |',
             '|---|---:|---:|---:|---:|']
    for name, m in report['aggregate'].items():
        lines.append('| '+name+' | '+' | '.join(f'{v:.4f}' for v in m['mae_mean_over_repeats'])+f" | {m['mean_mae']:.4f} |")
    lines += ['', '숫자는 2회 반복의 평균이며 독립 표본 수가 두 배가 되는 것은 아니다. 통계적 유의성·새 시험 성능을 주장하지 않는다.',
              '', f'저장 후보: {best}. 기존 GUI의 활성 모델은 바꾸지 않았다.', '',
              '세부 결과: report.json에 fold별 선택, 같은 조건의 유사 구조쌍 Δ 오차, 학습 구조와의 거리별 오차를 저장했다. 유사 구조쌍은 확인된 단일 치환쌍이 아니다.',
              '혼합물·기체 노출·개인별 지각·SERS 반응을 학습한 모델이 아니다. POM 사전학습 중복은 해결하지 못했다.']
    (args.out/'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(report['aggregate'], indent=2), flush=True)


if __name__ == '__main__':
    main()
