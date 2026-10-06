"""Train reproducible structure-to-odor baselines and compare aromatic H substitutions.

Predictions and tree/bootstrap dispersion are not measured odors or calibrated confidence.
Only trusted, locally trained joblib bundles should be loaded.
"""
from __future__ import annotations
import argparse, collections, csv, hashlib, json, platform
from pathlib import Path
import joblib
import numpy as np
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import Crippen, Descriptors, rdFingerprintGenerator
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
import sklearn

FP = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
RATINGS = ['Pleasantness', 'Intensity', 'Familiarity']
QUESTIONS = {'HOW PLEASANT IS THE SMELL?': 0, 'HOW STRONG IS THE SMELL?': 1,
             'HOW FAMILIAR IS THE SMELL?': 2}
SUBSTITUENTS = {'methyl': 'C', 'ethyl': 'CC', 'hydroxy': 'O', 'methoxy': 'OC',
                'fluoro': 'F', 'chloro': 'Cl', 'acetic_acid': 'CC(=O)O',
                'propionic_acid': 'CCC(=O)O'}
INDOLE = Chem.MolFromSmarts('c1ccc2[nH,n]ccc2c1')

def molecule(smiles):
    m = Chem.MolFromSmiles(smiles)
    if m is None or m.GetNumAtoms() == 0:
        raise ValueError('유효한 SMILES를 입력하세요.')
    if len(Chem.GetMolFrags(m)) != 1:
        raise ValueError('이 모델은 단일 연결 구조만 지원합니다. 혼합물·염은 별도 전처리가 필요합니다.')
    return Chem.MolFromSmiles(Chem.MolToSmiles(m, isomericSmiles=True))

def identity(m):
    return Chem.MolToInchiKey(m).split('-')[0]

def features(m):
    return FP.GetFingerprintAsNumPy(m).astype(np.float32)

def rating_features(m, concentration):
    if concentration not in (0.001, 0.00001):
        raise ValueError('Keller 평가 조건 0.001 또는 0.00001만 지원합니다.')
    return np.r_[features(m), Descriptors.MolWt(m), Crippen.MolLogP(m),
                 Descriptors.TPSA(m), np.log10(concentration)].astype(np.float32)

def editable_sites(smiles):
    m = molecule(smiles)
    return [{'atom_index': a.GetIdx(), 'element': a.GetSymbol()} for a in m.GetAtoms()
            if a.GetIsAromatic() and a.GetSymbol() in ('C', 'N') and a.GetTotalNumHs() > 0]

def substitute(smiles, atom_index, substituent):
    m = molecule(smiles)
    if atom_index not in {s['atom_index'] for s in editable_sites(smiles)}:
        raise ValueError('치환 위치는 수소가 있는 방향족 C 또는 N이어야 합니다.')
    if substituent not in SUBSTITUENTS:
        raise ValueError('지원하지 않는 치환기입니다.')
    frag = Chem.MolFromSmiles(SUBSTITUENTS[substituent])
    edit = Chem.RWMol(Chem.CombineMols(m, frag))
    a = edit.GetAtomWithIdx(atom_index)
    if a.GetNumExplicitHs():
        a.SetNumExplicitHs(a.GetNumExplicitHs() - 1)
    edit.AddBond(atom_index, m.GetNumAtoms(), Chem.BondType.SINGLE)
    result = edit.GetMol()
    Chem.SanitizeMol(result)
    return Chem.MolToSmiles(result, isomericSmiles=True)

def enumerate_edits(smiles):
    seen = {Chem.MolToSmiles(molecule(smiles))}; result = []
    for site in editable_sites(smiles):
        for name in SUBSTITUENTS:
            try:
                s = substitute(smiles, site['atom_index'], name)
            except (ValueError, RuntimeError):
                continue
            if s not in seen:
                result.append(dict(site, substituent=name, smiles=s)); seen.add(s)
    return result

def split_groups(groups, seed=42):
    groups = np.asarray(groups)
    train, rest = next(GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed).split(groups, groups=groups))
    val_part, test_part = next(GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed+1).split(rest, groups=groups[rest]))
    return {'train': train, 'val': rest[val_part], 'test': rest[test_part]}

def fit_classifiers(x, y, groups, members=3):
    rng = np.random.default_rng(42); result = []
    unique = np.unique(groups)
    group_rows = {g: np.flatnonzero(np.asarray(groups) == g) for g in unique}
    for member in range(members):
        idx = np.arange(len(x)) if member == 0 else np.concatenate([group_rows[g] for g in rng.choice(unique, len(unique), replace=True)])
        models = []
        for j in range(y.shape[1]):
            target = y[idx, j]
            if len(np.unique(target)) == 1:
                models.append(float(target[0])); continue
            model = LogisticRegression(C=1, solver='liblinear', max_iter=1000, random_state=42)
            model.fit(x[idx], target); models.append(model)
        result.append(models)
    return result

def descriptor_scores(models, x):
    return np.array([np.column_stack([np.full(len(x), m) if isinstance(m, float)
                    else m.predict_proba(x)[:, 1] for m in member]) for member in models])

def classification_metrics(y, p, thresholds, labels):
    per_label = {}
    for j, label in enumerate(labels):
        pos = int(y[:, j].sum()); neg = len(y)-pos
        per_label[label] = {'positive': pos, 'negative': neg,
            'average_precision': float(average_precision_score(y[:, j], p[:, j])) if pos and neg else None,
            'roc_auc': float(roc_auc_score(y[:, j], p[:, j])) if pos and neg else None,
            'f1': float(f1_score(y[:, j], p[:, j] >= thresholds[j], zero_division=0))}
    aps = [r['average_precision'] for r in per_label.values() if r['average_precision'] is not None]
    aucs = [r['roc_auc'] for r in per_label.values() if r['roc_auc'] is not None]
    return {'macro_average_precision': float(np.mean(aps)) if aps else None,
            'macro_roc_auc': float(np.mean(aucs)) if aucs else None,
            'macro_f1': float(np.mean([r['f1'] for r in per_label.values()])),
            'evaluable_labels': len(aps), 'per_label': per_label}

def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        yield from csv.DictReader(f)

def load_descriptors(root):
    records = []; rejected = []; labels = None
    for original_split in ('train', 'val', 'test'):
        for row in read_csv(root / 'data' / (original_split+'.csv')):
            if labels is None: labels = [k for k in row if k != 'smiles']
            try: m = molecule(row['smiles'])
            except ValueError as e:
                rejected.append({'smiles': row['smiles'], 'reason': str(e)}); continue
            records.append({'smiles': Chem.MolToSmiles(m), 'key': identity(m),
                            'y': [int(row[k]) for k in labels], 'original_split': original_split,
                            'indole': m.HasSubstructMatch(INDOLE)})
    return records, labels, rejected

def load_ratings(root):
    mols = {}
    for r in read_csv(root/'keller_2016_molecules.csv'):
        try: m = molecule(r['CanonicalSMILES'])
        except ValueError: continue
        mols[r['CID']] = (Chem.MolToSmiles(m), identity(m))
    stimuli = {}
    for r in read_csv(root/'keller_2016_stimuli.csv'):
        if r['CIDs'] in mols and float(r['Concentration']) in (0.001, 0.00001):
            stimuli[r['Stimulus']] = (*mols[r['CIDs']], float(r['Concentration']))
    values = collections.defaultdict(list)
    for r in read_csv(root/'keller_2016_behavior.csv'):
        if r['MeasurementValue'] not in QUESTIONS or r['Stimulus'] not in stimuli: continue
        try: value = float(r['Value'])
        except ValueError: continue
        if not np.isfinite(value) or not 0 <= value <= 100: continue
        s, k, c = stimuli[r['Stimulus']]
        values[(s, k, c, r['Subject'], QUESTIONS[r['MeasurementValue']])].append(value/100)
    aggregated = collections.defaultdict(lambda: collections.defaultdict(list))
    for (s,k,c,subject,j), vals in values.items():
        aggregated[(s,k,c)][j].append(float(np.mean(vals)))
    return [{'smiles':s,'key':k,'concentration':c,'y':[float(np.mean(v[j])) for j in range(3)],
             'n_subjects':[len(v[j]) for j in range(3)]} for (s,k,c),v in aggregated.items() if len(v)==3]

def rating_condition_evaluation(reg, records):
    """Compare at fixed dilution, so concentration alone cannot explain performance."""
    result = {}
    for concentration in (.001, .00001):
        train_rows=[r for r in records if r['split']=='train' and r['concentration']==concentration]
        test_rows=[r for r in records if r['split']=='test' and r['concentration']==concentration]
        x=np.array([rating_features(molecule(r['smiles']),concentration) for r in test_rows])
        target=np.array([r['y'] for r in test_rows]);pred=reg.predict(x)
        mean=np.array([r['y'] for r in train_rows]).mean(axis=0)
        result[str(concentration)]={'n':len(test_rows),'metrics':{}}
        for j,name in enumerate(RATINGS):
            result[str(concentration)]['metrics'][name]={
                'mae':float(mean_absolute_error(target[:,j],pred[:,j])),
                'same_concentration_baseline_mae':float(np.mean(np.abs(target[:,j]-mean[j]))),
                'pearson_r':float(np.corrcoef(target[:,j],pred[:,j])[0,1]) if np.std(pred[:,j])>0 and np.std(target[:,j])>0 else None}
    return result

def train(hari_root, keller_root, out):
    out = Path(out)
    if out.exists() and any(out.iterdir()): raise ValueError('비어 있는 새 실행 폴더를 지정하세요.')
    out.mkdir(parents=True, exist_ok=True)
    records, labels, rejected = load_descriptors(Path(hari_root))
    x = np.array([features(molecule(r['smiles'])) for r in records]); y = np.array([r['y'] for r in records])
    groups = np.array([r['key'] for r in records]); split = split_groups(groups)
    tr, va, te = [split[s] for s in ('train','val','test')]
    print(f'Descriptors: {len(records)} molecules; split {len(tr)}/{len(va)}/{len(te)}', flush=True)
    models = fit_classifiers(x[tr], y[tr], groups[tr])
    pv = descriptor_scores(models, x[va]).mean(axis=0)
    thresholds = np.array([max(np.arange(.05,.81,.05), key=lambda t: f1_score(y[va,j],pv[:,j]>=t,zero_division=0)) for j in range(len(labels))])
    pt = descriptor_scores(models,x[te]).mean(axis=0)
    desc_report = classification_metrics(y[te],pt,thresholds,labels)
    desc_report['baseline'] = classification_metrics(y[te],np.tile(y[tr].mean(axis=0),(len(te),1)),thresholds,labels)
    indole_idx = np.array([i for i,r in enumerate(records) if r['indole']], dtype=int)
    non_indole = np.array([i for i,r in enumerate(records) if not r['indole']], dtype=int)
    family_report = None
    if len(indole_idx):
        print('Independent indole-family holdout baseline...', flush=True)
        family_models = fit_classifiers(x[non_indole], y[non_indole], groups[non_indole], members=1)
        pf = descriptor_scores(family_models,x[indole_idx])[0]
        family_report = {'n':len(indole_idx), 'note':'Separate diagnostic model: trained on all non-indole records, fixed threshold 0.5, not the deployment model; very small family sample.',
                         'metrics':classification_metrics(y[indole_idx],pf,np.full(len(labels),.5),labels),
                         'predictions':[{'smiles':records[i]['smiles'],'observed':[labels[j] for j in np.flatnonzero(y[i])],
                                         'scores':dict(zip(labels,p.tolist()))} for i,p in zip(indole_idx,pf)]}
    print('Aggregating Keller ratings by molecule / concentration / subject...', flush=True)
    ratings = load_ratings(Path(keller_root)); rg = np.array([r['key'] for r in ratings]); rs = split_groups(rg)
    rx = np.array([rating_features(molecule(r['smiles']),r['concentration']) for r in ratings]); ry = np.array([r['y'] for r in ratings])
    reg = ExtraTreesRegressor(n_estimators=160, min_samples_leaf=3, max_features=.7, random_state=42, n_jobs=2)
    reg.fit(rx[rs['train']],ry[rs['train']]); pred = reg.predict(rx[rs['test']]); target = ry[rs['test']]
    rating_report = {}
    for j,name in enumerate(RATINGS):
        rating_report[name] = {'mae':float(mean_absolute_error(target[:,j],pred[:,j])),
          'baseline_mae':float(mean_absolute_error(target[:,j],np.full(len(target),ry[rs['train'],j].mean()))),
          'pearson_r':float(np.corrcoef(target[:,j],pred[:,j])[0,1]) if np.std(pred[:,j])>0 and np.std(target[:,j])>0 else None}
    for r in records: r['split'] = ''
    for s,idx in split.items():
        for i in idx: records[i]['split'] = s
    for s,idx in rs.items():
        for i in idx: ratings[i]['split'] = s
    inputs = list((Path(hari_root)/'data').glob('*.csv'))+list(Path(keller_root).glob('keller_2016_*.csv'))
    report = {'version':1,'seed':42,'environment':{'python':platform.python_version(),'rdkit':rdBase.rdkitVersion,'sklearn':sklearn.__version__},
      'sources':{'descriptors':'https://huggingface.co/datasets/Hari5115/molecular-odor-dataset','ratings':'https://github.com/pyrfume/pyrfume-data/tree/main/keller_2016'},
      'inputs':[{'file':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in inputs],
      'split_policy':'70/15/15 by InChIKey connectivity; original HF splits replaced. Stereoisomers and concentrations stay within one partition. Not a scaffold split.',
      'descriptor_counts':{s:len(i) for s,i in split.items()},'descriptor_rejected':rejected,
      'descriptor_test':desc_report,'indole_family_holdout':family_report,
      'rating_counts':{s:len(i) for s,i in rs.items()},'rating_unique_connectivity':len(set(rg)),
      'rating_test':rating_report,'rating_test_by_concentration':rating_condition_evaluation(reg, ratings),
      'rating_condition':'Keller 2016 paraffin oil, 1/1000 or 1/100000; within-subject repeats averaged, then subjects averaged; values divided by 100.',
      'limitations':['Descriptor scores are not calibrated probabilities. Missing annotations are encoded zero, not experimentally established negatives.',
        'Bootstrap/tree dispersion is not a confidence interval. Structural similarity is not validated applicability.',
        'Predicted edit differences are hypotheses, not experimentally validated or causal substituent effects.',
        'Human ratings are population means at specified dilution, not personal response or headspace intensity.',
        'No claim of superiority to published HF model; splits and labels differ.']}
    bundle={'version':1,'labels':labels,'models':models,'thresholds':thresholds,'regressor':reg,'records':records,'ratings':ratings,'report':report}
    joblib.dump(bundle,out/'model.joblib',compress=3)
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'split_manifest.json').write_text(json.dumps({'descriptors':records,'ratings':ratings},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'descriptor_test':{k:v for k,v in desc_report.items() if k not in ('per_label','baseline')},'rating_test':rating_report},indent=2),flush=True)
    return bundle

class OdorPredictor:
    def __init__(self, path):
        self.bundle = joblib.load(path)
        if self.bundle.get('version') != 1: raise ValueError('지원하지 않는 모델 버전입니다.')
        self.train_records = [r for r in self.bundle['records'] if r['split']=='train']
        self.train_fps = [FP.GetFingerprint(molecule(r['smiles'])) for r in self.train_records]
        self.rating_train = [r for r in self.bundle['ratings'] if r['split']=='train']
        self.rating_fps = [FP.GetFingerprint(molecule(r['smiles'])) for r in self.rating_train]

    def predict_many(self, smiles, concentration=.001):
        mols=[molecule(s) for s in smiles]; x=np.array([features(m) for m in mols])
        score_members=descriptor_scores(self.bundle['models'],x);scores=score_members.mean(axis=0);spread=score_members.std(axis=0)
        rx=np.array([rating_features(m,concentration) for m in mols]);reg=self.bundle['regressor']
        tree_predictions=np.array([tree.predict(rx) for tree in reg.estimators_]);rp=tree_predictions.mean(axis=0);rsp=tree_predictions.std(axis=0)
        result=[]
        for i,m in enumerate(mols):
            fp=FP.GetFingerprint(m); sims=DataStructs.BulkTanimotoSimilarity(fp,self.train_fps)
            near=int(np.argmax(sims)); rsims=DataStructs.BulkTanimotoSimilarity(fp,self.rating_fps)
            observed=[r for r in self.bundle['records'] if r['smiles']==Chem.MolToSmiles(m)]
            result.append({'smiles':Chem.MolToSmiles(m),'concentration':concentration,
                'descriptors':{label:{'score':float(scores[i,j]),'bootstrap_sd':float(spread[i,j]),'selected':bool(scores[i,j]>=self.bundle['thresholds'][j])} for j,label in enumerate(self.bundle['labels'])},
                'ratings':{name:{'score':float(rp[i,j]),'tree_sd':float(rsp[i,j])} for j,name in enumerate(RATINGS)},
                'nearest_descriptor_train':{'smiles':self.train_records[near]['smiles'],'similarity':float(sims[near])},
                'nearest_rating_train_similarity':float(max(rsims)),
                'observed_annotations':[{'labels':[self.bundle['labels'][j] for j,v in enumerate(r['y']) if v], 'split':r['split']} for r in observed],
                'molecular_weight':float(Descriptors.MolWt(m)),'logp_predicted':float(Crippen.MolLogP(m)),
                'evidence':'model_prediction; observational annotations stored separately'})
        return result

    def compare(self, base, candidate, concentration=.001):
        a,b=self.predict_many([base,candidate],concentration)
        delta={k:b['descriptors'][k]['score']-a['descriptors'][k]['score'] for k in self.bundle['labels']}
        return {'base':a,'candidate':b,'descriptor_delta':delta,
                'rating_delta':{k:b['ratings'][k]['score']-a['ratings'][k]['score'] for k in RATINGS},
                'interpretation':'Predicted difference under the same rating dilution, not a measured or causal effect.'}

    def scan(self, base, concentration=.001):
        edits=enumerate_edits(base);predictions=self.predict_many([base]+[r['smiles'] for r in edits],concentration);a=predictions[0]
        for edit,p in zip(edits,predictions[1:]):
            edit['prediction']=p
            edit['descriptor_delta']={k:p['descriptors'][k]['score']-a['descriptors'][k]['score'] for k in self.bundle['labels']}
            edit['rating_delta']={k:p['ratings'][k]['score']-a['ratings'][k]['score'] for k in RATINGS}
            edit['mean_absolute_descriptor_change']=float(np.mean(np.abs(list(edit['descriptor_delta'].values()))))
        return {'base':a,'candidates':sorted(edits,key=lambda r:r['mean_absolute_descriptor_change'],reverse=True),
                'note':'Ranking by predicted descriptor change only; not odor potency, synthesis feasibility or validated priority.'}

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    t=sub.add_parser('train');t.add_argument('--hari',type=Path,required=True);t.add_argument('--keller',type=Path,required=True);t.add_argument('--out',type=Path,required=True)
    for command in ('compare','scan'):
        c=sub.add_parser(command);c.add_argument('--model',type=Path,required=True);c.add_argument('--base',required=True);c.add_argument('--concentration',type=float,default=.001);c.add_argument('--out',type=Path,required=True)
        if command=='compare':c.add_argument('--candidate',required=True)
    args=p.parse_args()
    if args.command=='train':train(args.hari,args.keller,args.out)
    else:
        predictor=OdorPredictor(args.model)
        result=predictor.compare(args.base,args.candidate,args.concentration) if args.command=='compare' else predictor.scan(args.base,args.concentration)
        args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(args.out)

if __name__=='__main__':main()
