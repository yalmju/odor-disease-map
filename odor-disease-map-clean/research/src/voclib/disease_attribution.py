"""Exact decomposition and removal sensitivity of disease-odor annotation contrasts.

This explains an empirical DB profile, not patient predictions or mixture perception.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import fisher_exact
from rdkit import Chem
from .associations import bh_adjust
from .odor_edit import molecule, INDOLE


def decompose(target, background):
    target=np.asarray(target,float);background=np.asarray(background,float)
    if target.ndim!=2 or background.ndim!=2 or not len(target) or not len(background):
        raise ValueError('Two nonempty aligned matrices required')
    reference=background.mean(0)
    delta=target.mean(0)-reference
    contributions=(target-reference)/len(target)
    norm2=float(delta@delta)
    shares=contributions@delta/norm2 if norm2>1e-15 else np.zeros(len(target))
    np.testing.assert_allclose(contributions.sum(0),delta,atol=1e-12)
    return delta,contributions,shares


def contrast_test(target,background,labels):
    rows=[]
    for j,label in enumerate(labels):
        a=int(target[:,j].sum());b=int(background[:,j].sum())
        p=float(fisher_exact([[a,len(target)-a],[b,len(background)-b]],alternative='two-sided').pvalue)
        rows.append(dict(label=label,target_count=a,background_count=b,
                         difference_pp=float(100*(a/len(target)-b/len(background))),p=p))
    for row,q in zip(rows,bh_adjust([r['p'] for r in rows])):row['q']=q
    return rows


def removal_metrics(target,background,remove,full_delta):
    retained=np.delete(target,remove,axis=0)
    if not len(retained):return None
    delta=retained.mean(0)-background.mean(0)
    norm=np.linalg.norm(full_delta)
    return {'remaining_target_n':len(retained),
            'relative_profile_change':float(np.linalg.norm(delta-full_delta)/norm) if norm else None,
            'remaining_contrast_norm_ratio':float(np.linalg.norm(delta)/norm) if norm else None,
            'cosine_with_full':float(delta@full_delta/(np.linalg.norm(delta)*norm)) if norm and np.linalg.norm(delta) else None,
            'delta':delta.tolist()}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--disease',default='Colorectal cancer')
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    source=json.loads(args.input.read_text(encoding='utf-8'))
    rows=[r for r in source if r['clinical']['diseases']]
    assert len(rows)==len({r['inchikey'] for r in rows})
    labels=sorted({o for r in rows for o in r['observed_odors']})
    targets=[r for r in rows if args.disease in r['clinical']['diseases']]
    background=[r for r in rows if args.disease not in r['clinical']['diseases']]
    def matrix(records):return np.array([[int(o in r['observed_odors']) for o in labels] for r in records],float)
    x,b=matrix(targets),matrix(background)
    delta,contributions,shares=decompose(x,b)
    skatole_key=Chem.MolToInchiKey(molecule('Cc1c[nH]c2ccccc12'))
    family=[i for i,r in enumerate(targets) if molecule(r['smiles']).HasSubstructMatch(INDOLE)]
    skatole=[i for i,r in enumerate(targets) if r['inchikey']==skatole_key]
    family_bg=[i for i,r in enumerate(background) if molecule(r['smiles']).HasSubstructMatch(INDOLE)]
    tests=contrast_test(x,b,labels)
    ranked=[]
    for i,r in enumerate(targets):
        loo=removal_metrics(x,b,[i],delta)
        ranked.append({'name':r['name'],'inchikey':r['inchikey'],'smiles':r['smiles'],
                       'observed_odors':r['observed_odors'],
                       'signed_projection_share':float(shares[i]),
                       'profile_change_on_removal':loo['relative_profile_change'],
                       'descriptor_contribution':dict(zip(labels,contributions[i].tolist()))})
    ranked.sort(key=lambda r:-r['signed_projection_share'])
    ablations={};removal_nulls={}
    for name,idx in [('without_skatole',skatole),('without_indole_family',family)]:
        metrics=removal_metrics(x,b,idx,delta)
        if not metrics:continue
        if idx:
            size=len(idx)
            if size not in removal_nulls:
                rng=np.random.default_rng(42+size)
                subsets=[[i] for i in range(len(x))] if size==1 else [rng.choice(len(x),size,replace=False) for _ in range(2000)]
                removal_nulls[size]=[removal_metrics(x,b,s,delta)['relative_profile_change'] for s in subsets]
            random_changes=removal_nulls[size]
            metrics['random_same_size_removal_percentile']=float(np.mean(np.array(random_changes)<=metrics['relative_profile_change'])*100)
        metrics['removed']=[targets[i]['name'] for i in idx]
        metrics['sum_signed_projection_share']=float(shares[idx].sum())
        metrics['tests']=contrast_test(np.delete(x,idx,axis=0),b,labels)
        ablations[name]=metrics
    both_t=np.delete(x,family,axis=0);both_b=np.delete(b,family_bg,axis=0)
    both_delta=both_t.mean(0)-both_b.mean(0)
    ablations['remove_family_from_both_groups']={'target_n':len(both_t),'background_n':len(both_b),
        'removed_background':[background[i]['name'] for i in family_bg],
        'relative_profile_change':float(np.linalg.norm(both_delta-delta)/np.linalg.norm(delta)),
        'tests':contrast_test(both_t,both_b,labels)}
    only={name:{'n':len(idx),'mean_annotation_vector':x[idx].mean(0).tolist() if idx else None,
                'scope':'Subset profile only; not a disease classifier.'} for name,idx in [('skatole_only',skatole),('indole_family_only',family)]}
    report={'disease':args.disease,'target_n':len(x),'background_n':len(b),'labels':labels,
            'scope':'Explains the observed DB annotation contrast, not patient discrimination or odor mixture interactions.',
            'definition':'delta=mean(target)-mean(background). c_i=(x_i-mean(background))/N. signed_share_i=dot(c_i,delta)/dot(delta,delta).',
            'interpretation':'Signed shares sum to one, can be negative, depend on labels/background, and are not causal importance or clinical probability.',
            'fixed_background':'Primary ablations hold background fixed; remove-both-groups sensitivity is separate.',
            'substructure':'RDKit aromatic indole SMARTS c1ccc2[nH,n]ccc2c1; family uses only observed-odor atlas records.',
            'full_delta':delta.tolist(),'full_tests':tests,'ranking':ranked,'ablations':ablations,'subset_profiles':only,
            'limitations':['Missing disease annotation is unknown, not healthy.','No concentration weights or molecular interactions modeled.',
                           'Fisher/BH analyses exploratory; structure/source dependencies unmodeled; corrections within each scenario only.',
                           'Unobserved derivatives are excluded instead of inserting model predictions as observations.']}
    (args.out/'attribution.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 스캐톨이 질병–향 주석 차이를 설명하는가?','',report['scope'],'',
           f'대장암 주석 분자 {len(x)}개 / 비교 분자 {len(b)}개 / 향 주석 {len(labels)}개.',
           '', '## 분자 제거 실험', '', '| 제거 대상 | 제거 분자 수 | 부호 있는 기여율 합 | 전체 패턴 변화 |',
           '|---|---:|---:|---:|']
    for name,m in ablations.items():
        if 'removed' in m:lines.append(f"| {name} | {len(m['removed'])} | {100*m['sum_signed_projection_share']:.2f}% | {100*m['relative_profile_change']:.2f}% |")
    lines+=['','기여율은 104차원 주석 차이 벡터 방향으로 투영한 몫이다. 모델 정확도·실제 냄새 기여도·질병 위험도가 아니다. 패턴 변화는 제거 전후 차이 벡터의 L2 변화 / 원래 벡터 L2 크기다.',
            '', '## 제거한 인돌계열', '', ', '.join(targets[i]['name'] for i in family), '',
            '관측 향이 없는 IAA/IPA/ILA 등을 예측값으로 보충하지 않았다. 스캐톨 단독·계열 단독 평균 벡터는 JSON에 저장했다.',
            '', '## 전체 분자의 기여 순위 (상위 10개)', '', '| 분자 | 부호 있는 기여율 | 제거 시 패턴 변화 |','|---|---:|---:|']
    for r in ranked[:10]:lines.append(f"| {r['name']} | {100*r['signed_projection_share']:.2f}% | {100*r['profile_change_on_removal']:.2f}% |")
    lines+=['','모든 분자를 같은 조건으로 평가했으며 스캐톨을 특별 취급해 점수를 높이지 않았다. 분자 하나에 여러 향 주석이 가능하다. 동일한 주석 벡터는 동일한 점수를 받는다. 원문 근거 심사 및 독립 코호트 검증은 아직 하지 않았다.']
    (args.out/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'skatole':[r for r in ranked if r['inchikey']==skatole_key],
                      'family':[targets[i]['name'] for i in family],
                      'ablations':{k:{j:v for j,v in m.items() if j not in ['tests','delta']} for k,m in ablations.items()}},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
