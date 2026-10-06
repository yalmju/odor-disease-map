"""Evidence-gated panel design with auditable structural edits and Pareto tradeoffs.

Ranks experiment proposals, not disease risk or validated sensor efficacy.
Unknown delivery remains unknown, and disease label count is never a benefit score.
"""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
from rdkit import Chem, DataStructs
from .odor_edit import molecule, FP, enumerate_edits, OdorPredictor
from .disease_bridge import DiseaseBridge


def gate(candidate, sources):
    source=candidate.get('gas_source')
    if source and source in sources and sources[source].get('doi'):
        return 'literature_supported_delivery_candidate'
    return 'delivery_evidence_missing'


def pareto_indices(values):
    x=np.asarray(values,float)
    if len(x)==0:return []
    if not np.isfinite(x).all():raise ValueError('Pareto criteria must be finite')
    return [i for i in range(len(x)) if not any(np.all(x[j]>=x[i]) and np.any(x[j]>x[i]) for j in range(len(x)) if j!=i)]


def edit_proof(a,b):
    """Exact sanitized molecular identity after supported H substitution, not similarity."""
    ac=Chem.MolToSmiles(molecule(a));bc=Chem.MolToSmiles(molecule(b))
    for base,target,direction in [(ac,bc,'A_to_B'),(bc,ac,'B_to_A')]:
        for edit in enumerate_edits(base):
            if edit['smiles']==target:
                return {'direction':direction,'atom_index':edit['atom_index'],'substituent':edit['substituent'],
                        'scope':'Verified supported aromatic H substitution, not an exhaustive MMP search'}
    return None


def design(candidates,predictions,sources,bridge,budget=4):
    if budget<2:raise ValueError('Budget must allow at least two molecules')
    if len(candidates)!=len(predictions):raise ValueError('Candidate/prediction alignment mismatch')
    mols=[molecule(c['smiles']) for c in candidates]
    keys=[Chem.MolToInchiKey(m) for m in mols]
    if len(set(keys))!=len(keys):raise ValueError('Duplicate full molecular identity')
    fps=[FP.GetFingerprint(m) for m in mols]
    labels=sorted(predictions[0]['descriptors'])
    scores=np.array([[p['descriptors'][k]['score'] for k in labels] for p in predictions])
    cards=[]
    for c,p,key in zip(candidates,predictions,keys):
        evidence=bridge.lookup(c['smiles'])
        unknowns=['No delivery measurement under the proposed Au-SAM gas conditions',
                  'No validated Au-MBA/MPY response predictor', 'No clinical specificity established']
        if not p['observed_annotations']:unknowns.append('No observed odor labels in current model source')
        if gate(c,sources)=='delivery_evidence_missing':unknowns.append('No verified headspace source in this evidence ledger')
        spreads=[p['descriptors'][k]['bootstrap_sd'] for k in labels if p['descriptors'][k]['bootstrap_sd'] is not None]
        cards.append({**c,'inchikey':key,'gate':gate(c,sources),'ready_for_validated_gas_experiment':False,
                      'prediction':p,'clinical':evidence,
                      'uncertainty':{'mean_descriptor_bootstrap_sd':float(np.mean(spreads)) if spreads else None,
                                     'nearest_training_similarity':p['nearest_descriptor_train']['similarity'],
                                     'scope':'Model dispersion and applicability hints, not calibrated error probability'},
                      'missing_evidence':unknowns})
    pairs=[]
    for i,j in itertools.combinations(range(len(candidates)),2):
        proof=edit_proof(candidates[i]['smiles'],candidates[j]['smiles'])
        delta=scores[j]-scores[i]
        odt_a=candidates[i].get('odt_ug_per_l_air');odt_b=candidates[j].get('odt_ug_per_l_air')
        pairs.append({'a':i,'b':j,'names':[candidates[i]['name'],candidates[j]['name']],
                      'tanimoto':DataStructs.TanimotoSimilarity(fps[i],fps[j]),'edit_proof':proof,
                      'predicted_descriptor_mean_abs_delta':float(np.abs(delta).mean()),
                      'top_predicted_changes':sorted([{'label':k,'delta':float(v)} for k,v in zip(labels,delta)],key=lambda r:-abs(r['delta']))[:5],
                      'human_odt_fold_difference':max(odt_a,odt_b)/min(odt_a,odt_b) if odt_a and odt_b and candidates[i].get('gas_source')==candidates[j].get('gas_source') else None,
                      'odt_scope':'Same published study, mass concentration units; not equal-dose intensity or surface response',
                      'delivery_supported':all(cards[k]['gate']=='literature_supported_delivery_candidate' for k in (i,j))})
    matched=[p for p in pairs if p['edit_proof'] and p['delivery_supported']]
    frontier=pareto_indices([[p['tanimoto'],p['predicted_descriptor_mean_abs_delta']] for p in matched])
    selected=[];decisions=[]
    # No user-family preference is used. Deterministic tie-break: structural proximity first.
    if frontier:
        anchor=max([matched[i] for i in frontier],key=lambda p:(p['tanimoto'],p['predicted_descriptor_mean_abs_delta']))
        selected=[anchor['a'],anchor['b']]
        decisions.append({'selected_pair':anchor['names'],'reason':'Verified edit pair with headspace evidence; Pareto frontier of structural similarity and predicted descriptor change.'})
    eligible=[i for i,c in enumerate(cards) if c['gate']=='literature_supported_delivery_candidate']
    while len(selected)<budget:
        remaining=[i for i in eligible if i not in selected]
        if not remaining:break
        def diversity(i):return min(1-DataStructs.TanimotoSimilarity(fps[i],fps[j]) for j in selected) if selected else 1.
        i=max(remaining,key=lambda i:(diversity(i),cards[i]['name']))
        decisions.append({'selected':cards[i]['name'],'reason':'Maximin Morgan structural diversity among remaining delivery-supported candidates; cross-chemistry control.',
                          'min_structural_distance_to_selected':diversity(i)})
        selected.append(i)
    return {'scope':'Evidence-gated proposal from a curated pool; not globally optimal or prospectively validated.',
            'rules':{'budget':budget,'gas_gate':'Published headspace source required for primary shortlist; local gas-delivery readiness remains false.',
                     'pair_frontier':'Maximize verified-pair Tanimoto and predicted mean absolute descriptor change without weighted sum.',
                     'anchor_tie_break':'Highest Tanimoto, then model delta; heuristic for controlled structure comparison.',
                     'controls':'Greedy maximin structural diversity; no disease-count reward.',
                     'uncertainty':'Reported as an evidence gap, not converted into confidence or expected information gain.'},
            'sources':sources,'cards':cards,'pairs':pairs,'eligible_edit_frontier':[matched[i] for i in frontier],
            'selected':[cards[i]['name'] for i in selected],'decisions':decisions,
            'deferred':[c['name'] for c in cards if c['gate']=='delivery_evidence_missing'],
            'budget_unfilled':max(0,budget-len(selected)),
            'next_experiments':['Verify supplied gas concentration before comparing surface responses.',
                                'Compare Au-MBA and Au-MPY using independent substrates, blank and baseline measurements.',
                                'Only add spatial modeling after testing mean-spectrum and unordered-spectrum baselines.']}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--candidates',type=Path,required=True)
    ap.add_argument('--model',type=Path,required=True);ap.add_argument('--clinical',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--budget',type=int,default=4)
    a=ap.parse_args();config=json.loads(a.candidates.read_text(encoding='utf-8'))
    predictor=OdorPredictor(a.model)
    predictions=predictor.predict_many([c['smiles'] for c in config['candidates']])
    report=design(config['candidates'],predictions,config['sources'],DiseaseBridge(a.clinical),a.budget)
    # Selection should not depend on arbitrary input ordering for this unique-valued pool.
    reverse=design(list(reversed(config['candidates'])),list(reversed(predictions)),config['sources'],DiseaseBridge(a.clinical),a.budget)
    assert set(report['selected'])==set(reverse['selected'])
    report['checks']={'reversed_input_same_panel':True,'no_unknown_delivery_in_shortlist':all(c['gate']=='literature_supported_delivery_candidate' for c in report['cards'] if c['name'] in report['selected'])}
    report['model']='Existing desktop Logistic ensemble/ExtraTrees; not the separately benchmarked POM MLP'
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/'panel.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 실험 후보 선정 결과','',f"선정: {', '.join(report['selected'])}",'',
           '10개 시작 후보를 같은 규칙으로 평가했다. 인돌계열을 우대하는 점수는 넣지 않았다. 문헌에서 기체가 확인된 후보라도 우리 실험의 전달량과 센서 성능은 아직 검증되지 않았다.',
           '', '| 후보 | 제안 상태 | 질병 원문 주석 수 (순위에 미사용) | 관측 향 주석 |','|---|---|---:|---|']
    for c in report['cards']:
        lines.append(f"| {c['name']} | {'전달량 검증 우선 후보' if c['name'] in report['selected'] else '전달 근거 보완/추가 후보'} | {len(c['clinical']['diseases'])} | {'있음' if c['prediction']['observed_annotations'] else '현재 자료에 없음'} |")
    lines+=['','## 선정 논리','']
    for d in report['decisions']:lines.append('- '+json.dumps(d,ensure_ascii=False))
    lines+=['','## 구조 편집 확인 + 기체 근거를 모두 만족하는 비교쌍','']
    for p in report['eligible_edit_frontier']:
        lines.append(f"- {' / '.join(p['names'])}: {p['edit_proof']['substituent']} 치환 확인, Tanimoto {p['tanimoto']:.3f}, 모델 평균 |Δ descriptor| {p['predicted_descriptor_mean_abs_delta']:.4f}. 같은 문헌의 질량 농도 기준 후각 검출역치 차이 {p['human_odt_fold_difference']}배.")
    lines+=['','## 해석','',
            '후각 검출역치 차이는 기존 문헌의 사람 평가이며 새 모델의 정확도 검증이나 MBA/MPY 표면 차이가 아니다. 예측 향 변화·모델 산포·질병 근거·기체 근거를 별도 필드로 보존했다.',
            '보류는 비휘발성 확정이 아니라 현재 근거 장부에 전달 근거가 없다는 뜻이다. 후보 선정 결과는 시작 풀과 문헌 확인 범위에 의존한다. Phenol 등 보류 후보의 headspace 근거를 추가하면 패널이 달라질 수 있다.',
            '현재 0개 후보가 실제 Au–SAM 기체 실험 전달량을 검증받았다. 선정된 4개는 우선 검증 패널이며 검증 완료 패널이 아니다. 실제 기체 반응이나 질병 분류기를 학습하지 않았다.']
    (a.out/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'selected':report['selected'],'deferred':report['deferred'],'eligible_edit_pairs':report['eligible_edit_frontier'],'checks':report['checks']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
