"""Real frozen OpenPOM -> PyTorch MLP pilot, compared with Morgan -> same MLP.

Uses an existing molecule-grouped split manifest. Does not retrain OpenPOM.
POM pretraining overlap is not excluded: downstream holdout is not full-pipeline
independent validation. Test predictions are evaluated once after validation.
"""
import argparse,hashlib,json,platform,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from voclib.odor_edit import molecule,features,rating_features,RATINGS,enumerate_edits
from voclib.pom_backend import POMEncoder

class RatingHead(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(dim,128),nn.LayerNorm(128),nn.ReLU(),nn.Dropout(.15),
                               nn.Linear(128,64),nn.ReLU(),nn.Linear(64,3),nn.Sigmoid())
    def forward(self,x):return self.net(x)

def train_head(x,y,indices,out,seed=42):
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    scaler=StandardScaler().fit(x[indices['train']]);xx=scaler.transform(x).astype(np.float32)
    net=RatingHead(xx.shape[1]);optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.001)
    tensor=torch.from_numpy(xx);target=torch.tensor(y,dtype=torch.float32)
    history=[];best=float('inf');best_state=None;best_epoch=0;stale=0
    for epoch in range(1,201):
        net.train();total=0
        for idx in np.array_split(rng.permutation(indices['train']),max(1,int(np.ceil(len(indices['train'])/64)))):
            optimizer.zero_grad();loss=nn.functional.mse_loss(net(tensor[idx]),target[idx]);loss.backward();optimizer.step();total+=float(loss.detach())*len(idx)
        net.eval()
        with torch.no_grad():val=float(nn.functional.mse_loss(net(tensor[indices['val']]),target[indices['val']]))
        history.append({'epoch':epoch,'train_mse':total/len(indices['train']),'val_mse':val})
        if val<best-1e-6:
            best=val;best_epoch=epoch;best_state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()};stale=0
        else:stale+=1
        if stale>=25:break
    net.load_state_dict(best_state);net.eval()
    with torch.no_grad():pred=net(tensor).numpy()
    bundle={'state':best_state,'input_dim':xx.shape[1],'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),'best_epoch':best_epoch,'seed':seed,'ratings':RATINGS,'architecture':'128-LayerNorm-ReLU-Dropout0.15-64-ReLU-3-Sigmoid','evidence':'learned population mean under Keller dilution conditions'}
    torch.save(bundle,out)
    loaded=torch.load(out,map_location='cpu',weights_only=True);restored=RatingHead(loaded['input_dim']);restored.load_state_dict(loaded['state']);restored.eval()
    with torch.no_grad():np.testing.assert_allclose(restored(tensor).numpy(),pred,atol=1e-7)
    return net,scaler,pred,history,bundle

def evaluate(y,p,rows):
    result={}
    for condition in ['all',.001,.00001]:
        test=np.array([i for i,r in enumerate(rows) if r['split']=='test' and (condition=='all' or r['concentration']==condition)])
        metrics={}
        for j,name in enumerate(RATINGS):
            baseline=np.array([np.mean([r['y'][j] for r in rows if r['split']=='train' and r['concentration']==rows[i]['concentration']]) for i in test])
            metrics[name]={'mae':float(mean_absolute_error(y[test,j],p[test,j])),
               'condition_mean_baseline_mae':float(mean_absolute_error(y[test,j],baseline)),
               'pearson_r':float(np.corrcoef(y[test,j],p[test,j])[0,1]) if np.std(p[test,j])>0 else None}
        result[str(condition)]={'n':len(test),'metrics':metrics}
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',type=Path,required=True);parser.add_argument('--config',type=Path,required=True);parser.add_argument('--split',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);a=parser.parse_args()
    if a.out.exists() and any(a.out.iterdir()):raise ValueError('Use a new, empty output directory')
    a.out.mkdir(parents=True);torch.set_num_threads(2);started=time.time()
    manifest=json.loads(a.split.read_text(encoding='utf-8'));rows=manifest['ratings']
    indices={s:np.array([i for i,r in enumerate(rows) if r['split']==s]) for s in ['train','val','test']}
    sets={s:{rows[i]['key'] for i in idx} for s,idx in indices.items()}
    assert not(sets['train']&sets['test'] or sets['train']&sets['val'] or sets['val']&sets['test'])
    examples={'Indole':'c1ccc2[nH]ccc2c1','Skatole':'Cc1c[nH]c2ccccc12','IAA':'O=C(O)Cc1c[nH]c2ccccc12','IPA':'O=C(O)CCc1c[nH]c2ccccc12','ILA':'O=C(O)[C@@H](O)Cc1c[nH]c2ccccc12','Indoxyl sulfate':'O=S(=O)(O)Oc1c[nH]c2ccccc12'}
    edits=enumerate_edits(examples['Indole'])
    smiles=sorted({r['smiles'] for r in rows}|set(examples.values())|{r['smiles'] for r in edits});lookup={s:i for i,s in enumerate(smiles)}
    encoder=POMEncoder(a.checkpoint,a.config,a.out/'encoder-runtime')
    print(f'Encoding {len(smiles)} distinct SMILES with one frozen OpenPOM checkpoint...',flush=True)
    z,scores=encoder.encode(smiles)
    np.savez_compressed(a.out/'embeddings.npz',smiles=np.asarray(smiles),embeddings=z,descriptor_scores=scores)
    (a.out/'encoder_metadata.json').write_text(json.dumps(encoder.config,indent=2),encoding='utf-8')
    print(f'Embedding shape {z.shape}; descriptor shape {scores.shape}',flush=True)
    condition=np.array([rating_features(molecule(r['smiles']),r['concentration'])[-4:] for r in rows])
    inputs={'pom':np.column_stack([z[[lookup[r['smiles']] for r in rows]],condition]),
            'morgan':np.column_stack([np.array([features(molecule(r['smiles'])) for r in rows]),condition])}
    y=np.array([r['y'] for r in rows]);report={'scope':'Downstream molecule-grouped pilot; POM pretraining overlap not excluded. Single seed, fixed pretrained checkpoint, not full-pipeline independent validation.',
      'rows':len(rows),'split_counts':{s:len(i) for s,i in indices.items()},'encoder':encoder.config,
      'split_sha256':hashlib.sha256(a.split.read_bytes()).hexdigest(),'models':{},'checks':{'group_disjoint':True},
      'environment':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__},
      'conditions':'Paraffin oil 1/1000 or 1/100000; not gas-phase response.'}
    trained={}
    for name,x in inputs.items():
        print(f'Training {name} -> MLP...',flush=True)
        net,scaler,pred,history,bundle=train_head(x,y,indices,a.out/(name+'_rating_head.pt'))
        trained[name]=(net,scaler);report['models'][name]={'test':evaluate(y,pred,rows),'best_epoch':bundle['best_epoch'],'history':history,'input_dim':x.shape[1]}
        (a.out/(name+'_test_predictions.json')).write_text(json.dumps([{'smiles':rows[i]['smiles'],'concentration':rows[i]['concentration'],'observed':y[i].tolist(),'predicted':pred[i].tolist()} for i in indices['test']],indent=2),encoding='utf-8')
        print(name,json.dumps(report['models'][name]['test']['all']),flush=True)
    net,scaler=trained['pom']
    example_inputs=np.column_stack([z,np.array([rating_features(molecule(s),.001)[-4:] for s in smiles])])
    with torch.no_grad():rp=net(torch.tensor(scaler.transform(example_inputs),dtype=torch.float32)).numpy()
    def record(s):
        i=lookup[s]
        return {'smiles':s,'descriptors':dict(zip(encoder.config['labels'],scores[i].tolist())),
                'ratings':dict(zip(RATINGS,rp[i].tolist())),'concentration':.001,
                'evidence':'OpenPOM pretrained scores + newly trained MLP predictions; not experimental observations'}
    reference=[dict(name=name,**record(s)) for name,s in examples.items()];base=record(examples['Indole'])
    scan=[]
    for r in edits:
        result=record(r['smiles']);scan.append(dict(r,prediction=result,
          descriptor_delta={k:result['descriptors'][k]-base['descriptors'][k] for k in base['descriptors']},
          rating_delta={k:result['ratings'][k]-base['ratings'][k] for k in RATINGS}))
    (a.out/'indole_predictions.json').write_text(json.dumps(reference,indent=2),encoding='utf-8')
    (a.out/'indole_scan.json').write_text(json.dumps({'base':base,'candidates':scan},indent=2),encoding='utf-8')
    # Check graph input batching and serialization, not prediction quality.
    zz,pp=encoder.encode([examples['Skatole'],examples['Indole'],examples['Indole']])
    np.testing.assert_allclose(zz[1],zz[2],atol=1e-5);np.testing.assert_allclose(zz[1],z[lookup[examples['Indole']]],atol=1e-4,rtol=1e-4)
    np.testing.assert_allclose(pp[0],scores[lookup[examples['Skatole']]],atol=1e-5,rtol=1e-4)
    assert all(np.isfinite(rp).flat) and len(scan)==56
    report['checks'].update(head_save_load_roundtrip=True,embedding_batch_order=True,indole_candidates=56)
    report['elapsed_seconds']=time.time()-started
    (a.out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    lines=['# OpenPOM → MLP 실제 연결 시험','','## 실행 결과','','- 공개 가중치 해시 검증 후 weights_only 로드 성공.','- 256차원 임베딩 및 138개 향 점수 출력 성공.','- POM 임베딩+조건 → 작은 MLP로 사람의 지각 평가 3개 학습·저장·재로드 성공.','- 같은 downstream 분할과 동일한 hidden 구조로 Morgan+MLP 비교.','- indole 계열 6개 예시 및 치환 후보 56개 예측 저장.','','## Downstream 시험 MAE (낮을수록 좋음)','','| 항목 | POM + MLP | Morgan + MLP | 농도별 평균 baseline |','|---|---:|---:|---:|']
    for label in RATINGS:
        pm=report['models']['pom']['test']['all']['metrics'][label];mm=report['models']['morgan']['test']['all']['metrics'][label]
        lines.append(f'| {label} | {pm["mae"]:.4f} | {mm["mae"]:.4f} | {pm["condition_mean_baseline_mae"]:.4f} |')
    lines+=['','## 해석 범위','','이 결과는 실제 코드 연결과 새 회귀 head의 학습 가능성을 확인한 pilot이다. POM 사전학습과 downstream 시험 분자의 중복을 제외하지 않았으므로 완전한 외부 검증, 인돌 신규 치환기의 검증, 방법 우월성을 주장하지 않는다. 단일 seed 결과이며 원천 데이터의 관측 불확실성도 남아 있다.','', '현재 기존 데스크톱 앱의 활성 예측기는 바꾸지 않았다. 새 backend/임베딩 캐시/MLP head는 별도 파일로 저장되어 있다. NNLS는 이 경로에서 사용하지 않는다.','','## 출력','','- embeddings.npz: 문자열 SMILES, 256차원 벡터, 138개 pretrained descriptor 점수.','- pom_rating_head.pt / morgan_rating_head.pt: 학습된 회귀 head와 스케일러·구조 메타데이터.','- report.json: 성능, 분할 해시, 출처, 학습 이력, 실행 검사.','- indole_predictions.json / indole_scan.json: 인돌 예시 및 56개 치환 결과.']
    (a.out/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print('POM_MLP_PILOT_COMPLETE',flush=True)

if __name__=='__main__':main()
