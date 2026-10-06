"""Post-hoc candidate evidence in original embedding space; no 2D distance used."""
import argparse,json,hashlib,csv
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--annotations',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
c=np.load(a.cache,allow_pickle=False);d=json.loads(a.annotations.read_text(encoding='utf-8-sig'));smiles=c['smiles'].tolist();X=c['embeddings'].astype(float)
if X.ndim!=2 or X.shape[0]!=len(smiles) or not np.isfinite(X).all() or len(set(smiles))!=len(smiles):raise ValueError('Invalid/duplicate cache rows')
norm=np.linalg.norm(X,axis=1,keepdims=True)
if (norm==0).any():raise ValueError('Zero embedding')
X=X/norm;dist=np.clip(1-X@X.T,0,2)
i,j=[smiles.index(r['smiles']) for r in d['focus'][:2]]
if i==j:raise ValueError('Distinct candidates required')
ranks=[]
for q,t in [(i,j),(j,i)]:
 other=np.arange(len(X))!=q
 # Competition rank; include all ties in the neighbor fraction.
 ranks.append({'query':1 if q==i else 2,'target':2 if q==i else 1,'rank':int(1+np.sum(dist[q,other]<dist[q,t])),'candidates':int(other.sum()),'fraction_at_or_closer_pct':float(100*np.mean(dist[q,other]<=dist[q,t]))})
pairs=dist[np.triu_indices(len(X),1)]
r={'analysis':'post-hoc support, not a prospectively validated selection algorithm','dimension':X.shape[1],'molecules':len(X),'pair_cosine_similarity':float(1-dist[i,j]),'pair_cosine_distance':float(dist[i,j]),'neighbor_ranks':ranks,'all_pairs_count':len(pairs),'fraction_pairs_at_or_closer_pct':float(100*np.mean(pairs<=dist[i,j])),'median_all_pair_distance':float(np.median(pairs)),'cache_sha256':hashlib.sha256(a.cache.read_bytes()).hexdigest(),'annotation_sha256':hashlib.sha256(a.annotations.read_bytes()).hexdigest(),'interpretation':'Similarity is relative to this reference library, not perceptual equivalence, a p-value, or diagnostic evidence.'}
a.out.mkdir(parents=True,exist_ok=True);(a.out/'candidate_evidence.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
with (a.out/'candidate_evidence.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.writer(f);w.writerow(['metric','value']);w.writerow(['cosine_similarity',r['pair_cosine_similarity']]);w.writerow(['cosine_distance',r['pair_cosine_distance']]);w.writerow(['pair_distance_percentile',r['fraction_pairs_at_or_closer_pct']]);w.writerow(['1_to_2_rank',ranks[0]['rank']]);w.writerow(['2_to_1_rank',ranks[1]['rank']])
print(json.dumps(r,indent=2))
