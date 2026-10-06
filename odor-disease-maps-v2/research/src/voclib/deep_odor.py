"""Frozen OpenPOM graph encoder + trained PyTorch perceptual heads.

Uses the same public encoder checkpoint as the internal CV benchmark. Descriptor
scores are pretrained outputs, while ratings are locally learned population means.
"""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from rdkit import Chem, DataStructs
from .odor_edit import OdorPredictor, molecule, rating_features, FP, RATINGS


class RatingHead(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(dim,128),nn.LayerNorm(128),nn.ReLU(),nn.Dropout(.15),
                               nn.Linear(128,64),nn.ReLU(),nn.Linear(64,3),nn.Sigmoid())
    def forward(self,x):return self.net(x)


class DeepOdorPredictor(OdorPredictor):
    def __init__(self, model_path):
        self.root=Path(model_path).resolve().parent
        meta=json.loads(Path(model_path).read_text(encoding='utf-8'));self.meta=meta
        self.bundle={'labels':meta['encoder']['labels'],'report':{'backend':'pom_mlp',**meta['benchmark']}}
        cache=np.load(self.root/'inference_cache.npz',allow_pickle=False)
        self.cache={s:(z,p) for s,z,p in zip(cache['smiles'],cache['embeddings'],cache['descriptor_scores'])}
        self.heads=[]
        for filename in meta['heads']:
            state=torch.load(self.root/filename,map_location='cpu',weights_only=True)
            net=RatingHead(state['input_dim']);net.load_state_dict(state['state']);net.eval()
            self.heads.append((net,np.array(state['mean']),np.array(state['scale'])))
        self.refs=meta['training_smiles'];self.fps=[FP.GetFingerprint(molecule(s)) for s in self.refs]
        self.encoder=None

    def predict_many(self,smiles,concentration=.001):
        canonical=[Chem.MolToSmiles(molecule(s)) for s in smiles]
        missing=sorted(set(canonical)-self.cache.keys())
        if missing:
            if self.encoder is None:
                from .pom_backend import POMEncoder
                self.encoder=POMEncoder(self.meta['checkpoint_path'],self.root/'encoder_config.json',self.root/'encoder-runtime')
                if self.encoder.config!=self.meta['encoder']:raise ValueError('Encoder metadata mismatch')
            z,p=self.encoder.encode(missing)
            self.cache.update({s:(zz,pp) for s,zz,pp in zip(missing,z,p)})
        mols=[molecule(s) for s in canonical]
        x=np.column_stack([np.array([self.cache[s][0] for s in canonical]),
                           np.array([rating_features(m,concentration)[-4:] for m in mols])])
        with torch.no_grad():
            samples=np.array([net(torch.tensor((x-mean)/scale,dtype=torch.float32)).numpy() for net,mean,scale in self.heads])
        mean=samples.mean(0);sd=samples.std(0);result=[]
        for i,(s,m) in enumerate(zip(canonical,mols)):
            sims=DataStructs.BulkTanimotoSimilarity(FP.GetFingerprint(m),self.fps);near=int(np.argmax(sims))
            observed=self.meta['observed_annotations'].get(Chem.MolToInchiKey(m),[])
            result.append({'smiles':s,'concentration':concentration,
                'descriptors':{label:{'score':float(score),'bootstrap_sd':None,'selected':False} for label,score in zip(self.bundle['labels'],self.cache[s][1])},
                'ratings':{k:{'score':float(mean[i,j]),'tree_sd':float(sd[i,j])} for j,k in enumerate(RATINGS)},
                'observed_annotations':[{'labels':observed,'split':'external atlas annotation'}] if observed else [],
                'nearest_descriptor_train':{'smiles':self.refs[near],'similarity':float(sims[near])},
                'nearest_rating_train_similarity':float(sims[near]),
                'similarity_scope':'Downstream rating training structures; POM pretraining membership unknown',
                'evidence':'Frozen pretrained OpenPOM descriptor scores + locally trained 3-seed MLP mean ratings; not experimental gas observations',
                'dispersion_scope':'Ratings field tree_sd contains MLP seed dispersion for legacy UI compatibility; descriptor uncertainty unavailable'})
        return result
