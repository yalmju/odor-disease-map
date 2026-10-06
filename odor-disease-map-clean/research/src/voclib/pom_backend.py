"""Optional OpenPOM adapter. Requires the separately pinned OpenPOM environment."""
import hashlib
import json
from pathlib import Path
import numpy as np

class POMEncoder:
    def __init__(self, checkpoint, config, model_dir):
        import torch
        from openpom.models.mpnn_pom import MPNNPOMModel
        from openpom.feat.graph_featurizer import GraphFeaturizer, GraphConvConstants
        self.config=json.loads(Path(config).read_text(encoding='utf-8'))
        digest=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
        if digest != self.config['checkpoint_sha256']:
            raise ValueError('Checkpoint hash mismatch')
        self.featurizer=GraphFeaturizer()
        self.model=MPNNPOMModel(n_tasks=len(self.config['labels']), batch_size=128,
            class_imbalance_ratio=None,loss_aggr_type='sum',node_out_feats=100,
            edge_hidden_feats=75,edge_out_feats=100,num_step_message_passing=5,
            mpnn_residual=True,message_aggregator_type='sum',mode='classification',
            number_atom_features=GraphConvConstants.ATOM_FDIM,
            number_bond_features=GraphConvConstants.BOND_FDIM,n_classes=1,
            readout_type='set2set',num_step_set2set=3,num_layer_set2set=2,
            ffn_hidden_list=[392,392],ffn_embeddings=256,ffn_activation='relu',
            ffn_dropout_p=.12,ffn_dropout_at_input_no_act=False,weight_decay=1e-5,
            self_loop=False,optimizer_name='adam',log_frequency=32,
            model_dir=str(model_dir),device_name='cpu')
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        self.model.model.load_state_dict(state['model_state_dict'],strict=True)
        self.model.model.eval()

    def encode(self, smiles):
        import deepchem as dc
        from rdkit import Chem
        if not smiles:raise ValueError('Empty molecule list')
        for s in smiles:
            m=Chem.MolFromSmiles(s)
            if m is None or not m.GetNumAtoms() or len(Chem.GetMolFrags(m))!=1:
                raise ValueError('Expected valid single-component SMILES: '+str(s))
        graphs=self.featurizer.featurize(list(smiles))
        if len(graphs)!=len(smiles) or any(not hasattr(g,'node_features') for g in graphs):
            raise ValueError('Graph featurization failed')
        dataset=dc.data.NumpyDataset(graphs)
        # One fixed checkpoint: hidden coordinates are never averaged across models.
        z=np.asarray(self.model.predict_embedding(dataset),dtype=np.float32)
        scores=np.asarray(self.model.predict(dataset),dtype=np.float32)
        if z.shape!=(len(smiles),self.config['embedding_dim']) or not np.isfinite(z).all():
            raise ValueError('Unexpected/nonfinite POM embeddings')
        if scores.shape!=(len(smiles),len(self.config['labels'])) or not np.isfinite(scores).all():
            raise ValueError('Unexpected/nonfinite POM scores')
        return z,scores
