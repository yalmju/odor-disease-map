import os
os.environ.setdefault("DGLBACKEND","pytorch")
import argparse
from pathlib import Path
import numpy as np
from voclib.pom_backend import POMEncoder
p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--smiles-file',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
a.out.parent.mkdir(parents=True,exist_ok=True)
smiles=[s.strip() for s in a.smiles_file.read_text().splitlines() if s.strip()]
model=POMEncoder(a.checkpoint,a.config,a.out.parent/'encoder-runtime')
z,scores=model.encode(smiles)
np.savez_compressed(a.out,smiles=np.array(smiles),embeddings=z,descriptor_scores=scores)
print(z.shape,scores.shape)
