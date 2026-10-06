"""Run on the target Mac to verify imports, native graph execution and checkpoint parity."""
import argparse,json,platform,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--config',type=Path,default=Path('examples/openpom_checkpoint_config.json'));p.add_argument('--reference',type=Path,default=Path('examples/pom_cpu_reference.json'));p.add_argument('--out',type=Path,default=Path('runs/runtime-check'));p.add_argument('--gui',action='store_true');a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    report={'system':platform.system(),'machine':platform.machine(),'python':sys.version,'apple_silicon_native':platform.system()=='Darwin' and platform.machine()=='arm64','checks':{},'passed':False}
    exitcode=0
    try:
        import numpy as np
        import torch,dgl,deepchem,rdkit
        from voclib.pom_backend import POMEncoder
        torch.set_num_threads(2)
        report['versions']={'torch':torch.__version__,'dgl':dgl.__version__,'deepchem':deepchem.__version__,'rdkit':rdkit.__version__,'numpy':np.__version__}
        report['mps_available']=torch.backends.mps.is_available()
        report['pom_device']='cpu'
        if platform.system()=='Darwin' and platform.machine()!='arm64':
            raise RuntimeError('Use native ARM64 Python; this interpreter is not Apple Silicon native.')
        ref=json.loads(a.reference.read_text(encoding='utf-8'))
        encoder=POMEncoder(a.checkpoint,a.config,a.out/'encoder-runtime')
        z,scores=encoder.encode(ref['smiles'])
        expected_z=np.asarray(ref['embeddings']);expected_p=np.asarray(ref['descriptor_scores'])
        report['embedding_max_abs_difference']=float(np.max(np.abs(z-expected_z)))
        report['descriptor_max_abs_difference']=float(np.max(np.abs(scores-expected_p)))
        np.testing.assert_allclose(z,expected_z,atol=1e-3,rtol=1e-3)
        np.testing.assert_allclose(scores,expected_p,atol=1e-4,rtol=1e-3)
        report['checks']['checkpoint_forward_cpu_parity']=True
        # Check gradient/update/save/load for the same layer types as the pilot head.
        from run_pom_mlp_pilot import RatingHead
        torch.manual_seed(42);head=RatingHead(260);head.train();x=torch.randn(4,260);y=torch.full((4,3),.5)
        opt=torch.optim.AdamW(head.parameters(),lr=.001);opt.zero_grad();loss=torch.nn.functional.mse_loss(head(x),y);loss.backward()
        assert all(torch.isfinite(v.grad).all() for v in head.parameters() if v.grad is not None)
        opt.step();head.eval();torch.save(head.state_dict(),a.out/'smoke_head.pt')
        restored=RatingHead(260);restored.load_state_dict(torch.load(a.out/'smoke_head.pt',map_location='cpu',weights_only=True));restored.eval()
        with torch.no_grad():torch.testing.assert_close(head(x),restored(x))
        report['checks']['mlp_backward_and_reload']=True
        if a.gui:
            import tkinter as tk
            from PIL import ImageTk
            from rdkit.Chem import Draw
            from voclib.odor_edit import molecule
            root=tk.Tk();root.withdraw();pic=ImageTk.PhotoImage(Draw.MolToImage(molecule(ref['smiles'][0])));label=tk.Label(root,image=pic);label.pack();root.update_idletasks();root.destroy()
            report['checks']['tkinter_rdkit_image']=True
        report['passed']=True
    except Exception as exc:
        report['error']=f'{type(exc).__name__}: {exc}';exitcode=1
    (a.out/'runtime_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2));raise SystemExit(exitcode)

if __name__=='__main__':main()
