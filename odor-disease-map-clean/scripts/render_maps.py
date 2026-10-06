import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from scipy.spatial.distance import cdist
import argparse
import hashlib
import platform
import importlib.metadata
from validate_data import validate
parser=argparse.ArgumentParser(description='Render fixed molecular coordinates with odor and disease annotations.')
parser.add_argument('--input',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--disease',default='Colorectal cancer')
parser.add_argument('--font',default='DejaVu Sans',help='Use Arial to match the poster if installed.')
args=parser.parse_args()
O=args.output; O.mkdir(parents=True,exist_ok=True)
d=json.loads(args.input.read_text(encoding='utf-8-sig'));validate(d)
xy=np.array([[p['x'],p['y']] for p in d['points']])
labs=['fruity','floral','green','woody','roasted','animal']
cols=['#D5BC35','#C976A6','#83B951','#967CDA','#639BCD','#DE837D']
crc='#BA497E';bg='#B9C1CB';ink='#24303B'
plt.rcParams.update({'font.family':args.font,'svg.fonttype':'none','pdf.fonttype':42,'text.color':ink})
bw=2.6;lo=xy.min(0)-5*bw;hi=xy.max(0)+5*bw;extent=(lo[0],hi[0],lo[1],hi[1]);X,Y=np.meshgrid(np.linspace(lo[0],hi[0],540),np.linspace(lo[1],hi[1],420));grid=np.c_[X.ravel(),Y.ravel()]
fields=[]
for lab in labs:
 mask=np.array([lab in p['odors'] for p in d['points']]);z=np.exp(-cdist(grid,xy[mask],'sqeuclidean')/(2*bw*bw)).sum(1).reshape(X.shape);fields.append((mask,z/z.max() if z.max()>0 else z))
cmask=np.array([args.disease in p['diseases'] for p in d['points']])
# Coordinates and membership are reused exactly; no embedding or association recomputation.

import zipfile

def focus(ax,size=15):
 for i,p in enumerate(d['focus'][:2]):
  ax.scatter(p['x'],p['y'],s=28,facecolors='none',edgecolors=ink,lw=1.3,zorder=6)
  ax.annotate(str(i+1),(p['x'],p['y']),xytext=((-24,-29) if i==0 else (25,24)),textcoords='offset points',fontsize=size,fontweight="bold",color=ink,ha='center',va='center',zorder=8,arrowprops=dict(arrowstyle='-|>',color=ink,lw=1.3,mutation_scale=11,shrinkA=3,shrinkB=4))
def finish(ax):
 ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),aspect='equal');ax.axis('off')
def odor(ax,j):
 mask,z=fields[j];col=cols[j];ax.scatter(*xy.T,s=7,c=bg,lw=0)
 layer=np.ones((*z.shape,4));layer[...,:3]=to_rgb(col);layer[...,3]=.97*z**.65
 ax.imshow(layer,origin='lower',extent=extent,interpolation='bilinear');ax.scatter(*xy[mask].T,s=9,c=col,alpha=.75,lw=0)
 focus(ax);finish(ax)
def disease(ax):
 ax.scatter(*xy.T,s=12,c=bg,lw=0)
 ax.scatter(*xy[cmask].T,s=30,c=crc,alpha=1,lw=0)
 for i,p in enumerate(d['focus'][:2]):
  col=crc if args.disease in p['diseases'] else bg
  ax.scatter(p['x'],p['y'],s=185,c=col,edgecolors='white',linewidths=3,zorder=8)
  ax.scatter(p['x'],p['y'],s=185,facecolors='none',edgecolors=ink,linewidths=1.8,zorder=9)
  ax.annotate(str(i+1),(p['x'],p['y']),xytext=((-32,-38) if i==0 else (34,34)),textcoords='offset points',fontsize=23,fontweight='bold',color=ink,ha='center',va='center',zorder=10,arrowprops=dict(arrowstyle='-|>',color=ink,lw=1.8,mutation_scale=14,shrinkA=4,shrinkB=9))
 finish(ax)

def save(fig,name):
 for ext in ['png','svg','pdf']:fig.savefig(O/(name+'.'+ext),dpi=300,facecolor='white',bbox_inches='tight',pad_inches=.025)
 plt.close(fig)
for j,lab in enumerate(labs):
 fig,ax=plt.subplots(figsize=(6,5.5));odor(ax,j);fig.subplots_adjust(left=0,right=1,bottom=0,top=1);save(fig,f'{j+1:02d}_{lab}_arrows')
fig,axs=plt.subplots(2,3,figsize=(12,8))
for j,ax in enumerate(axs.flat):odor(ax,j)
fig.subplots_adjust(left=.005,right=.995,bottom=.005,top=.995,wspace=.01,hspace=.02);save(fig,'07_odor_six_panels_arrows')
fig,ax=plt.subplots(figsize=(7,7));disease(ax);fig.subplots_adjust(left=0,right=1,bottom=0,top=1);save(fig,'08_disease_arrows')
fig=plt.figure(figsize=(18,8));gs=fig.add_gridspec(2,5,left=.005,right=.995,bottom=.005,top=.995,wspace=.04,hspace=.02,width_ratios=[1,1,1,.78,.78])
for j in range(6):odor(fig.add_subplot(gs[j//3,j%3]),j)
disease(fig.add_subplot(gs[:,3:]));save(fig,'09_combined_arrows')
manifest={'input_sha256':hashlib.sha256(args.input.read_bytes()).hexdigest(),
 'point_count':len(xy),'disease':args.disease,'disease_count':int(cmask.sum()),
 'odor_counts':{lab:int(mask.sum()) for lab,(mask,z) in zip(labs,fields)},
 'coordinate_source':d.get('coordinate_source','user-supplied fixed coordinates'),
 'synthetic':d.get('synthetic',False),'bandwidth':bw,'density_exponent':.65,
 'font_requested':args.font,'python':platform.python_version(),
 'dependencies':{k:importlib.metadata.version(k) for k in ['numpy','scipy','matplotlib']}}
(O/'render_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print('Rendered 9 figures as PNG/SVG/PDF. Manifest: '+str(O/'render_manifest.json'))
