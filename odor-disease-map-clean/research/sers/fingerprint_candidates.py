import numpy as np, csv, sys
from pathlib import Path
from scipy.ndimage import minimum_filter1d, uniform_filter1d
from scipy.signal import find_peaks
from scipy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ROOT=Path(sys.argv[1]); OUT=Path(sys.argv[2]); OUT.mkdir(exist_ok=True,parents=True)
CH={'pristine':None,'mba':1078,'nat':1064,'mpy':1095}; AN=['blank','ind','skt','mix']
def load(ch,an):
    a=np.loadtxt(ROOT/f'{ch}_{an}.txt'); w=a[:,0]; Y=a[:,1:].T
    B=np.array([y-uniform_filter1d(minimum_filter1d(y,151),151) for y in Y]); B=np.array([np.convolve(b,np.ones(5)/5,'same') for b in B]); return w,B
def normalise(w,B,nb):
    if nb is None: return B  # pristine: no reporter band; use baseline-removed counts as measured
    s=(w>=nb-12)&(w<=nb+12); return B/B[:,s].max(1,keepdims=True)
rows=[]
for ch,nb in CH.items():
    D={}; 
    for an in AN:
        w,B=load(ch,an); D[an]=normalise(w,B,nb)
    sel=(w>=600)&(w<=1800); ws=w[sel]
    noise=np.median([D[an][:,(w>=1700)&(w<=1800)].std(1).mean() for an in AN])
    # peak union across group means
    peaks=[]
    for an in AN:
        mu=D[an].mean(0)[sel]; pk,_=find_peaks(mu,prominence=max(3*noise,0.02*mu.max()),distance=4)
        peaks+= [ws[i] for i in pk]
    peaks=sorted(peaks); merged=[]
    for p in peaks:
        if merged and p-merged[-1][-1]<=8: merged[-1].append(p)
        else: merged.append([p])
    centres=[float(np.mean(m)) for m in merged]
    for c in centres:
        s=(w>=c-8)&(w<=c+8); H={an:D[an][:,s].max(1) for an in AN}
        in_blank=H['blank'].mean()>3*noise
        r={'channel':ch,'batch_note':'','band_cm-1':round(c,1),'present_in_blank':'yes' if in_blank else 'no','normalisation':('none (counts)' if nb is None else f'I/I{nb}'),'blank_mean':round(H['blank'].mean(),4),'noise':round(noise,4)}
        for an in ['ind','skt','mix']:
            r[f'{an}_mean']=round(H[an].mean(),4); r[f'{an}_fold_vs_blank']=round(H[an].mean()/max(H['blank'].mean(),1e-9),2); r[f'{an}_p_vs_blank']=f"{stats.mannwhitneyu(H[an],H['blank']).pvalue:.1g}"
        r['p_ind_vs_skt']=f"{stats.mannwhitneyu(H['ind'],H['skt']).pvalue:.1g}"; r['p_mix_vs_ind']=f"{stats.mannwhitneyu(H['mix'],H['ind']).pvalue:.1g}"; r['p_mix_vs_skt']=f"{stats.mannwhitneyu(H['mix'],H['skt']).pvalue:.1g}"
        r['ind_fold_vs_skt']=round(H['ind'].mean()/max(H['skt'].mean(),1e-9),2)
        pi,ps,pm=[stats.mannwhitneyu(H[an],H['blank']).pvalue for an in ['ind','skt','mix']]; pis=stats.mannwhitneyu(H['ind'],H['skt']).pvalue
        di,ds=np.sign(H['ind'].mean()-H['blank'].mean()),np.sign(H['skt'].mean()-H['blank'].mean())
        roles=[]
        if pi<0.01 and ps<0.01 and di==ds: roles.append('presence (ind & skt same direction)')
        if pi<0.01 and ps>=0.05: roles.append('indole-specific')
        if ps<0.01 and pi>=0.05: roles.append('skatole-specific')
        if pi<0.01 and ps<0.01 and di!=ds: roles.append('opposite sign ind/skt')
        if pis<0.001: roles.append('discriminates ind vs skt')
        pmi=stats.mannwhitneyu(H['mix'],H['ind']).pvalue; pms=stats.mannwhitneyu(H['mix'],H['skt']).pvalue
        if pm<0.01 and pmi<0.01 and pms<0.01 and np.sign(H['mix'].mean()-H['ind'].mean())==np.sign(H['mix'].mean()-H['skt'].mean()): roles.append('mixture-specific (differs from both pures)')
        r['role']='; '.join(roles) if roles else ''
        r['score']=round(-np.log10(max(pis,1e-6))+ (1 if 'opposite' in r['role'] else 0),1)
        rows.append(r)
cols=list(rows[0].keys())
with open(OUT/'fingerprint_candidates.csv','w',newline='') as f:
    wr=csv.DictWriter(f,fieldnames=cols); wr.writeheader(); [wr.writerow(r) for r in rows]
# figure: per channel mean spectra with candidate bands marked by role
fig,axes=plt.subplots(len(CH),1,figsize=(14,3.3*len(CH)))
colr={'presence':'#777777','indole-specific':'tab:purple','skatole-specific':'tab:red','opposite':'tab:orange','discriminates':'tab:blue','mixture':'tab:green'}
for ax,(ch,nb) in zip(axes,CH.items()):
    D={an:normalise(*load(ch,an),nb) for an in AN}; w=load(ch,'blank')[0]; sel=(w>=600)&(w<=1800)
    for an,c in (('blank','0.45'),('ind','tab:purple'),('skt','tab:red'),('mix','tab:green')): ax.plot(w[sel],D[an].mean(0)[sel],color=c,lw=1,label=an)
    ymax=max(D[an].mean(0)[sel].max() for an in AN)
    for r in [r for r in rows if r['channel']==ch and r['role']]:
        role=r['role']; key='opposite' if 'opposite' in role else 'discriminates' if 'discriminates' in role else 'indole-specific' if 'indole' in role else 'skatole-specific' if 'skatole' in role else 'mixture' if 'mixture' in role else 'presence'
        ax.axvline(r['band_cm-1'],color=colr[key],alpha=0.5,lw=1); ax.text(r['band_cm-1'],ymax*1.02,f"{r['band_cm-1']:.0f}",rotation=90,fontsize=6.5,color=colr[key],ha='center',va='bottom')
    ax.set_title(f"{ch} (normalised to {nb if nb else 'none: baseline-removed counts'}); marks: grey=presence, purple=indole-specific, red=skatole-specific, orange=opposite sign, blue=discriminates ind/skt, green=mixture-specific",fontsize=8.5)
    ax.legend(fontsize=7,loc='upper right'); ax.set_ylim(top=ymax*1.25)
axes[-1].set_xlabel('Raman shift (cm$^{-1}$)'); fig.tight_layout(); fig.savefig(OUT/'fingerprint_candidates.png',dpi=140)
# console summary: top candidates per channel
for ch in CH:
    rs=[r for r in rows if r['channel']==ch and r['role']]
    print(f'== {ch}: {len([r for r in rows if r["channel"]==ch])} peaks, {len(rs)} with a role')
    print('  -- presence candidates (ind & skt both change vs blank, same direction):')
    for r in [r for r in rs if 'presence' in r['role']][:6]:
        print(f"   {r['band_cm-1']:7.1f}  in_blank={r['present_in_blank']:3}  ind x{r['ind_fold_vs_blank']:5.2f} skt x{r['skt_fold_vs_blank']:5.2f} mix x{r['mix_fold_vs_blank']:5.2f}  p(ind vs blank)={r['ind_p_vs_blank']} p(skt vs blank)={r['skt_p_vs_blank']}")
    print('  -- discriminating candidates (ind vs skt):')
    for r in sorted([r for r in rs if 'discriminates' in r['role'] or 'opposite' in r['role']],key=lambda r:-r['score'])[:8]:
        print(f"   {r['band_cm-1']:7.1f}  in_blank={r['present_in_blank']:3}  ind x{r['ind_fold_vs_blank']:5.2f} skt x{r['skt_fold_vs_blank']:5.2f} mix x{r['mix_fold_vs_blank']:5.2f}  p(ind/skt)={r['p_ind_vs_skt']:>6}  {r['role']}")
