"""Native desktop workbench for molecular edits and predicted odor changes."""
from __future__ import annotations
import argparse, csv, json, queue, threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import ImageTk
from rdkit.Chem import Draw
from .odor_edit import OdorPredictor, RATINGS, molecule

PRESETS = {'Indole':'c1ccc2[nH]ccc2c1', 'Skatole':'Cc1c[nH]c2ccccc12',
           'Indole-3-acetic acid':'O=C(O)Cc1c[nH]c2ccccc12',
           'Indole-3-propionic acid':'O=C(O)CCc1c[nH]c2ccccc12',
           'Indolelactic acid':'O=C(O)[C@@H](O)Cc1c[nH]c2ccccc12'}

def display_sd(value):return '—' if value is None else f'{value:.3f}'

class Workbench:
    def __init__(self, root, model_path, clinical_path=None):
        self.root=root; self.model_path=Path(model_path)
        self.deep=self.model_path.suffix.lower()=='.json'
        if self.deep:
            from .deep_odor import DeepOdorPredictor
            self.predictor=DeepOdorPredictor(model_path)
        else:self.predictor=OdorPredictor(model_path)
        from .disease_bridge import DiseaseBridge
        self.disease_bridge=DiseaseBridge(clinical_path) if clinical_path else None
        self.scan_result=None;self.comparison=None;self.events=queue.Queue();self.busy=False;self.images=[]
        root.title('VOCLIB · Molecular Odor Editor');root.geometry('1440x930');root.minsize(1100,760)
        style=ttk.Style();style.theme_use('clam');style.configure('.',font=('맑은 고딕',10))
        style.configure('Treeview',rowheight=25);style.configure('Title.TLabel',font=('맑은 고딕',18,'bold'))
        main=ttk.Frame(root,padding=16);main.pack(fill='both',expand=True)
        ttk.Label(main,text='Molecular Odor Editor',style='Title.TLabel').pack(anchor='w')
        ttk.Label(main,text=f"구조 편집 → {len(self.predictor.bundle['labels'])}개 향 점수와 쾌적도·강도·친숙도 | "+('OpenPOM + PyTorch MLP' if self.deep else 'Baseline')).pack(anchor='w',pady=(2,10))
        controls=ttk.Frame(main);controls.pack(fill='x')
        self.preset=tk.StringVar(value='Indole');choice=ttk.Combobox(controls,textvariable=self.preset,values=list(PRESETS),state='readonly',width=24);choice.pack(side='left')
        self.base=tk.StringVar(value=PRESETS['Indole']);choice.bind('<<ComboboxSelected>>',lambda e:self.base.set(PRESETS[self.preset.get()]))
        ttk.Label(controls,text=' 기준 SMILES ').pack(side='left');ttk.Entry(controls,textvariable=self.base,width=50).pack(side='left',fill='x',expand=True)
        self.dilution=tk.StringVar(value='1/1000');ttk.Label(controls,text=' 희석 ').pack(side='left');ttk.Combobox(controls,textvariable=self.dilution,values=['1/1000','1/100000'],state='readonly',width=12).pack(side='left')
        self.scan_button=ttk.Button(controls,text='치환 후보 생성·예측',command=self.scan);self.scan_button.pack(side='left',padx=8)
        manual=ttk.Frame(main);manual.pack(fill='x',pady=8);ttk.Label(manual,text='비교할 SMILES').pack(side='left')
        self.candidate=tk.StringVar(value=PRESETS['Skatole']);ttk.Entry(manual,textvariable=self.candidate).pack(side='left',fill='x',expand=True,padx=8)
        self.compare_button=ttk.Button(manual,text='직접 비교',command=self.compare);self.compare_button.pack(side='left')
        ttk.Button(manual,text='성능·한계',command=self.show_metrics).pack(side='left',padx=8)
        self.disease_button=ttk.Button(manual,text='질병 근거 보기',command=self.show_disease)
        self.disease_button.pack(side='left',padx=4)
        if not self.disease_bridge:self.disease_button.state(['disabled'])
        self.disease_run=self.model_path.resolve().parents[1]/'disease-odor-v1'
        profile_button=ttk.Button(manual,text='질병별 향·분자',command=self.show_disease_profiles)
        profile_button.pack(side='left',padx=4)
        if not (self.disease_run/'disease_profiles.json').exists():profile_button.state(['disabled'])
        ttk.Button(manual,text='JSON 저장',command=self.export_json).pack(side='left');ttk.Button(manual,text='변화량 CSV',command=self.export_csv).pack(side='left',padx=8)
        self.panel_run=self.model_path.resolve().parent if self.deep else self.model_path.resolve().parents[1]/'panel-design-v1'
        panel_toolbar=ttk.Frame(main);panel_toolbar.pack(fill='x',pady=(0,5))
        panel_button=ttk.Button(panel_toolbar,text='실험 후보 선정 근거',command=self.show_panel_design);panel_button.pack(side='left')
        ttk.Label(panel_toolbar,text='  구조 비교쌍 · 기체 문헌 · 질병 주석 · 부족한 근거').pack(side='left')
        if not (self.panel_run/'panel.json').exists():panel_button.state(['disabled'])
        self.status=tk.StringVar(value='학습된 모델을 로드했습니다. 후보 생성 또는 직접 비교를 실행하세요.')
        ttk.Label(main,textvariable=self.status,wraplength=1320).pack(anchor='w',pady=(0,7))
        panes=ttk.Panedwindow(main,orient='horizontal');panes.pack(fill='both',expand=True)
        left=ttk.Frame(panes);right=ttk.Frame(panes);panes.add(left,weight=2);panes.add(right,weight=3)
        ttk.Label(left,text='치환 후보 · 예측 향 변화량 순서 / 위치는 그림의 원자 번호').pack(anchor='w')
        self.candidates=self.make_table(left,['위치','치환기','평균 |Δ|','최근접 유사도'],[48,145,85,105],height=10)
        self.candidates.bind('<<TreeviewSelect>>',self.select_candidate)
        self.structure=ttk.Label(left);self.structure.pack(fill='x',pady=8)
        self.evidence=tk.StringVar(value='');ttk.Label(left,textvariable=self.evidence,wraplength=500,justify='left').pack(anchor='w')
        ttk.Label(right,text='향 descriptor · 후보 − 기준 / 점수는 보정된 확률이 아닙니다').pack(anchor='w')
        self.descriptors=self.make_table(right,['Descriptor','기준','후보','Δ','후보 모델 SD'],[150,75,75,85,150],height=15)
        ttk.Label(right,text='Perceptual ratings · 같은 희석 조건의 집단 평균 예측 (0–1)').pack(anchor='w',pady=(12,0))
        self.ratings=self.make_table(right,['Rating','기준','후보','Δ','후보 seed SD' if self.deep else '후보 tree SD'],[150,75,75,85,150],height=3)
        self.support=tk.StringVar();ttk.Label(right,textvariable=self.support,wraplength=720,justify='left').pack(anchor='w',pady=7)
        ttk.Label(main,text='실측 향·휘발성·합성 가능성을 보장하지 않습니다. SD는 모델 간 산포이며 신뢰구간이 아닙니다. 표면/SERS 예측은 포함하지 않습니다.',wraplength=1300).pack(anchor='w',pady=(8,0))
        root.after(100,self.poll)

    def make_table(self,parent,columns,widths,height):
        frame=ttk.Frame(parent);frame.pack(fill='both',expand=True)
        tree=ttk.Treeview(frame,columns=columns,show='headings',height=height,selectmode='browse')
        for name,width in zip(columns,widths):tree.heading(name,text=name);tree.column(name,width=width,minwidth=45,anchor='w')
        scroll=ttk.Scrollbar(frame,orient='vertical',command=tree.yview);tree.configure(yscrollcommand=scroll.set)
        tree.pack(side='left',fill='both',expand=True);scroll.pack(side='right',fill='y');return tree

    def concentration(self):return .001 if self.dilution.get()=='1/1000' else .00001

    def start(self,fn,kind):
        if self.busy:return
        self.busy=True;self.scan_button.state(['disabled']);self.compare_button.state(['disabled']);self.status.set('분자 구조와 모델 예측을 계산하고 있습니다…')
        def worker():
            try:self.events.put((kind,fn()))
            except Exception as e:self.events.put(('error',str(e)))
        threading.Thread(target=worker,daemon=True).start()

    def scan(self):
        base=self.base.get();c=self.concentration();self.start(lambda:self.predictor.scan(base,c),'scan')

    def compare(self):
        base=self.base.get();candidate=self.candidate.get();c=self.concentration();self.start(lambda:self.predictor.compare(base,candidate,c),'compare')

    def poll(self):
        try:
            kind,result=self.events.get_nowait();self.busy=False;self.scan_button.state(['!disabled']);self.compare_button.state(['!disabled'])
            if kind=='error':self.status.set('입력/실행 오류: '+result);messagebox.showerror('실행 오류',result)
            elif kind=='scan':self.set_scan(result)
            else:self.show_comparison(result);self.status.set('두 분자의 예측 비교 완료. 자동 생성 후보 목록과는 별개의 직접 비교입니다.')
        except queue.Empty:pass
        self.root.after(100,self.poll)

    def set_scan(self,result):
        self.scan_result=result;self.candidates.delete(*self.candidates.get_children())
        for i,r in enumerate(result['candidates']):
            self.candidates.insert('', 'end',iid=str(i),values=(r['atom_index'],r['substituent'],f"{r['mean_absolute_descriptor_change']:.3f}",f"{r['prediction']['nearest_descriptor_train']['similarity']:.3f}"))
        self.status.set(f"{len(result['candidates'])}개 치환 후보 생성 완료. 구조적으로 동일한 후보는 중복 제거했습니다.")
        if result['candidates']:
            self.candidates.selection_set('0');self.select_candidate()
        else:self.status.set('치환 가능한 방향족 C–H / N–H 위치가 없습니다.')

    def select_candidate(self,event=None):
        selection=self.candidates.selection()
        if not selection or self.scan_result is None:return
        r=self.scan_result['candidates'][int(selection[0])];self.candidate.set(r['smiles'])
        self.show_comparison({'base':self.scan_result['base'],'candidate':r['prediction'],
            'descriptor_delta':r['descriptor_delta'],'rating_delta':r['rating_delta'],
            'edit':{'atom_index':r['atom_index'],'substituent':r['substituent']},
            'interpretation':'Predicted difference, not a measured or causal effect.'})

    def show_comparison(self,result):
        if self.disease_bridge:result=self.disease_bridge.attach_comparison(result)
        self.comparison=result;a=result['base'];b=result['candidate']
        self.descriptors.delete(*self.descriptors.get_children());self.ratings.delete(*self.ratings.get_children())
        for name,delta in sorted(result['descriptor_delta'].items(),key=lambda kv:abs(kv[1]),reverse=True):
            self.descriptors.insert('','end',values=(name,f"{a['descriptors'][name]['score']:.3f}",f"{b['descriptors'][name]['score']:.3f}",f'{delta:+.3f}',display_sd(b['descriptors'][name]['bootstrap_sd'])))
        for name in RATINGS:
            self.ratings.insert('','end',values=(name,f"{a['ratings'][name]['score']:.3f}",f"{b['ratings'][name]['score']:.3f}",f"{result['rating_delta'][name]:+.3f}",f"{b['ratings'][name]['tree_sd']:.3f}"))
        mols=[molecule(a['smiles']),molecule(b['smiles'])];options=Draw.MolDrawOptions();options.addAtomIndices=True
        pic=Draw.MolsToGridImage(mols,molsPerRow=2,subImgSize=(250,220),legends=['Base','Candidate'],drawOptions=options)
        self.images=[ImageTk.PhotoImage(pic)];self.structure.configure(image=self.images[0])
        def annotation(r):
            return ' / '.join(', '.join(x['labels'])+' ['+x['split']+']' for x in r['observed_annotations']) or '현재 학습 자료에 관측 향 주석 없음'
        self.evidence.set('관측 주석 (예측과 별개)\n기준: '+annotation(a)+'\n후보: '+annotation(b)+'\n\n후보 SMILES: '+b['smiles'])
        if self.disease_bridge:
            context=result['disease_context']
            def diseases(side):return ', '.join(context[side]['diseases'][:4]) or '연결된 질병 주석 없음(미확인)'
            self.evidence.set(self.evidence.get()+'\n\nHMDB 질병 주석 · 진단 예측 아님\n기준: '+diseases('base')+'\n후보: '+diseases('candidate'))
        self.support.set(f"후보–학습 구조 최대 유사도: descriptor {b['nearest_descriptor_train']['similarity']:.3f} / rating {b['nearest_rating_train_similarity']:.3f}\nKeller paraffin oil 희석 조건: {a['concentration']:g}. 유사도는 정확도 보장이 아닙니다.")
        if self.deep:self.support.set(f"후보–rating head 학습 구조 최대 유사도: {b['nearest_rating_train_similarity']:.3f}\nPOM 사전학습 포함 여부 미확인. 향 SD — = 미평가. Rating SD = 3개 seed 산포.\nKeller paraffin oil 희석 {a['concentration']:g}; 기체 반응 예측이 아닙니다.")

    def show_panel_design(self):
        path=self.panel_run/'panel.json'
        if not path.exists():return
        report=json.loads(path.read_text(encoding='utf-8'))
        window=tk.Toplevel(self.root);window.title('실험 후보 선정 · 근거와 보류 사유');window.geometry('1100x780')
        ttk.Label(window,text='우선 전달량 검증 패널: '+', '.join(report['selected']),font=('맑은 고딕',13,'bold')).pack(anchor='w',padx=12,pady=10)
        ttk.Label(window,text='시작 후보 10개에 대한 규칙 기반 제안. Au–SAM 성능·최적 질병 패널이 검증된 것은 아닙니다.',wraplength=1000).pack(anchor='w',padx=12)
        table=self.make_table(window,['후보','상태','학습 구조 유사도','모델 산포','질병 주석 수'],[200,290,160,130,120],12)
        for i,c in enumerate(report['cards']):
            state='우선 전달량 검증' if c['name'] in report['selected'] else '기체 근거 보완 필요'
            table.insert('','end',iid=str(i),values=(c['name'],state,f"{c['uncertainty']['nearest_training_similarity']:.3f}",display_sd(c['uncertainty']['mean_descriptor_bootstrap_sd']),len(c['clinical']['diseases'])))
        detail=tk.Text(window,height=16,wrap='word');detail.pack(fill='both',expand=True,padx=12,pady=10)
        def select(event=None):
            items=table.selection()
            if not items:return
            c=report['cards'][int(items[0])]
            decisions=[d for d in report['decisions'] if d.get('selected')==c['name'] or c['name'] in d.get('selected_pair',[])]
            evidence=report['sources'].get(c.get('gas_source'),{})
            display={'candidate':c['name'],'selection_reason':decisions,'published_gas_evidence':evidence,
                     'human_detection_threshold_ug_per_l_air':c.get('odt_ug_per_l_air'),
                     'missing_evidence':c['missing_evidence'],'uncertainty':c['uncertainty'],
                     'note':'질병 주석 수는 선정 점수에 사용하지 않습니다. 관측 향·모델 예측·기체 근거를 구분합니다.'}
            detail.configure(state='normal');detail.delete('1.0','end');detail.insert('1.0',json.dumps(display,ensure_ascii=False,indent=2));detail.configure(state='disabled')
        table.bind('<<TreeviewSelect>>',select)
        if report['cards']:table.selection_set('0');select()

    def show_disease_profiles(self):
        path=self.disease_run/'disease_profiles.json'
        if not path.exists():return
        data=json.loads(path.read_text(encoding='utf-8'))
        molecules={m['inchikey']:m for m in json.loads((self.disease_run/'molecule_evidence.json').read_text(encoding='utf-8'))}
        profiles={p['disease']:p for p in data['profiles']}
        window=tk.Toplevel(self.root);window.title('질병·상태별 연결 분자와 관측 향');window.geometry('1050x760')
        ttk.Label(window,text='연결 분자의 향 주석 분포입니다. 환자의 실제 냄새·진단 확률이 아닙니다.',wraplength=1000).pack(anchor='w',padx=10,pady=8)
        choice=tk.StringVar(value='Colorectal cancer' if 'Colorectal cancer' in profiles else next(iter(profiles)))
        combo=ttk.Combobox(window,textvariable=choice,values=sorted(profiles),width=65,state='readonly');combo.pack(anchor='w',padx=10)
        summary=tk.StringVar();ttk.Label(window,textvariable=summary,wraplength=1000).pack(anchor='w',padx=10,pady=8)
        table=self.make_table(window,['분자','관측 향 주석','HMDB'],[250,560,160],15)
        detail=tk.Text(window,height=9,wrap='word');detail.pack(fill='both',padx=10,pady=8)
        def refresh(event=None):
            p=profiles[choice.get()];table.delete(*table.get_children())
            summary.set(f"연결 분자 {p['n_molecules']}개 | "+', '.join(f'{k}: {v}개' for k,v in list(p['odor_annotation_counts'].items())[:8]))
            for key in p['molecule_keys']:
                m=molecules[key];table.insert('','end',iid=key,values=(m['name'],', '.join(m['observed_odors']),', '.join(r['accession'] for r in m['clinical']['records'])))
        def select(event=None):
            from .disease_bridge import evidence_lines
            keys=table.selection()
            if keys:
                detail.configure(state='normal');detail.delete('1.0','end');detail.insert('1.0',evidence_lines(molecules[keys[0]]['clinical']));detail.configure(state='disabled')
        combo.bind('<<ComboboxSelected>>',refresh);table.bind('<<TreeviewSelect>>',select);refresh()

    def show_disease(self):
        if not self.comparison or 'disease_context' not in self.comparison:return
        from .disease_bridge import evidence_lines
        context=self.comparison['disease_context']
        window=tk.Toplevel(self.root);window.title('구조–향–질병 · 원문 근거');window.geometry('1000x700')
        text=tk.Text(window,wrap='word');scroll=ttk.Scrollbar(window,command=text.yview)
        text.configure(yscrollcommand=scroll.set);scroll.pack(side='right',fill='y');text.pack(fill='both',expand=True)
        text.insert('1.0','정확한 분자 식별자로 연결한 HMDB 주석입니다. 질병 위험·치환의 인과 효과가 아닙니다.\n'
                    +'비정상 농도는 증가/감소로 자동 해석하지 않습니다. 전체 근거는 JSON 저장에도 포함됩니다.\n\n'
                    +'기준 분자\n'+evidence_lines(context['base'])+'\n\n후보 분자\n'+evidence_lines(context['candidate']))
        text.configure(state='disabled')

    def show_metrics(self):
        if self.deep:
            r=self.predictor.bundle['report']
            messagebox.showinfo('딥러닝 내부 검증',json.dumps(r,ensure_ascii=False,indent=2))
            return
        r=self.predictor.bundle['report'];d=r['descriptor_test'];lines=[f"Descriptor: AP {d['macro_average_precision']:.3f} / ROC-AUC {d['macro_roc_auc']:.3f} / F1 {d['macro_f1']:.3f}",r['split_policy'],'']
        for name,m in r['rating_test'].items():lines.append(f"{name}: MAE {m['mae']:.3f}; mean baseline {m['baseline_mae']:.3f}; r {m['pearson_r']:.3f}")
        fm=r['indole_family_holdout']['metrics']
        lines+=['',f"별도 인돌 전체 제외 평가: n=6 / 8개 평가 가능 향 AUC {fm['macro_roc_auc']:.3f}.",
                '인돌계열 신규 구조 일반화가 약하며 표본이 작습니다.',
                'Perceptual ratings = 집단 평균 예측. 개인차 예측 모델이 아닙니다.','치환 효과의 실험 검증은 아직 없습니다.','인돌계열 제외 평가의 상세 결과는 report.json에 있습니다.']
        messagebox.showinfo('독립 시험 성능과 한계','\n'.join(lines))

    def export_json(self):
        if self.comparison is None:return
        path=filedialog.asksaveasfilename(defaultextension='.json',initialfile='odor_comparison.json')
        if path:Path(path).write_text(json.dumps({'comparison':self.comparison,'scan':self.scan_result,'model':str(self.model_path),'report':self.predictor.bundle['report']},ensure_ascii=False,indent=2),encoding='utf-8')

    def export_csv(self):
        if self.comparison is None:return
        path=filedialog.asksaveasfilename(defaultextension='.csv',initialfile='odor_delta.csv')
        if not path:return
        r=self.comparison
        with open(path,'w',encoding='utf-8-sig',newline='') as f:
            w=csv.writer(f);w.writerow(['task','label','base_smiles','candidate_smiles','concentration','base_score','candidate_score','delta','evidence'])
            for task,delta_key in [('descriptors','descriptor_delta'),('ratings','rating_delta')]:
                for label,delta in r[delta_key].items():w.writerow([task,label,r['base']['smiles'],r['candidate']['smiles'],r['base']['concentration'],r['base'][task][label]['score'],r['candidate'][task][label]['score'],delta,'model_prediction'])

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',type=Path,required=True);parser.add_argument('--smoke',action='store_true');parser.add_argument('--clinical',type=Path);args=parser.parse_args()
    if args.clinical is None:
        default=Path(__file__).resolve().parents[3]/'odor_algorithm_data_2026-09-30/analysis/hmdb_clinical.sqlite'
        if default.exists():args.clinical=default
    root=tk.Tk()
    if args.smoke:root.withdraw()
    app=Workbench(root,args.model,args.clinical)
    if args.smoke:
        app.set_scan(app.predictor.scan(PRESETS['Indole']))
        root.update_idletasks()
        assert len(app.candidates.get_children())==56
        assert len(app.descriptors.get_children())==len(app.predictor.bundle['labels'])
        assert len(app.ratings.get_children())==3
        if args.clinical:
            assert 'disease_context' in app.comparison
            app.show_disease()
            app.show_disease_profiles()
        app.show_panel_design()
        print(f"Native GUI smoke passed: 56 candidates, {len(app.predictor.bundle['labels'])} descriptors, 3 ratings, molecular drawings.");root.destroy()
    else:
        root.after(200,app.scan)
        root.mainloop()

if __name__=='__main__':main()
