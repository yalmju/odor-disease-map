"""Exact structure -> HMDB evidence, kept separate from predicted odor.

An unannotated molecule is unknown, never a healthy control. Disease edges
are imported database assertions, not independently adjudicated biomarkers.
"""
import json
import sqlite3
from pathlib import Path
from rdkit import Chem
from .odor_edit import molecule


class DiseaseBridge:
    def __init__(self, database):
        self.path = Path(database).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)

    def lookup(self, smiles):
        key = Chem.MolToInchiKey(molecule(smiles))
        return self.lookup_key(key)

    def lookup_key(self, key):
        con = sqlite3.connect(self.path.as_uri()+'?mode=ro', uri=True)
        try:
            rows = con.execute('SELECT accession,name,update_date,evidence_json FROM clinical WHERE inchikey=?', (key,)).fetchall()
        finally:
            con.close()
        records = [dict(accession=a, name=n, update_date=u,
                        url=f'https://hmdb.ca/metabolites/{a}', **json.loads(e)) for a,n,u,e in rows]
        diseases = sorted({d['name'] for r in records for d in r.get('diseases',[]) if d.get('name')})
        return {'inchikey': key, 'identity_match': 'exact_full_inchikey',
                'status': 'disease_annotation_present' if diseases else ('clinical_record_without_disease_annotation' if records else 'no_linked_clinical_record'),
                'diseases': diseases, 'records': records,
                'evidence_level': 'HMDB archived database annotation; primary references not individually adjudicated',
                'direction': 'not_inferred',
                'interpretation': 'Not diagnosis, disease specificity, causality, or evidence of gas-phase odor. Missing link means unknown.'}

    def attach_comparison(self, comparison):
        result = dict(comparison)
        a = self.lookup(comparison['base']['smiles'])
        b = self.lookup(comparison['candidate']['smiles'])
        result['disease_context'] = {'base':a, 'candidate':b,
            'shared_annotations':sorted(set(a['diseases']) & set(b['diseases'])),
            'base_only_annotations':sorted(set(a['diseases']) - set(b['diseases'])),
            'candidate_only_annotations':sorted(set(b['diseases']) - set(a['diseases'])),
            'interpretation':'Annotation differences, not a causal change in disease risk after substitution.'}
        return result


def evidence_lines(evidence):
    lines = [evidence['inchikey'], evidence['status'], evidence['interpretation']]
    for record in evidence['records']:
        lines += ['', record['accession']+' | '+record['name'], record['url']]
        for d in record.get('diseases',[]):
            refs = ', '.join(r.get('pubmed_id','') for r in d.get('references',[]) if r.get('pubmed_id'))
            lines.append('질병 주석: '+d.get('name','')+' | PMID: '+(refs or '없음'))
        for d in record.get('abnormal_concentrations',[]):
            refs = ', '.join(r.get('pubmed_id','') for r in d.get('references',[]) if r.get('pubmed_id'))
            lines.append('비정상 농도 기록: '+d.get('patient_information','미기재')+' | '+d.get('biospecimen','')+
                         ' | '+d.get('concentration_value','')+' '+d.get('concentration_units','')+
                         ' | 증가/감소 미확인 | PMID: '+(refs or '없음'))
    return '\n'.join(lines)
