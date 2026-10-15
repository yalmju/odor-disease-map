"""Build a local HMDB clinical table and list odor molecules linked to one disease.

Step 1 (build): stream hmdb_metabolites.zip/.xml into hmdb_clinical.sqlite.
  The table layout matches research/src/voclib/disease_bridge.py, so the
  existing DiseaseBridge can read the output directly.
Step 2 (list): join a molecule set to that table by exact full InChIKey and
  write every molecule whose HMDB disease annotations include the target.

A missing link means "unknown", never "healthy". Annotations are archived
database assertions, not diagnosis, causality or volatile abundance.
"""
import argparse
import csv
import json
import sqlite3
import sys
import zipfile
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET

NS = '{http://www.hmdb.ca}'
SHOWN = ['fruity', 'floral', 'green', 'woody', 'roasted', 'animal']


def _text(el, tag):
    child = el.find(NS + tag)
    return (child.text or '').strip() if child is not None else ''


def _refs(el):
    out = []
    for r in el.iterfind(f'{NS}references/{NS}reference'):
        pmid, text = _text(r, 'pubmed_id'), _text(r, 'reference_text')
        if pmid or text:
            out.append({'pubmed_id': pmid, 'reference_text': text})
    return out


def _open_xml(path):
    path = Path(path)
    if path.suffix.lower() == '.zip':
        z = zipfile.ZipFile(path)
        names = [n for n in z.namelist() if n.lower().endswith('.xml')]
        if len(names) != 1:
            raise SystemExit(f'Expected one XML inside {path}, found {names}')
        return z.open(names[0])
    return open(path, 'rb')


def build(args):
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f'{out} exists; delete it or choose another --out')
    con = sqlite3.connect(out)
    con.execute('CREATE TABLE clinical (accession TEXT PRIMARY KEY, name TEXT, update_date TEXT, '
                'inchikey TEXT, smiles TEXT, version TEXT, evidence_json TEXT)')
    con.execute('CREATE INDEX clinical_inchikey ON clinical(inchikey)')
    seen = kept = 0
    with _open_xml(args.xml) as fh:
        for _, el in ET.iterparse(fh, events=('end',)):
            if el.tag != NS + 'metabolite':
                continue
            seen += 1
            diseases = [{'name': _text(d, 'name'), 'omim_id': _text(d, 'omim_id'), 'references': _refs(d)}
                        for d in el.iterfind(f'{NS}diseases/{NS}disease') if _text(d, 'name')]
            abnormal = [{k: _text(c, k) for k in ['biospecimen', 'concentration_value', 'concentration_units',
                                                  'patient_information', 'subject_condition']} | {'references': _refs(c)}
                        for c in el.iterfind(f'{NS}abnormal_concentrations/{NS}concentration')]
            key = _text(el, 'inchikey').replace('InChIKey=', '')
            if key and (diseases or abnormal):
                con.execute('INSERT OR REPLACE INTO clinical VALUES (?,?,?,?,?,?,?)',
                            (_text(el, 'accession'), _text(el, 'name'), _text(el, 'update_date'), key,
                             _text(el, 'smiles'), _text(el, 'version'),
                             json.dumps({'diseases': diseases, 'abnormal_concentrations': abnormal}, ensure_ascii=False)))
                kept += 1
            el.clear()
            if seen % 20000 == 0:
                con.commit()
                print(f'  parsed {seen:,} metabolites, kept {kept:,}', file=sys.stderr)
    con.commit()
    con.close()
    print(f'Done: {seen:,} metabolites parsed, {kept:,} clinical records -> {out}')


def _inchikey(smiles):
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    return Chem.MolToInchiKey(mol) if mol else ''


def load_molecules(args):
    """Return rows with name, smiles, inchikey, odors from one of three sources."""
    rows = []
    if args.leffingwell:
        d = Path(args.leffingwell)
        labels = {}
        with open(d / 'behavior.csv', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                labels[r['Stimulus']] = [k for k, v in r.items() if k != 'Stimulus' and v == '1']
        with open(d / 'molecules.csv', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                smi = r['IsomericSMILES']
                rows.append({'name': r['name'], 'smiles': smi, 'inchikey': _inchikey(smi),
                             'odors': labels.get(r['CID'], [])})
    elif args.overlay:
        doc = json.loads(Path(args.overlay).read_text(encoding='utf-8-sig'))
        for p in doc['points']:
            smi = p.get('smiles', '')
            rows.append({'name': p.get('name', str(p['index'])), 'smiles': smi,
                         'inchikey': p.get('inchikey') or _inchikey(smi), 'odors': p.get('odors', []),
                         'index': p['index'], 'overlay_diseases': p.get('diseases', [])})
    elif args.molecules:
        with open(args.molecules, encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                smi = r.get('smiles') or r.get('SMILES') or ''
                rows.append({'name': r.get('name', ''), 'smiles': smi,
                             'inchikey': r.get('inchikey') or _inchikey(smi),
                             'odors': [o for o in (r.get('odors') or '').split(';') if o]})
    else:
        raise SystemExit('Give one of --leffingwell, --overlay or --molecules')
    bad = [r for r in rows if not r['inchikey']]
    if bad:
        print(f'Warning: {len(bad)} molecules without a parsable structure were skipped', file=sys.stderr)
    return [r for r in rows if r['inchikey']]


def list_disease(args):
    con = sqlite3.connect(Path(args.db).resolve().as_uri() + '?mode=ro', uri=True)
    rows = load_molecules(args)
    target = args.disease.casefold()
    hits, names_like = [], Counter()
    for r in rows:
        recs = con.execute('SELECT accession,name,evidence_json FROM clinical WHERE inchikey=?',
                           (r['inchikey'],)).fetchall()
        matched, pmids = [], set()
        for acc, hname, ev in recs:
            for d in json.loads(ev)['diseases']:
                if args.keyword and args.keyword.casefold() in d['name'].casefold():
                    names_like[d['name']] += 1
                if d['name'].casefold() == target:
                    matched.append((acc, hname))
                    pmids.update(x['pubmed_id'] for x in d['references'] if x['pubmed_id'])
        if matched:
            hits.append({'name': r['name'], 'hmdb_accession': ';'.join(sorted({a for a, _ in matched})),
                         'hmdb_name': ';'.join(sorted({n for _, n in matched})),
                         'inchikey': r['inchikey'], 'smiles': r['smiles'],
                         'shown_odors': ';'.join(o for o in SHOWN if o in r['odors']),
                         'all_odors': ';'.join(r['odors']), 'pubmed_ids': ';'.join(sorted(pmids)),
                         'index': r.get('index', '')})
    con.close()
    hits.sort(key=lambda h: h['name'].casefold())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(hits[0]) if hits else ['name'])
        w.writeheader()
        w.writerows(hits)
    print(f'{len(rows):,} molecules checked; {len(hits)} linked to "{args.disease}" -> {out}')
    if names_like:
        print(f'Disease names containing "{args.keyword}" in this molecule set (check spelling variants):')
        for n, c in names_like.most_common():
            print(f'  {c:4d}  {n}')
    if args.overlay:
        overlay_n = sum(args.disease in r['overlay_diseases'] for r in rows)
        print(f'Overlay file itself marks {overlay_n} points with "{args.disease}".')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build', help='HMDB XML/ZIP -> hmdb_clinical.sqlite')
    b.add_argument('--xml', required=True, help='hmdb_metabolites.zip or hmdb_metabolites.xml')
    b.add_argument('--out', default='private_data/hmdb_clinical.sqlite')
    b.set_defaults(func=build)
    l = sub.add_parser('list', help='List molecules linked to one disease')
    l.add_argument('--db', default='private_data/hmdb_clinical.sqlite')
    src = l.add_mutually_exclusive_group(required=True)
    src.add_argument('--leffingwell', help='Folder with pyrfume leffingwell molecules.csv and behavior.csv')
    src.add_argument('--overlay', help='evidence_overlay.json with smiles or inchikey per point')
    src.add_argument('--molecules', help='CSV with smiles (or inchikey), optional name and odors=a;b')
    l.add_argument('--disease', default='Colorectal cancer', help='Exact HMDB disease name (case-insensitive)')
    l.add_argument('--keyword', default='colorectal', help='Also report disease names containing this word')
    l.add_argument('--out', default='results/hmdb_disease/colorectal_cancer_molecules.csv')
    l.set_defaults(func=list_disease)
    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
