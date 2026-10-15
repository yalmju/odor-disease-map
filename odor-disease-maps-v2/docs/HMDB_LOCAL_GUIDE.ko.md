# 로컬 작업 지침서: 대장암 연결 냄새분자 리스트 뽑기

포스터의 "Disease association" 지도에서 분홍 점(대장암 HMDB 주석 분자)을 다시 계산하는 절차입니다.
HMDB 원본이 커서 로컬 PC에서 실행합니다. GPU는 필요 없습니다.

## 0. 먼저 확인: 비공개 파일이 이미 있는 경우

`private_data/evidence_overlay.json` 이 있으면 HMDB를 받을 필요가 없습니다.
그 파일의 `diseases` 에 이미 HMDB 주석이 들어 있으니 바로 뽑으면 됩니다.

```sh
python -c "import json; d=json.load(open('private_data/evidence_overlay.json',encoding='utf-8-sig')); h=[p for p in d['points'] if 'Colorectal cancer' in p['diseases']]; print(len(h)); [print(p['index'], p.get('name',''), p.get('smiles',''), ';'.join(p['odors'])) for p in h]"
```

여기서 나온 개수가 포스터의 분홍 점 개수입니다. 아래 절차는 이 파일이 없거나 최신 HMDB로 다시 확인하고 싶을 때 씁니다.

## 1. 환경 준비

레포의 `odor-disease-maps-v2` 폴더에서 실행합니다. Python 3.10 이상이면 됩니다.

```sh
cd odor-disease-maps-v2
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux:        source .venv/bin/activate
python -m pip install rdkit
```

## 2. 데이터 받기

| 파일 | 받는 곳 | 넣을 위치 |
|---|---|---|
| HMDB 전체 대사체 XML | https://hmdb.ca/downloads 의 "All Metabolites" (XML). 파일명 `hmdb_metabolites.zip` | `private_data/hmdb_metabolites.zip` |
| Leffingwell 향 데이터 | https://github.com/pyrfume/pyrfume-data/tree/main/leffingwell 의 `molecules.csv`, `behavior.csv` | `private_data/leffingwell/` |

- ZIP은 풀지 않아도 됩니다. 스크립트가 ZIP 안의 XML을 스트리밍으로 읽어서 메모리를 적게 씁니다.
- 받은 날짜와 HMDB 버전을 기록해 두세요. 질병 주석은 버전마다 달라집니다.
- `private_data/` 는 `.gitignore` 대상이라 커밋되지 않습니다. HMDB 원본은 재배포 조건을 확인하기 전까지 올리지 마세요.

## 3. HMDB → 로컬 질병 테이블 만들기 (한 번만)

```sh
python scripts/hmdb_disease_list.py build --xml private_data/hmdb_metabolites.zip --out private_data/hmdb_clinical.sqlite
```

- 질병 주석이나 비정상 농도 기록이 있는 대사체만 `clinical` 테이블에 저장합니다.
- 테이블 구조는 `research/src/voclib/disease_bridge.py` 의 `DiseaseBridge` 와 같아서 기존 연구 코드에서도 그대로 읽힙니다.
- 진행 상황이 2만 개마다 찍힙니다. PC 사양에 따라 수십 분 걸릴 수 있습니다.

## 4. 대장암 연결 분자 리스트 뽑기

분자 집합은 셋 중 하나를 고릅니다.

```sh
# (가) Leffingwell 전체 3,522개 분자 기준
python scripts/hmdb_disease_list.py list --db private_data/hmdb_clinical.sqlite --leffingwell private_data/leffingwell --out results/hmdb_disease/leffingwell_colorectal.csv

# (나) 포스터의 532개 지도 분자 기준 (overlay에 smiles 또는 inchikey가 있어야 함)
python scripts/hmdb_disease_list.py list --db private_data/hmdb_clinical.sqlite --overlay private_data/evidence_overlay.json --out results/hmdb_disease/atlas_colorectal.csv

# (다) 직접 만든 분자 목록 CSV (열: name, smiles, odors=라벨;라벨)
python scripts/hmdb_disease_list.py list --db private_data/hmdb_clinical.sqlite --molecules my_molecules.csv
```

- 연결 기준은 연구 코드와 같은 "전체 InChIKey 정확 일치"입니다. 입체이성질체가 다르면 다른 분자로 봅니다.
- 다른 질병은 `--disease "Crohn's disease"` 처럼 HMDB 표기 그대로 넣습니다. 대소문자는 무시합니다.
- 실행 후 화면에 "colorectal" 이 들어간 질병명 변형이 모두 출력됩니다. "Colorectal adenoma" 처럼 다른 이름이 보이면 포함할지 직접 판단하세요.

## 5. 결과 파일 읽는 법

`results/hmdb_disease/*.csv` 는 엑셀에서 바로 열립니다(UTF-8 BOM).

| 열 | 뜻 |
|---|---|
| `name` | 입력 분자 이름 (Leffingwell 또는 overlay) |
| `hmdb_accession`, `hmdb_name` | 연결된 HMDB 번호와 HMDB 이름 |
| `inchikey`, `smiles` | 연결에 쓴 구조 |
| `shown_odors` | 포스터 6개 향(fruity, floral, green, woody, roasted, animal) 중 해당 라벨 |
| `all_odors` | 그 분자의 전체 향 라벨 |
| `pubmed_ids` | 해당 질병 주석의 HMDB 근거 문헌 |
| `index` | overlay 입력일 때 지도 위 점 번호 |

## 6. 숫자가 34개와 다를 때 점검할 것

1. **분자 집합이 다름:** (가) Leffingwell 전체는 3,522개라 지도 532개보다 많이 잡힙니다. 포스터와 맞추려면 (나)를 쓰세요.
2. **HMDB 버전이 다름:** 원래 작업은 2026-09-30 보관본을 썼습니다. 최신 버전은 주석이 늘거나 줄 수 있습니다.
3. **질병명 표기:** 정확히 "Colorectal cancer" 만 셉니다. 4단계에서 출력된 변형 이름을 확인하세요.
4. **overlay 자체 값과 비교:** (나)로 실행하면 마지막 줄에 overlay 파일에 원래 기록된 개수가 함께 나옵니다. 두 숫자가 다르면 HMDB 버전 차이입니다.

## 해석 주의

HMDB 주석은 데이터베이스에 기록된 연관일 뿐입니다. 진단, 인과, 질병 특이성, 기체 상태 농도를 뜻하지 않습니다.
주석이 없는 분자는 "건강" 이 아니라 "모름" 입니다.
