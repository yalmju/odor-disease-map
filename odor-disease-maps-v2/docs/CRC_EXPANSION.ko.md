# 대장암 분자 확장 지도와 비교쌍 분석

포스터의 532개 분자 지도에 대장암 관련 분자를 더하고, OpenPOM 예측 향과 비교쌍 순위를 계산하는 스크립트 묶음입니다. **기존 532개 좌표는 바꾸지 않습니다.** t-SNE를 다시 돌리지 않으므로 포스터 그림과 그대로 겹쳐 쓸 수 있습니다.

## 필요한 비공개 파일 (`private_data/`, 커밋 금지)

| 파일 | 용도 |
|---|---|
| `evidence_overlay.json` | 고정 좌표, 관측 향, HMDB 질병 주석 |
| `inference_cache.npz` | 532개 분자의 OpenPOM 임베딩(256)과 향 점수(138) |
| `checkpoint2.pt` | OpenPOM 체크포인트. 해시는 `research/examples/openpom_checkpoint_config.json` |
| `leffingwell/molecules.csv`, `leffingwell/behavior.csv` | pyrfume Leffingwell 관측 향 라벨 |

`data/crc_additions_curated.csv`는 저장소에 포함됩니다. 이름, CAS, SMILES, 근거 종류만 있고 향 라벨은 없습니다.
- HMDB 근거 9개: HMDB 2021 XML의 "Colorectal cancer" 주석이 있고, Leffingwell 향 라벨이 있으며, 지도에 없던 분자 중에서 휘발성으로 볼 만한 것을 사람이 골랐습니다.
- 문헌 근거 28개: 연구실 VOC 라이브러리(호기·대변 GC-MS)의 대장암 항목 중 Leffingwell에 있는 분자입니다.

## 실행 순서

```sh
# 1) 지도에 없는 분자의 위치 (OpenPOM 환경: research/docs/pom-*-lock.txt, DGLBACKEND=pytorch)
python scripts/place_by_embedding.py --cache private_data/inference_cache.npz \
  --overlay private_data/evidence_overlay.json --molecules data/crc_additions_curated.csv \
  --checkpoint private_data/checkpoint2.pt --out results/crc_placement

# 2) 향 6종 + 대장암 지도, Origin용 CSV (기본 환경 + rdkit)
python scripts/render_crc_expansion.py --overlay private_data/evidence_overlay.json \
  --additions data/crc_additions_curated.csv --placement results/crc_placement/placement.csv \
  --leffingwell private_data/leffingwell --output results/crc_map --font Arial

# 3) OpenPOM 예측 향 패널
python scripts/render_pom_facets.py --points results/crc_map/data/points_all.csv \
  --cache private_data/inference_cache.npz --new-scores results/crc_placement/new_embeddings.npz \
  --labels cheesy,sour,sweaty,fatty,sulfurous,phenolic --output results/crc_pom_facets --font Arial

# 4) 질병 연결이 엇갈리는 비교쌍 순위
python scripts/pair_ranking.py --cache private_data/inference_cache.npz \
  --overlay private_data/evidence_overlay.json --out results/pair_ranking
```

## 방법 요약

- **위치 결정 (1단계):** 새 분자를 같은 체크포인트로 임베딩합니다. 길이를 1로 맞춘 뒤 코사인 유사도가 가장 높은 지도 분자 5개의 좌표를 유사도로 가중평균한 곳에 놓습니다. 실행 전에 캐시된 분자 8개를 다시 계산해 체크포인트가 같은 값을 내는지 확인합니다. 같은 규칙을 기존 532개에 leave-one-out으로 적용한 오차도 함께 보고합니다(연구 데이터 기준 중앙값 0.92, 상위 10% 4.6 t-SNE 단위, 지도 폭 61).
- **그림 (2·3단계):** KDE 폭 2.6, 540×420 격자, 라벨별 최댓값으로 정규화, 투명도 `0.97 × 밀도^0.65`. `render_maps.py`와 같은 규칙입니다. 모든 분자를 같은 점으로 그리며, 출처 구분은 `08b_disease_by_source` 그림과 CSV 열로만 남깁니다.
- **예측 향 (3단계):** OpenPOM 점수 0.5 이상을 그 향으로 봅니다. `descriptor_lift.csv`에 질병 연결 분자 쪽 평균 점수가 높은 향 순위를 남겨 `--labels` 고르는 근거로 씁니다.
- **비교쌍 (4단계):** 질병 연결 분자 × 연결 없는 분자 중 둘 다 관측 향 라벨이 있는 모든 쌍을 임베딩 코사인 순으로 정렬합니다. 같은 쌍의 Morgan Tanimoto(r=2, 2048비트)도 함께 적습니다. 임베딩 공간이 구조 유사도, 예측 향 프로필과 얼마나 맞는지도 Spearman과 최근접 10개 겹침으로 보고합니다.

## 출력 (Origin용)

- `results/crc_map/data/points_all.csv`: 좌표, 관측 향, 6개 향 0/1, `disease_evidence`(HMDB / literature / HMDB+literature / none), `focus`(1·2)
- `results/crc_map/data/kde_<향>_matrix.csv`: 첫 행이 x 격자, 첫 열이 y 격자인 밀도 행렬
- `results/crc_pom_facets/pom_scores_all.csv`: 모든 점의 138개 예측 향 점수
- `results/pair_ranking/discordant_pairs.csv`, `pair_ranking.json`

## 해석할 때 주의할 점

- 새로 놓은 분자의 위치는 원래 t-SNE 최적화에 들어간 좌표가 아니라 근사값입니다.
- 예측 향은 관측 라벨과 다른 근거입니다. OpenPOM은 Leffingwell을 포함해 학습했으므로, 둘이 맞는다고 해서 독립 검증이 되지 않습니다.
- 비교쌍 순위는 이미 고른 쌍을 사후에 설명하는 근거입니다. 검증된 선정 알고리즘이 아니며, 장내 대사산물 여부와 휘발성은 사람이 판단합니다.
- 질병 연결은 "보고가 있다"는 뜻입니다. 주석이 없다고 해서 관련이 없다는 근거가 되지 않습니다.
