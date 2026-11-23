# 향 구름 합본 지도

향 6개(fruity, floral, green, woody, roasted, animal) 구름을 한 지도에 겹쳐 그리고, 필요하면 질병 점과 1·2번 초점 분자를 올립니다.
좌표, 색, 커널 폭(2.6), 진하기 곡선(0.97 × 밀도^0.65)은 `render_maps.py` 와 같습니다.

## 방법 A. 라벨 데이터가 있을 때 (권장)

`evidence_overlay.json` 처럼 점마다 `odors` 라벨이 있는 파일을 씁니다.

```sh
python scripts/render_merged_odor.py --input private_data/evidence_overlay.json --output results/merged --focus-names "Indole,Skatole"

# 일부 향만 합치기
python scripts/render_merged_odor.py --input private_data/evidence_overlay.json --output results/merged4 --labels fruity,floral,green,woody
```

출력은 PNG(300 dpi), SVG, PDF입니다.

- `10_odor_merged`: 향 구름 합본, 회색 점, 1·2번 표시
- `11_odor_merged_disease`: 위 그림에 질병 점(기본 Colorectal cancer)을 흰 테두리 분홍 점으로 추가
- `merged_manifest.json`: 입력 해시, 라벨별 점 수, 섞는 방식 기록

## 방법 B. 포스터 이미지밖에 없을 때

향 패널 그림에서 구름 진하기를 거꾸로 계산해 씁니다. 분자 라벨은 복원되지 않고 구름 모양만 복원됩니다.

1. 기준 좌표 JSON을 준비합니다. 원본 overlay가 없으면 포스터 질병 패널에서 읽어 낸 좌표 파일을 씁니다.
2. 이미지에서 향 패널마다 제목을 뺀 그림 영역의 픽셀 범위(x0,y0,x1,y1)를 확인합니다. 그림판이나 미리보기 앱에서 마우스 위치로 읽으면 됩니다.
3. 구름을 뽑습니다.

```sh
python scripts/extract_odor_fields.py --image poster.png --points points.json --out results/fields.npz \
  --panel fruity=600,40,900,300 --panel floral=920,40,1220,300 --panel green=1240,40,1540,300 \
  --panel woody=600,320,900,580 --panel roasted=920,320,1220,580 --panel animal=1240,320,1540,580
```

4. 화면에 찍히는 `scale_ratio_x_over_y` 가 1에 가까운지 봅니다. 0.95~1.05를 벗어나면 패널 범위가 틀렸거나 점 경계 자동 인식이 실패한 것이니 `--pixel-box label=x0,y0,x1,y1` 로 점들이 차지하는 범위를 직접 넣으세요.
5. 뽑은 구름으로 합본을 그립니다.

```sh
python scripts/render_merged_odor.py --input points.json --fields results/fields.npz --output results/merged_from_poster --focus-names "Indole,Skatole"
```

검증: 가상 데모로 그린 향 패널을 4분의 1 크기로 줄여 스크린샷처럼 만든 뒤 방법 B로 복원했을 때, 라벨로 계산한 구름과의 상관이 6개 향 모두 0.996 이상이었습니다. 실제 포스터는 JPEG 압축, 크기 조정, 다른 그림 요소 때문에 이보다 낮을 수 있습니다.

## 해석 주의

- 향마다 자기 최댓값으로 정규화하므로, 색 진하기로 향끼리 많고 적음을 비교할 수 없습니다.
- 구름이 겹치는 곳은 진하기 비율대로 색을 섞습니다. 그리는 순서와 무관하지만, 섞인 색은 "두 향이 함께 있는 영역"이라는 뜻일 뿐 새 향을 뜻하지 않습니다.
- floral 구름(분홍)과 질병 점(진분홍)은 색이 비슷해서, 질병 점에는 흰 테두리를 둘렀습니다.
- t-SNE 위의 거리는 냄새나 질병의 실제 유사성을 증명하지 않습니다. 질병 점은 HMDB 주석일 뿐 진단이나 인과가 아닙니다.
