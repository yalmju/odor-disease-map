# SERS 리포터 채널 분석 지침 (다른 세션용)

목적: 리포터 채널(pristine / 4-MBA / 4-MPY / 2-NT) SERS 데이터를 같은 방식으로 처리해, 2026-10-13까지 확정된 내용과 비교하고 남은 확인 항목을 닫는다. 이 폴더의 스크립트만 있으면 되고, 데이터는 레포에 넣지 않는다.

환경: Python 3.11, `numpy scipy matplotlib`(requirements.txt). 레포 루트에서 실행.

## 0. 표기 규칙 (지키지 않으면 비교가 안 됨)
- 지표는 **같은 스펙트럼 안의 밴드 세기 비**(I_a / I_ref). ref는 채널의 가장 강한 리포터 밴드: MBA 1078, MPY 1095, NT 1064. pristine은 1340. `sers_config.py`에 정의.
- 통계는 평균 ± SD, blank 대비 배수, Mann–Whitney p, leave-one-out 정확도만. 효과크기(d) 같은 새 지표 쓰지 않음.
- 전체 스펙트럼 PCA로 "분류됨"이라고 쓰지 않는다. 세트별 세기·배경 차이를 읽어서 100%가 나온다(10/13 pristine에서 화학 변화 없는 스카톨까지 100% 갈라짐). 산점도·분류는 비율 지표로만.
- 밴드 위치 이동은 2 cm⁻¹ 미만(격자 1.7)이면 주장하지 않는다.
- 한 조건이 측정 세트 1개(필름 1장)면 LOO 정확도는 "세트 내" 숫자다. 캡션에 적는다.

## 1. 데이터 형식과 파일명
**포인트**: `<channel>_<analyte>.txt`, 탭 구분, 1열 Raman shift(2001행, 100–3500), 2열부터 포인트별 세기(와이드). channel ∈ {pristine, mba, mpy, nat}, analyte ∈ {blank, ind, skt, mix, …}. 장비 원본 파일명을 이 규칙으로 복사해 둔다(예: `skt_mba_633_50_500ms_10points.txt` → `mba_skt.txt`).

**맵**: 매핑 소프트웨어 CSV 그대로. `<name>.csv`와 `<name>(bg).csv`(배경 제거본, 있으면 이걸 씀). 헤더 2줄(`X num,10` / `Y num,10`), 3행 `X,Y,<shift...>`, 이후 픽셀 1행. 폴더별 `Table.csv`의 Band 열은 전부 0이라 쓰지 않는다.

## 2. 스크립트
| 스크립트 | 입력 | 출력 |
|---|---|---|
| `point_spectra.py --root DIR --out OUT` | 포인트 txt 폴더 | 채널별 평균±SD, blank 대비 차이 스펙트럼, 밴드 위치·세기 JSON |
| `point_ratios.py --root DIR --out OUT` | 포인트 txt 폴더 | `point_features.csv`(포인트별 비율), `group_summary.csv`(평균·SD·배수·p), `ratio_scatter.png`, `pure_confusion.csv`(LOO), `pure_pca_scatter.png` |
| `fingerprint_candidates.py DIR OUT` | 포인트 txt 폴더 | `fingerprint_candidates.csv`(피크별 blank 존재 여부, 배수, p, 역할), `fingerprint_candidates.png` |
| `map_unmix.py --root DIR --out OUT --pure A,B --mix C,D [--channel mba]` | 맵 CSV 폴더 | `pixel_ratios.csv`(film,map,analyte + 비율; analyte는 맵 이름에서 끝 숫자를 뗀 것, 예 ind-1→ind, ind-skt2→ind-skt; 아래 4단계 입력), NNLS 분해 몫·공간맵·잔차 |
| `../scripts/channel_validation.py --input pixel_ratios.csv --channels R.. --out OUT` | 픽셀 비율 CSV | 랜덤 픽셀 분할 vs leave-one-map-out vs leave-one-film-out 정확도·혼동행렬, 채널 단독 vs 조합, 분산 분해 |

실행 예:
```sh
python research/sers/point_ratios.py --root private_data/points_2026-10-13 --out results/points
python research/sers/fingerprint_candidates.py private_data/points_2026-10-13 results/points
python research/sers/map_unmix.py --root private_data/maps_2026-10-08/2026-10-08_pure --out results/maps --pure ind-1,skt-1 --mix ind-skt2,ind-skt3,ind-skt4
python research/scripts/channel_validation.py --input results/maps/pixel_ratios.csv --channels R1010/1340,R1490/1340,R1580/1340 --out results/maps/validation
```

## 3. 확정된 기준값 (새 데이터는 이것과 비교)
| 항목 | 값 | 출처 |
|---|---|---|
| pristine에서 스카톨 | 거의 무신호 (blank와 LOO 혼동 3/10, 맵 픽셀 SNR 1.1) | 10/8 맵, 10/13 포인트 |
| MBA 1078 밴드 | blank 2,460 → 인돌 0.55배(1,350) / 스카톨 1.30배(3,200); 예전 배치 0.70배 / 1.05배 | 두 배치 |
| MBA 전대칭/곁밴드 비 (1078+1590)/(691+1178+1480) | blank 4.00±0.12, 인돌 3.33 (p 0.002), 스카톨 5.21 (p 0.0002) | 10/13 |
| MBA 인돌만 움직이는 밴드 | 1230 신규(5배), 1340/1078 2.7배, 1420 1.8배, 1700 1.5배. 스카톨은 1340/1078 0.78배 | 10/13 |
| NT 1375/1064 | blank 0.52 → 인돌 1.09, 스카톨 1.07, 혼합물 1.12 (셋 다 p 0.0002) | 10/13 |
| NT 1580/1064 "감소" | 리포터 모드 아님. blank의 폭 80 cm⁻¹ 1550 성분이 빠지는 것. 1617은 불변. 지표로 쓰지 않음 | 10/13 |
| MPY R1611/1574 | blank 1.27, 인돌 0.95, 스카톨 0.61, IAA 1.0 (예전 배치 채택). 10/13 MPY 노출분은 방향이 반대라 제외, blank 1.28은 일치 | 예전 배치 |
| 순수 3클래스 LOO(비율) | pristine 0.83, MBA 0.90, NT 0.97. 오분류는 blank↔스카톨만 | 10/13 |
| 채널 역할 | MBA 인돌 식별, MPY 스카톨 식별, NT 존재(인돌계 공통), pristine 인돌만 | 종합 |
| 혼합물 | 순수의 선형 합 아님. 맵 NNLS R² 0.5–0.6, 공통 잔차 722·931·999–1023·1118·1181–1198·1397. 순수 학습 모델이 같은 혼합물을 pristine "인돌", MBA "blank", NT "스카톨"로 읽음 | 10/8, 10/13 |
| 혼합물 시료 주의 | 세기 2–6배, NT·MPY에 1350 상승 + 1550 폭 80 둔덕(탄소 D/G형). 시료 확인 전 혼합물 고유 패턴 주장은 pristine·MBA 지표만 | 10/13 |

## 4. 맵 데이터가 오면 할 일 (순서대로)
1. 파일 배치: 조건별 맵 CSV를 한 폴더에. 같은 조건이 필름 여러 장이면 `--film-of ind-A=F1,ind-B=F2,...`로 필름을 지정한다. **이게 이번 맵의 핵심이다. 필름이 1장이면 4단계가 불가능하다.**
2. `map_unmix.py`로 픽셀 비율 테이블과 NNLS 몫. 확인: 순수 맵의 자기 몫 중앙값 ≥ 0.9(sanity), 혼합물 R² (0.5 이하면 선형 합 아님이 재현).
3. 채널별 비율 맵 그림(MBA R1340/1078, NT R1375/1064)에서 인돌 맵만 밝아지는지. 10/8 맵의 Moran's I(+0.3)처럼 공간 상관이 있으면 픽셀은 독립 표본이 아니라고 캡션에 적는다.
4. `channel_validation.py`: 랜덤 픽셀 분할 정확도와 leave-one-film-out 정확도를 나란히 보고한다. 둘의 차이가 곧 "누수" 크기다. 포스터 혼동행렬 n·검증 단위 빈칸은 이 숫자로 채운다.
5. 분산 분해: 필름 간 / 맵 간 / 맵 내. MPY·NT에서 "스카톨 세트가 전 구간 고르게 상승"하는 현상이 맵 간 분산으로 나오면 배경, 맵 내면 흡착이다.

## 5. NT "인돌계 근처" 확인 (비교군 필요)
현재는 인돌·스카톨·혼합물 셋이 NT 1375/1064를 똑같이 두 배로 올린다는 것까지만 사실이다. "인돌계 특이"라고 쓰려면:
- 크레졸·페놀(포스터 후보 그림에 있음) 또는 비방향족 VOC의 NT 포인트 10점.
- 판정: 1375/1064가 blank 수준(0.52±0.06)이면 인돌계 근처 신호, 두 배로 오르면 "방향족 전부"로 문장을 바꾼다.
- 같은 시료를 MBA에서도 재서 1230·1340/1078이 안 오르면 MBA의 인돌 특이성도 함께 확인된다.

## 6. 혼합물 시료 확인
- 혼합물 포인트의 원시 세기(600–1800 평균)가 순수의 2배 이상이면 시료 문제를 먼저 의심한다(10/13: pristine 576 vs 76–163, MPY 2073 vs 331).
- 1300–1350 수준이 blank의 2배 이상이고 1550 폭 80 cm⁻¹ 성분이 남으면 탄소질·형광 성분. 이 경우 NT·MPY의 혼합물 특이 밴드는 쓰지 않는다.
- pristine·MBA의 혼합물 특징(인돌 약화, MBA 691·804 억제, 인돌 표식 소실)은 둔덕과 무관한 밴드라 유지 가능.

## 7. 가설 (증명 전까지 가설로만)
- 인돌 NH–MBA 카복실레이트 수소결합을 C3 메틸이 막는다. 근거: 인돌만 1420·1700·1230 변화.
- 결합 여부가 MBA–금 전하이동 증강을 반대로 민다. 근거: 전대칭 모드(1078, 1590) 선택적 변화, 두 배치 재현.
- 검증 경로: N-메틸인돌 대조(효과 소실 여부), DFT(MBA⁻ + 인돌/스카톨 복합체 결합 에너지와 1420/1700 이동, B3LYP-D3/def2-TZVP; 금 클러스터는 ORCA로), 여기 파장 의존성 또는 전기화학 SERS.

## 8. 발표 자료 확인 사항
- 포스터 박스플롯 라벨: "Pristine" 상자 값이 예전 그림의 indole, "Indole 1" 상자 값이 blank(MBA·MPY 둘 다). 라벨이 바뀐 것인지 슬라이드 12와 인쇄본에서 확인.
- 슬라이드 20 Kwon 비교: Mixtures 열은 "Quantified, not explained"(그 논문은 혼합물 조성·농도 정량함). 센서 수는 약 100개.
- 혼합물 캡션: "Non-odorant model three-component mixture: a first step before odorant mixtures".
