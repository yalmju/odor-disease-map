# ODOR MAP / OPTICAL NOSE
### Mapping the chemical space of disease-relevant odorants, and making their odor patterns visible

> Two halves of one question: *where* do disease-relevant volatiles sit in the space of smells, and *what do they look like* when you measure them. Placed 532 odorants in a learned odor-perception space and overlaid archived disease links as a separate evidence layer; then built a multi-receptor plasmonic array that renders a single odorant as a spatial Raman pattern you can look at side by side. An abstract coordinate becomes an image with nameable bands.

---

## The problem

| Pain point | Why it mattered |
|---|---|
| **Chemical space is large and sparsely labelled** | Tens of thousands of volatiles exist; only a fraction carry curated odor annotations and fewer carry disease links. Without a shared coordinate system there is no way to ask where a disease's volatiles actually sit, or whether they cluster at all. |
| **Structure is a poor proxy for odor** | Indole and skatole differ by one methyl group. Elsewhere a near-identical scaffold is odorless while its neighbour is floral. A map built on structural fingerprints puts the wrong molecules together, so the space has to be organised by *perception*, not formula. |
| **"Disease odor" is treated as a single marker** | Reporting one biomarker volatile implies a single note. Whether a disease's volatiles concentrate in one region of odor space or scatter across it is an empirical question that had no picture. |
| **A smell has no image** | Odor is the one major sense with no standard visual representation. Chromatography gives a peak list, an electronic nose gives a bar pattern with no chemistry attached — neither is something you can put next to another and *see* the difference. |
| **Evidence layers get conflated** | Curated odor labels, model-predicted descriptors and database disease links are different kinds of evidence. Collapsing them into one score makes a figure that cannot be audited. |

---

## What I built

### Chemical space

- **A shared coordinate system for odorants.** Placed 532 molecules in a 2-D projection of frozen pretrained OpenPOM graph-neural-network embeddings (256-D, cosine t-SNE), so position reflects *learned odor perception* rather than structural similarity — the distinction the indole/skatole case demands.
- **Separate evidence layers over one geometry.** Curated odor annotations and archived disease links are rendered as independent overlays on identical coordinates, never fused into a single score. Position comes from the model; colour comes from the database; the two are documented as non-interchangeable.
- **Where disease volatiles actually sit.** Overlaid disease-linked molecules on the map to show the distribution rather than assert it — whether they cluster in one odor region or span many.
- **Attribution and sensitivity analysis.** Decomposed the annotation contrast by structural family, with leave-out sensitivity (remove skatole, remove the whole indole family) and Benjamini–Hochberg-corrected enrichment, so a visible contrast is traced to the molecules producing it.

### Odor pattern visualization

- **An odorant rendered as an image.** Built a plasmonic hydrogel array carrying three reporter monolayers chosen for *orthogonal* interaction chemistry — 4-MBA (charge transfer), 4-MPY (hydrogen bonding), 2-NAT (π–π stacking) — and read each exposure as a spatial SERS map per channel. One molecule becomes a three-channel picture sampled at roughly 100 positions, not a single reading.
- **Patterns that can be compared by eye and by band.** Because the channels are designed to disagree, different odorants produce visibly different maps; because the features are Raman bands, the difference can also be stated chemically rather than as a latent-space distance.
- **Mixtures made visible.** Showed that a binary mixture produces a pattern resembling neither pure component — attributed to competitive adsorption at the reporter surface and quantified by non-negative least-squares unmixing against pure-component spectra. The non-additivity is something you can see in the maps, not an inference from a residual.

### Making it auditable

- **Reproducible pipeline.** ~2,240 lines of Python across 17 modules: frozen-encoder inference with checkpoint SHA-256 verification, leakage-controlled splits, enrichment statistics with multiple-testing correction, and a figure renderer emitting a manifest of input hashes, counts, styling parameters and package versions.
- **Scope statements as output, not prose.** Each module writes what its result is *not* — not a disease classifier, not calibrated confidence, not evidence of absence — into the result JSON alongside the numbers.

---

## Key results

| Area | Result |
|---|---|
| **A map of odor space** | 532 molecules placed on one fixed set of coordinates, with six odor-quality facets and a disease-annotation overlay rendered on identical geometry. |
| **Disease odor is distributed, not a single note** | Disease-linked molecules span multiple odor qualities across the map rather than concentrating in one region — an argument for multi-channel readout over single-biomarker detection. |
| **Perception ≠ structure, shown on the map** | Molecules a single substitution apart sit as neighbours in the learned space, making explicit why a structure-only representation would misgroup exactly the cases of interest. |
| **An odorant has a picture** | Each odorant yields a distinct three-channel spatial pattern at ~100 sampled positions per condition, so responses are distributions with visible spread rather than single values. |
| **Mixtures visibly break additivity** | A binary mixture reproduces neither pure pattern; channels reverse direction and marker bands of one component disappear, with NNLS assigning that component a zero share in the charge-transfer channel. |
| **Named features** | Discriminating features are specific Raman bands with assignable vibrational origin, so a visual difference carries a chemical explanation. |
| **Reproducibility** | Fresh encoder inference reproduced the archived embedding cache to a maximum absolute difference of 1.6 × 10⁻⁶; checkpoint integrity verified by hash before any inference. |
| **Leakage control** | Composition-disjoint splits by molecular identity plus an independent structural-family holdout trained on all non-indole records. |
| **Dissemination** | Featured Poster Presentation, µTAS 2026 (T032.b). Figures export to PNG (300 dpi), SVG and PDF with a provenance manifest. |

---

## CV entries (pick by space)

**One line**
> Mapped 532 odorants in a learned odor-perception space with disease annotations as a separate evidence layer, and built a three-reporter plasmonic SERS array that renders each odorant as a spatial pattern with nameable Raman bands — making odor space navigable and individual smells visible.

**Three bullets**
- Built a **shared coordinate system for odor space**: 532 molecules projected from frozen pretrained graph-neural-network embeddings, with curated odor labels and archived disease links overlaid as **independent evidence layers** on identical geometry, showing that disease-linked volatiles span many odor qualities rather than forming a single marker note.
- Designed a **multi-receptor plasmonic SERS array** with three orthogonal reporter chemistries that turns one odorant into a three-channel spatial map (~100 sampled positions per condition) — giving a smell a visual representation whose discriminating features are **assignable Raman bands**, not latent-space coordinates.
- Showed that a binary odorant mixture **produces a pattern resembling neither component**, attributed it to competitive adsorption and quantified it by non-negative least-squares; built the supporting reproducible pipeline (~2,240 lines, 17 modules) with checkpoint-hash verification reproducing archived embeddings to 1.6 × 10⁻⁶.

**Full project entry**
> **Odor Map / Optical Nose — Chemical-Space Mapping and Visualization of Odor Patterns** | SERS, Python, PyTorch, scikit-learn | 2025–2026
> - Placed 532 odorants on one fixed set of coordinates derived from frozen pretrained OpenPOM embeddings (256-D, cosine t-SNE), organising the space by learned perception rather than structure — the distinction required when single-substitution pairs differ in odor.
> - Overlaid curated odor annotations and archived disease links as separate layers on identical geometry, never fused into one score, and showed disease-linked volatiles spanning multiple odor qualities instead of clustering as a single marker.
> - Built a three-reporter plasmonic hydrogel array (charge transfer / hydrogen bonding / π–π stacking) read as per-channel spatial SERS maps, rendering each odorant as a comparable image whose discriminating features are named vibrational bands.
> - Demonstrated mixture non-additivity: a binary mixture reproduced neither pure pattern, with marker-band loss and a zero NNLS share in one channel, establishing that composition must be learned from mixtures rather than unmixed from pure spectra.
> - Engineered the pipeline for auditability — checkpoint SHA-256 verification, embedding reproduction to 1.6 × 10⁻⁶, composition-disjoint splits with a structural-family holdout, BH-corrected enrichment, and a render manifest of input hashes and package versions.
> - Featured Poster, µTAS 2026; supported by NRF grant RS-2025-00560226.

---

## 국문 이력서용

**냄새 화학 공간 지도화 및 냄새 패턴 시각화: 다중 리포터 플라스모닉 어레이 기반 광학 후각 센서** (2025–2026)

- **냄새 화학 공간의 좌표계 구축**: 동결된 사전학습 그래프 신경망 임베딩(256차원)을 코사인 t-SNE로 투영해 분자 532종을 하나의 고정 좌표에 배치, 구조 유사도가 아닌 **학습된 후각 지각**을 기준으로 공간을 조직 (메틸기 하나 차이로 향이 갈리는 사례가 구조 기반 표현으로는 잘못 묶이는 문제를 해결)
- **근거 층 분리 설계**: 향 주석과 질병 주석을 동일 좌표 위의 **독립적인 오버레이**로 유지해 단일 점수로 합치지 않음. 위치는 모델, 색은 데이터베이스에서 오며 상호 교환 불가임을 문서화
- 질병 연관 휘발성 분자가 **하나의 향 영역에 모이지 않고 여러 향 특성에 걸쳐 분포**함을 지도 위에서 제시 — 단일 바이오마커가 아니라 다채널 판독이 필요한 근거
- **냄새 패턴의 시각화**: 상호작용 화학이 서로 다른 리포터 3종(전하이동·수소결합·π–π 스택킹)을 올린 플라스모닉 하이드로겔 어레이를 채널별 **공간 SERS 맵**으로 판독, 분자 하나를 조건당 약 100개 측정 위치의 3채널 이미지로 변환. 판별 근거는 잠재 공간 거리가 아닌 **귀속 가능한 라만 밴드**
- **혼합물의 비선형성 가시화**: 이성분 혼합물이 어느 순수 성분 패턴도 재현하지 않음을 확인, 경쟁 흡착으로 해석하고 NNLS 분해로 정량 — 잔차 추론이 아니라 맵에서 직접 보이는 결과
- 재현성 파이프라인 구축(약 2,240행, 17개 모듈): 체크포인트 SHA-256 검증, 임베딩 재현 오차 1.6 × 10⁻⁶, 조성 분리 분할과 구조 계열 독립 홀드아웃, 다중검정 보정 농축 분석, 입력 해시·패키지 버전 기록 매니페스트
- µTAS 2026 Featured Poster 발표 (T032.b), 한국연구재단 RS-2025-00560226 지원

---

## Notes before you send this

- **Why this framing.** The entries above are built on two verbs — *mapping* chemical space and *visualizing* odor patterns — rather than on sensor discrimination performance. That matches what the work actually established and avoids an accuracy claim the project's own documents refuse. It also makes the two halves read as one project instead of a computational piece bolted to an experimental one.
- **Verify the measurement numbers before use.** The band assignments, per-channel response percentages, the NNLS zero share and the ~100 positions per condition come from the presentation deck, not from a file in this repository, which holds only the computational side. Confirm each against your measurement records.
- **Numbers verifiable here.** The 1.6 × 10⁻⁶ embedding reproduction and the "not implemented" status of a SERS map model are in `odor-disease-maps-v2/docs/VALIDATION.json`. The 532 molecules and t-SNE settings are in `docs/METHODS.md`. The ~2,240 lines and 17 modules are a count of the current tree.
- **One number to fill in.** The entries say disease-linked volatiles "span many odor qualities" without a count. The deck cites 38 colorectal-cancer-linked molecules. Add the figure once you have confirmed it, and name the disease — "38 volatiles linked to one disease, spanning six odor qualities" is far stronger than the qualitative phrasing.
- **Do not claim what the project's documents refuse.** `cli.py` records "No validated SERS or clinical model"; `METHODS.md` records that no prospective spatial-map validation is included and that nearby map points do not prove perceptual equivalence. Keep any discrimination framed as *molecular*, never diagnostic.
- **The MLP mixture classifier is not included.** It is not in this repository. To add it, supply architecture, input features, split policy and accuracy — and state whether the split is spatially independent. If pixels from one map appear in both training and test, the accuracy is not held-out and a reviewer will ask.
- **"Designed and built" rather than "sole developer"**, matching your PLASTIQ entry.
- **Collaborator and institution names** are left out. Add them only if the project allows it.
- **Pairs with PLASTIQ.** Both turn Raman hyperspectral maps into auditable chemical evidence; this one adds the chemical-space side. If both appear on one CV, connect them in a single clause rather than describing Raman mapping twice.
