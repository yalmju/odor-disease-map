# OPTICAL NOSE
### Chemically interpretable multi-receptor plasmonic array for disease-relevant odorant fingerprinting

> Built a SERS sensing platform that reads an odorant as a *named* chemical fingerprint rather than an anonymous pattern — three reporter chemistries on one plasmonic hydrogel array, read as spectral maps, with every discriminating band traceable to a vibrational mode. Paired it with a reproducible computational pipeline that places disease-linked volatiles on a learned odor-perception map, and showed that an odorant mixture is not the linear sum of its components.

---

## The problem

| Pain point | Why it mattered |
|---|---|
| **Pattern without chemistry** | Both the human nose and today's electronic noses identify a smell from a pattern across cross-reactive channels. Neither says *which molecule* produced it, so a positive call cannot be audited. |
| **Chemistry without a pattern** | GC–MS identifies molecules but returns a peak list, not the combinatorial response that makes odor discrimination work — and it is not portable. |
| **Structure does not predict odor** | Indole and skatole differ by one methyl group; molecules with near-identical scaffolds can be odorless or floral. Structural similarity is a poor proxy for perceptual similarity, so a structure-only readout misidentifies the cases that matter. |
| **Mixtures are assumed additive** | Real breath and headspace samples are mixtures. Treating a mixture as the sum of its pure components is convenient and, as this work shows, wrong. |
| **Unaudited claims** | Odor–disease associations are routinely reported without separating database annotation from model prediction, or stating what the evidence cannot support. |

---

## What I built

- **Multi-receptor plasmonic array.** Gold nanoparticles dispersed through a mucus-like polysaccharide hydrogel film, functionalised with three reporter monolayers chosen for *orthogonal* interaction chemistry — 4-MBA (charge transfer), 4-MPY (hydrogen bonding), 2-NAT (π–π stacking). The channels are designed to disagree, rather than being cross-reactive by accident.
- **Spectral-map readout.** Each exposure is measured as a spatial SERS map per channel, so a response is a distribution over sampled positions rather than a single number, and band ratios — not bare intensities — carry the signal.
- **Interpretable discrimination.** Discriminating features are specific Raman bands with assignable vibrational origin, so a classification can be explained in chemical terms instead of as a latent-space distance.
- **Mixture analysis.** Showed that a binary odorant mixture does not reproduce either pure component's channel pattern, attributed it to competitive adsorption at the reporter surface, and quantified the departure from additivity by non-negative least-squares unmixing against pure-component spectra.
- **Odor-perception map.** Placed disease-linked volatiles on a 2-D projection of frozen pretrained OpenPOM graph-neural-network embeddings (532 molecules, 256-D, cosine t-SNE), overlaying curated odor labels and archived disease annotations as *separate evidence layers* that are never mixed into a single score.
- **Reproducible research pipeline.** ~2,240 lines of Python across 17 modules: frozen-encoder inference with checkpoint hash verification, leakage-controlled model training, annotation-enrichment statistics with multiple-testing correction, attribution and removal-sensitivity analysis, and a figure renderer that emits a manifest recording input SHA-256, counts, styling parameters and package versions.
- **Limitation discipline as a feature.** Every module docstring and output record states what the result is *not*: not a disease classifier, not a validated SERS model, not calibrated confidence, not evidence of absence. Scope statements are emitted into the result JSON, not left in prose.

---

## Key results

| Area | Result |
|---|---|
| **Channel orthogonality** | The three reporters respond to the same odorant in different directions and magnitudes, so the response is a genuine three-channel signature rather than three copies of one measurement. |
| **Odorant discrimination** | Indole, skatole and nonanal separate on the three-channel pattern. Indole and skatole differ by a single methyl group, which is the hard case. |
| **Mixture non-additivity** | An indole + skatole mixture reproduces neither pure pattern: channels that rose for the single components move toward or past zero, and non-negative least-squares assigns an indole share of zero in the charge-transfer channel. |
| **Named evidence** | Indole marker bands at 1340 and 1230 cm⁻¹ disappear in the mixture, giving a chemical explanation for the pattern change rather than an unexplained shift. |
| **Spatial sampling** | Each channel–exposure condition is sampled at roughly 100 map positions, so responses are reported as distributions with their own spread rather than single readings. |
| **Computational reproducibility** | Fresh encoder inference reproduced the archived embedding cache to a maximum absolute difference of 1.6 × 10⁻⁶; checkpoint integrity is verified by SHA-256 before any inference runs. |
| **Leakage control** | Splits are composition-disjoint by molecular identity, with an independent indole-family holdout trained on all non-indole records, so related structures cannot leak between train and test. |
| **Dissemination** | Featured Poster Presentation at µTAS 2026 (T032.b). Figures export to PNG (300 dpi), SVG and PDF with a provenance manifest. |

---

## CV entries (pick by space)

**One line**
> Built a chemically interpretable optical nose — a three-reporter plasmonic SERS array whose discriminating features are named Raman bands — and showed that odorant mixtures depart measurably from additivity; presented at µTAS 2026.

**Three bullets**
- Designed and built a **multi-receptor plasmonic hydrogel SERS array** with three orthogonal reporter chemistries (charge transfer, hydrogen bonding, π–π stacking), read as spatial spectral maps, that discriminates odorants differing by a **single methyl group** while keeping every discriminating feature traceable to an assignable vibrational band.
- Demonstrated that a binary odorant mixture **is not the linear sum of its components** — competitive adsorption drives channel responses past zero and removes indole marker bands at 1340 / 1230 cm⁻¹ — establishing that mixture composition must be learned from mixtures rather than unmixed from pure spectra.
- Built the accompanying **reproducible research pipeline** (~2,240 lines, 17 modules): frozen graph-neural-network inference with checkpoint-hash verification reproducing the archived cache to 1.6 × 10⁻⁶, leakage-controlled splits, enrichment statistics with multiple-testing correction, and figure rendering with a full provenance manifest.

**Full project entry**
> **Chemically Interpretable Optical Nose — Multi-Receptor Plasmonic Arrays for Odorant Fingerprinting** | SERS, Python, PyTorch, scikit-learn | 2025–2026
> - Built a plasmonic hydrogel array carrying three reporter monolayers selected for orthogonal interaction chemistry, read as per-channel spatial SERS maps (~100 sampled positions per condition), so each odorant produces a three-channel signature whose discriminating features are named Raman bands rather than latent-space coordinates.
> - Discriminated indole, skatole and nonanal — the first two differing by one methyl group — and showed their binary mixture reproduces neither pure pattern, attributing the departure to competitive adsorption and quantifying it by non-negative least-squares unmixing.
> - Placed disease-linked volatiles on a learned odor-perception map (frozen OpenPOM 256-D embeddings, 532 molecules, cosine t-SNE), keeping curated odor labels and archived disease annotations as separate evidence layers that are never combined into a single score.
> - Engineered the supporting pipeline for auditability: checkpoint SHA-256 verification, embedding reproduction to 1.6 × 10⁻⁶, composition-disjoint splits with an independent structural-family holdout, Benjamini–Hochberg-corrected enrichment, and a render manifest recording input hashes and package versions.
> - Presented as a Featured Poster at µTAS 2026; supported by NRF grant RS-2025-00560226.

---

## 국문 이력서용

**화학적으로 해석 가능한 광학 후각 센서: 다중 리포터 플라스모닉 어레이 기반 냄새 분자 지문화** (2025–2026)

- 상호작용 화학이 서로 다른 리포터 3종(전하이동·수소결합·π–π 스택킹)을 하나의 플라스모닉 하이드로겔 어레이에 구성, 채널별 **공간 SERS 맵**(조건당 약 100개 측정 위치)으로 읽어 냄새 분자당 3채널 서명을 획득
- 메틸기 하나 차이인 **인돌과 스카톨을 구분**하고, 판별 근거를 잠재 공간 거리가 아니라 **귀속 가능한 라만 밴드**로 제시 — 블랙박스가 아닌 해석 가능한 판별
- 이성분 혼합물이 순수 성분의 선형 합이 **아님**을 실증: 경쟁 흡착으로 채널 응답이 역전되고 인돌 지표 밴드(1340 / 1230 cm⁻¹)가 소실, NNLS 분해에서 전하이동 채널의 인돌 기여도 0
- 질병 연관 휘발성 분자를 **학습된 후각 지각 공간**(동결 OpenPOM 256차원 임베딩, 532분자, 코사인 t-SNE)에 배치하고, 향 주석과 질병 주석을 **별도 근거 층**으로 유지해 단일 점수로 혼합하지 않도록 설계
- 재현성 파이프라인 구축(약 2,240행, 17개 모듈): 체크포인트 SHA-256 검증, 임베딩 재현 오차 1.6 × 10⁻⁶, 조성 분리 분할과 구조 계열 독립 홀드아웃, 다중검정 보정 농축 분석, 입력 해시·패키지 버전을 기록하는 렌더 매니페스트
- µTAS 2026 Featured Poster 발표 (T032.b), 한국연구재단 RS-2025-00560226 지원

---

## Notes before you send this

- **Verify the measurement numbers before use.** The band assignments (1340 / 1230 cm⁻¹), the per-channel response percentages, the NNLS indole share and the ~100 positions per condition are taken from the presentation deck, not from a file in this repository. The repository holds only the computational side. Confirm each against your measurement records before putting them in a CV.
- **Numbers that are verifiable here.** The 1.6 × 10⁻⁶ embedding reproduction and the "not implemented" status of a SERS map model come from `odor-disease-maps-v2/docs/VALIDATION.json`. The 532 molecules and t-SNE settings come from `docs/METHODS.md`. The ~2,240 lines and 17 modules are a count of the current tree.
- **Do not claim what the project's own documents refuse.** `cli.py` records "No validated SERS or clinical model"; `METHODS.md` records "No SERS raw data or prospective spatial-map validation are included" and that nearby map points do not prove perceptual equivalence. Keep classification framed as *molecular* discrimination, never diagnostic performance.
- **The mixture classifier is not described above.** You mentioned an MLP mixture classifier that is not in this repository. If you want it in the CV, add its architecture, input features, split policy and accuracy — and state whether the split is spatially independent. If pixels from one map appear in both training and test, the accuracy cannot be called held-out, and a reviewer will ask.
- **"Designed and built" rather than "sole developer"**, matching your PLASTIQ entry, keeps the claim safe whichever way the authorship question is asked.
- **Collaborators and institution names** are left out. Add them only if the project allows it.
- **Positioning.** The honest differentiator here is *interpretability under stated limits*, not raw accuracy — the discriminating features are named bands and the documentation refuses to overclaim. For research or regulated-instrument roles that is the strongest framing. For a sensor-engineering role, lead instead with the orthogonal-reporter array design and the mixture result.
- **Pairs well with PLASTIQ.** Both are Raman spectral-map platforms; together they read as a line of work on turning hyperspectral maps into auditable chemical evidence. If both appear on one CV, say so in a single connecting clause rather than repeating the Raman-map description twice.
