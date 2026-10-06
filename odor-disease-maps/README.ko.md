# 향·질병 분자 지도 재현 패키지

현재 포스터의 향 6종 지도와 질병 주석 산점도를 그리는 코드입니다. **OpenPOM 학습이나 t-SNE 재계산을 수행하지 않고 저장된 좌표를 재사용합니다.**

## 실행

```sh
python -m pip install -r requirements.txt
python scripts/render_maps.py --input data/demo.json --output results/demo --disease "Example disease"
```

예제는 가상 데이터이며 연구 결과가 아닙니다. 개인 재현용 ZIP의 `evidence_overlay.json`을 `private_data/`에 넣으면 실제 그림을 재현합니다.

```sh
python scripts/render_maps.py --input private_data/evidence_overlay.json --output results/research --font Arial
```

Arial 설치가 없으면 기본 DejaVu Sans를 사용하세요. Windows에서 실행 검증했으며 Apple Silicon 실제 실행은 아직 검증하지 않았습니다. GPU는 필요하지 않습니다.

## GitHub에 올릴 파일

공개용 ZIP만 업로드하세요. 개인 재현용 ZIP은 별도 보관합니다. 코드 라이선스는 아직 지정하지 않았고 DB 주석의 재배포 허가도 확인 전입니다. `.gitignore`가 private_data와 생성 결과를 제외합니다. 현재 GitHub 저장소 생성·push는 하지 않았습니다.

지도 좌표는 고정된 OpenPOM 임베딩의 t-SNE, 향 색은 Leffingwell 기반 주석, 질병 색은 보관된 HMDB 주석입니다. 서로 다른 근거이므로 모델 향 점수와 혼합해서 해석하지 않습니다. 자세한 방법과 한계는 영문 README 및 docs를 확인하세요.
