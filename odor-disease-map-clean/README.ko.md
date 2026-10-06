# 향·질병 분자 지도

고정된 OpenPOM 분자 좌표에 알려진 향과 HMDB 질병 주석을 연결하는 연구 코드입니다.

- `scripts/`: 지도 재현과 원래 임베딩 공간에서의 후보 비교
- `research/`: 기존 OpenPOM 추론·지각평가 MLP 학습 코드와 혼합물 거리 연구 프로토타입
- `data/demo.json`: 가상 예제이며 실제 분자 연구 결과가 아닙니다.

실행 명령과 환경은 [영문 README](README.md)를 참고하세요. 실제 좌표·주석·모델 가중치는 별도 준비가 필요합니다. `private_data/`와 `results/`는 Git에서 제외됩니다.

현재 SERS 공간맵 분류 모델과 향–SERS 대응 학습 모델은 포함되어 있지 않습니다. Windows CPU 추론은 검증했고 Apple Silicon 실기기 검증은 남아 있습니다.
