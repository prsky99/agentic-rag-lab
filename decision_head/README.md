# decision_head — 자체 결정 헤드 (5장)

베이스 모델을 동결하고 출력층의 A~Z 문자 행만 학습해, 4장 게이트가 쓰는 `/v1/systemone` 계약으로 서빙합니다.

## 준비
- 추가 의존성: `pip install torch transformers` (저장소 `requirements.txt`에는 없음)
- 데이터: 5.4절 형식의 JSONL(`state`/`question`/`answer`)을 **학습·보정·평가 세 파일로 미리 분리**
  - 예: `data/train.jsonl`, `data/calib.jsonl`(최소 200건), `data/test.jsonl`
- 베이스: 문자 A~Z가 각각 단일 토큰인 모델(학습 시작 시 자동 검사). 상업 이용 시 베이스 라이선스 확인

## 실행 순서 (저장소 루트에서)
1. 학습 — `python -m decision_head.train --base Qwen/Qwen3-1.7B --data data/train.jsonl --out artifacts/head.pt`
2. 온도 보정 — `python -m decision_head.calibrate --artifact artifacts/head.pt --data data/calib.jsonl`
   (타입별 T를 아티팩트에 기록, 보정 전후 ECE 출력)
3. 평가 — `python -m decision_head.evaluate --artifact artifacts/head.pt --data data/test.jsonl`
   (5.7절 기준: 다수 클래스 대비 +10포인트, ECE ≤ 0.05, 신뢰도 0.9 이상 구간 정확도 ≥ 0.85)
4. 서빙 — `DECISION_HEAD_BASE=Qwen/Qwen3-1.7B DECISION_HEAD_ARTIFACT=artifacts/head.pt uvicorn decision_head.server:app --port 3000`
5. 게이트 연결 — 하네스 `domains/rag/harness/config.yaml`의 `generation.decision_base_url`을 `http://<호스트>:3000`으로 지정(게이트 코드 변경 없음)

## 파일
| 파일 | 내용 | 절 |
|---|---|---|
| `train.py` | 데이터 로드·프롬프트 렌더·문자 행 학습·아티팩트 저장 | 5.4 |
| `calibrate.py` | 타입별 온도 적합(`fit_temperature`)·아티팩트 갱신 | 5.5 |
| `metrics.py` | ECE·신뢰도 표 | 5.5 |
| `server.py` | `/v1/systemone`, `/v1/version` | 5.6 |
| `evaluate.py` | 정확도·다수 기준선·macro-F1·ECE 보고 | 5.7 |
