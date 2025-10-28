
# POS ECOD Anomaly Detection (with Mock Data)

PyOD + ECOD 기반의 POS 하드웨어 이상 탐지 에이전트 MVP  
실시간(psutil) 수집 또는 `mock/*.csv` 재생으로 테스트할 수 있습니다.

## 구성
```
pos-ecod/
├─ mock/
│  ├─ pos_hw_metrics_normal.csv     # 정상 상태 (1시간)
│  ├─ pos_hw_metrics_anomaly.csv    # 이상 상태 (45분 이후 열화)
│  └─ README.md                     # 데이터 컬럼 설명
├─ src/pos_ecod/
│  ├─ agent.py                      # ECOD 기반 탐지기 (--mock 지원)
│  └─ __init__.py
├─ pyproject.toml
└─ README.md
```

## 실행 (uv)
```bash
uv venv
source .venv/bin/activate   # Windows PowerShell: .venv\Scripts\Activate.ps1
uv pip install -e .

# 1) Mock 데이터로 테스트
uv run pos-ecod --mock mock/pos_hw_metrics_normal.csv
uv run pos-ecod --mock mock/pos_hw_metrics_anomaly.csv

# 2) 실시간 수집(psutil)
uv run pos-ecod

# 옵션
uv run pos-ecod --interval 15 --window 5 --baseline 60 --threshold_pct 98 --sustain 6/10
uv run pos-ecod --demo   # 가짜 이상치 주입
```

## Mock 재생을 빠르게 보는 법
- `--mock`가 지정되면 기본 대기 간격은 `--mock-interval`(기본 0.01초, env `PE_MOCK_INTERVAL`)을 사용합니다.
- 실시간 간격을 유지하고 싶다면 `PE_MOCK_REALTIME=1`을 설정하세요.
- 예시
  - 빠른 재생: `uv run pos-ecod --mock mock/pos_hw_metrics_anomaly.csv`
  - 더 빠르게: `uv run pos-ecod --mock mock/pos_hw_metrics_anomaly.csv --mock-interval 0.001`
  - 실시간 유지: `PE_MOCK_REALTIME=1 uv run pos-ecod --interval 15 --mock mock/pos_hw_metrics_anomaly.csv`

## Mock 길이에 따른 baseline 자동 조정
- mock CSV 전체 길이와 `--window`를 바탕으로 가능한 윈도 개수를 계산합니다.
- `--baseline`이 가능한 윈도 개수보다 크면, mock에서도 이상 탐지까지 진행되도록 baseline을 자동으로 축소합니다(대략 절반).
- 조정 시 콘솔에 경고가 출력됩니다. 필요하면 명시적으로 `--baseline`을 지정해 덮어쓸 수 있습니다.

## 이상 감지 알림 출력
- ECOD baseline 학습 완료: `ECOD baseline fitted. threshold (p98.0) = <값>`
- 일시적 이상: `[ANOMALY DETECTED] score=<s> thr=<thr> sustain=X/Y`
- 지속 이상(경보): `[ANOMALY SUSTAINED] …` 이후 JSON ALERT가 출력됩니다.
- ALERT 페이로드 예시
```json
{
  "event": "POS_HW_ANOMALY",
  "score": 33.673,
  "threshold": 18.5067,
  "sustained": "10/10",
  "top_features": ["cpu_temp_c_max", "mem_used_pct_p95", "disk_await_ms_p95"],
  "feature_snapshot": {"cpu_temp_c_max": 86.4, "mem_used_pct_p95": 87.5, "disk_await_ms_p95": 16.0},
  "ts": 1730,
  "mock_path": "mock/pos_hw_metrics_anomaly.csv"
}
```

## 주요 옵션 요약
- `--interval`: 실시간 수집 간격(초). 기본 15
- `--window`: 집계 윈도 크기(샘플 수). 기본 5
- `--baseline`: baseline 윈도 개수. 기본 60 (mock 길이에 따라 자동 축소 가능)
- `--threshold_pct`: ECOD 스코어 임계값 백분위. 기본 98.0
- `--sustain N/M`: 이상 지속 조건. 기본 6/10
- `--mock PATH`: CSV 재생 모드
- `--mock-interval`: mock 재생 시 루프 대기(초). 기본 0.01
- `--demo`: 데모용 이상치 노이즈 주입

### 환경변수
- `PE_INTERVAL`, `PE_WINDOW`, `PE_BASELINE`, `PE_THR_PCT`, `PE_SUSTAIN`
- `PE_MOCK_INTERVAL`: mock 대기 시간(초)
- `PE_MOCK_REALTIME=1`: mock에서도 실시간 간격 강제

## 빠른 데모 조합 예시
- 빠르게 ALERT 보기: `uv run pos-ecod --mock mock/pos_hw_metrics_anomaly.csv --sustain 2/3`
- 더 짧은 baseline/window: `uv run pos-ecod --mock mock/pos_hw_metrics_anomaly.csv --baseline 20 --window 3`

## 수집/특징 항목 설명
아래 항목들을 샘플 단위로 수집한 뒤, 윈도(window) 단위로 집계하여 ECOD에 입력합니다.

**Raw 샘플 메트릭**
- `cpu_total_pct`: 전체 CPU 사용률(%) — `psutil.cpu_percent`. 높은 값은 부하를 의미합니다. src/pos_ecod/agent.py:100
- `cpu_iowait_pct`: CPU가 I/O 대기 중인 시간 비율(%) — 디스크 병목 신호. src/pos_ecod/agent.py:101
- `mem_used_pct`: 물리 메모리 사용률(%) — 메모리 압박 신호. src/pos_ecod/agent.py:102
- `swap_used_pct`: 스왑 사용률(%) — 스왑 사용 증가 시 지연 위험. src/pos_ecod/agent.py:103
- `cpu_temp_c`: CPU 온도(섭씨) — 과열 시 성능 저하/쓰로틀링 가능. src/pos_ecod/agent.py:104
- `disk_await_ms`: 디스크 I/O 평균 대기시간(ms/IO) — psutil 디스크 카운터로 추정. src/pos_ecod/agent.py:105
- `fs_used_pct_root`: 루트 파일시스템 사용률(%) — 가용 공간 부족 위험. src/pos_ecod/agent.py:106
- `nic_err_rate`: 네트워크 인터페이스 오류율(비율) — 현재 MVP에선 placeholder. src/pos_ecod/agent.py:107
- `load1`: 시스템 1분 load average — CPU 대기열의 대략적 지표. src/pos_ecod/agent.py:108

참고: mock CSV에는 위 항목 대부분이 포함되어 있으며(`mock/README.md`), `load1`은 mock 재생 시 0으로 채웁니다.

**윈도 집계 특징(ECOD 입력)**
- `cpu_total_pct_p95`: 샘플들의 95퍼센타일 CPU 사용률 — 스파이크 내성 있도록. src/pos_ecod/agent.py:131
- `cpu_iowait_pct_p95`: 95퍼센타일 I/O wait — 간헐적 I/O 병목 반영. src/pos_ecod/agent.py:132
- `mem_used_pct_p95`: 95퍼센타일 메모리 사용률 — 순간 치솟음 반영. src/pos_ecod/agent.py:133
- `swap_used_pct_p95`: 95퍼센타일 스왑 사용률 — 스왑 급증 포착. src/pos_ecod/agent.py:134
- `load1_max`: 윈도 내 최대 load1 — 부하 피크 반영. src/pos_ecod/agent.py:135
- `cpu_temp_c_max`: 윈도 내 최대 CPU 온도 — 과열 피크 포착. src/pos_ecod/agent.py:136
- `disk_await_ms_p95`: 95퍼센타일 디스크 대기시간 — tail latency에 민감. src/pos_ecod/agent.py:137
- `fs_used_pct_root_p95`: 95퍼센타일 루트 FS 사용률 — 지속적 공간 압박. src/pos_ecod/agent.py:138
- `nic_err_rate_max`: 윈도 내 최대 NIC 오류율 — 간헐적 네트워크 이슈. src/pos_ecod/agent.py:139

**파생 특징(윈도 기반)**
- `io_saturation`: `min(1.0, disk_await_ms_p95 / 100.0)` — 디스크 포화 근사(1.0은 강한 포화). src/pos_ecod/agent.py:141
- `thermal_pressure`: `max(0, (cpu_temp_c_max - 70) / 15)` — 70℃ 이상 온도 초과분을 압력으로 환산. src/pos_ecod/agent.py:145

이 집계/파생 특징들의 벡터를 표준화(`StandardScaler`) 후 ECOD에 입력하여 스코어를 계산합니다. baseline 구간 스코어의 `threshold_pct` 백분위를 임계값으로 사용하여 이상 여부를 판정합니다. src/pos_ecod/agent.py:167
