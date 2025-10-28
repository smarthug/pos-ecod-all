
import argparse
import os
import time
import math
import json
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psutil
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from sklearn.preprocessing import StandardScaler
from pyod.models.ecod import ECOD

console = Console()

# ---------- Disk IO helpers ----------

@dataclass
class IOStatSnapshot:
    ts: float
    read_count: int
    write_count: int
    read_time_ms: int
    write_time_ms: int

def _io_snapshot() -> IOStatSnapshot:
    io = psutil.disk_io_counters()
    return IOStatSnapshot(
        ts=time.time(),
        read_count=io.read_count,
        write_count=io.write_count,
        read_time_ms=io.read_time,
        write_time_ms=io.write_time,
    )

def _await_ms(prev: IOStatSnapshot, cur: IOStatSnapshot) -> float:
    d_ops = (cur.read_count - prev.read_count) + (cur.write_count - prev.write_count)
    d_time = (cur.read_time_ms - prev.read_time_ms) + (cur.write_time_ms - prev.write_time_ms)
    if d_ops <= 0:
        return 0.0
    return max(0.0, d_time / d_ops)

# ---------- Collectors ----------

def collect_once(prev_io: Optional[IOStatSnapshot]) -> Tuple[Dict[str, float], IOStatSnapshot]:
    # CPU
    cpu_times = psutil.cpu_times_percent(interval=None)
    cpu_total = psutil.cpu_percent(interval=None)
    cpu_iowait = getattr(cpu_times, "iowait", 0.0)

    # MEM
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()

    # LOAD
    try:
        load1, load5, load15 = os.getloadavg()
    except OSError:
        load1 = load5 = load15 = 0.0

    # TEMP
    cpu_temp_c = -1.0
    try:
        temps = psutil.sensors_temperatures()
        for key in temps:
            entries = temps[key]
            if entries:
                vals = [e.current for e in entries if e.current is not None]
                if vals:
                    cpu_temp_c = float(max(vals))
                    break
    except Exception:
        pass

    # DISK await
    cur_io = _io_snapshot()
    if prev_io is None:
        await_ms = 0.0
    else:
        await_ms = _await_ms(prev_io, cur_io)

    # FS
    try:
        fs_used_pct = psutil.disk_usage("/").percent
    except Exception:
        fs_used_pct = 0.0

    # NET (placeholder in MVP)
    nic_err_rate = 0.0

    features = {
        "cpu_total_pct": float(cpu_total),
        "cpu_iowait_pct": float(cpu_iowait),
        "mem_used_pct": float(mem.percent),
        "swap_used_pct": float(swap.percent),
        "cpu_temp_c": float(cpu_temp_c),
        "disk_await_ms": float(await_ms),
        "fs_used_pct_root": float(fs_used_pct),
        "nic_err_rate": float(nic_err_rate),
        "load1": float(load1),
    }
    return features, cur_io

def next_mock_row(mock_df: pd.DataFrame, idx: int) -> Dict[str, float]:
    row = mock_df.iloc[idx]
    return {
        "cpu_total_pct": float(row["cpu_total_pct"]),
        "cpu_iowait_pct": float(row["cpu_iowait_pct"]),
        "mem_used_pct": float(row["mem_used_pct"]),
        "swap_used_pct": float(row["swap_used_pct"]),
        "cpu_temp_c": float(row["cpu_temp_c"]),
        "disk_await_ms": float(row["disk_await_ms"]),
        "fs_used_pct_root": float(row["fs_used_pct_root"]),
        "nic_err_rate": float(row["nic_err_rate"]),
        "load1": 0.0,
    }

# ---------- Feature Aggregation ----------

def aggregate_window(buf: List[Dict[str, float]]) -> Dict[str, float]:
    df = pd.DataFrame(buf)
    aggr = {
        "cpu_total_pct_p95": float(np.percentile(df["cpu_total_pct"], 95)),
        "cpu_iowait_pct_p95": float(np.percentile(df["cpu_iowait_pct"], 95)),
        "mem_used_pct_p95": float(np.percentile(df["mem_used_pct"], 95)),
        "swap_used_pct_p95": float(np.percentile(df["swap_used_pct"], 95)),
        "load1_max": float(df["load1"].max() if "load1" in df else 0.0),
        "cpu_temp_c_max": float(df["cpu_temp_c"].max()),
        "disk_await_ms_p95": float(np.percentile(df["disk_await_ms"], 95)),
        "fs_used_pct_root_p95": float(np.percentile(df["fs_used_pct_root"], 95)),
        "nic_err_rate_max": float(df["nic_err_rate"].max()),
    }
    aggr["io_saturation"] = float(min(1.0, (aggr["disk_await_ms_p95"] / 100.0)))
    therm = 0.0
    if aggr["cpu_temp_c_max"] > 0:
        therm = max(0.0, (aggr["cpu_temp_c_max"] - 70.0) / 15.0)
    aggr["thermal_pressure"] = float(therm)
    return aggr

# ---------- ECOD Wrapper ----------

class ECODDetector:
    def __init__(self, feature_names: List[str], baseline_size: int = 60, threshold_pct: float = 98.0):
        self.feature_names = feature_names
        self.baseline_size = baseline_size
        self.threshold_pct = threshold_pct
        self.scaler = StandardScaler()
        self.model = ECOD()
        self._fit_done = False
        self._baseline_rows: List[List[float]] = []
        self._score_threshold: Optional[float] = None

    def partial_fit_baseline(self, row: Dict[str, float]):
        self._baseline_rows.append([row[k] for k in self.feature_names])

    def ready(self) -> bool:
        return len(self._baseline_rows) >= self.baseline_size

    def fit(self):
        X = np.asarray(self._baseline_rows, dtype=float)
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs)
        scores = self.model.decision_function(Xs)
        self._score_threshold = float(np.percentile(scores, self.threshold_pct))

    def score(self, row: Dict[str, float]) -> float:
        x = np.asarray([[row[k] for k in self.feature_names]], dtype=float)
        xs = self.scaler.transform(x)
        s = float(self.model.decision_function(xs)[0])
        return s

    @property
    def threshold(self) -> Optional[float]:
        return getattr(self, "_score_threshold", None)

# ---------- CLI ----------

def cli():
    parser = argparse.ArgumentParser(description="POS ECOD anomaly agent (PyOD + ECOD)")
    parser.add_argument("--interval", type=int, default=int(os.getenv("PE_INTERVAL", "15")),
                        help="샘플링 주기(초). 기본 15")
    parser.add_argument("--window", type=int, default=int(os.getenv("PE_WINDOW", "5")),
                        help="윈도 크기(샘플 수). 기본 5")
    parser.add_argument("--baseline", type=int, default=int(os.getenv("PE_BASELINE", "60")),
                        help="베이스라인 윈도 개수. 기본 60")
    parser.add_argument("--threshold_pct", type=float, default=float(os.getenv("PE_THR_PCT", "98.0")),
                        help="임계값 백분위 (기본 98)")
    parser.add_argument("--sustain", type=str, default=os.getenv("PE_SUSTAIN", "6/10"),
                        help="지속성 조건 N/M (기본 6/10)")
    parser.add_argument("--demo", action="store_true", help="데모 모드: 가짜 이상치 주입")
    parser.add_argument("--mock", type=str, default=None, help="CSV 경로를 지정하면 mock 데이터 재생")
    parser.add_argument(
        "--mock-interval",
        type=float,
        default=float(os.getenv("PE_MOCK_INTERVAL", "0.01")),
        help="mock 재생 시 샘플링 대기(초). 기본 0.01 (매우 빠르게 재생)",
    )
    args = parser.parse_args()

    sustain_N, sustain_M = (6, 10)
    try:
        sustain_N, sustain_M = map(int, args.sustain.split("/"))
    except Exception:
        pass

    # 미리 mock 데이터 로드 및 baseline 자동 조정
    mock_df = None
    mock_idx = 0
    if args.mock:
        mock_df = pd.read_csv(args.mock)
        available_windows = max(0, len(mock_df) - args.window + 1)
        if available_windows > 0 and args.baseline > available_windows:
            # mock 데이터 길이에 맞게 baseline을 단축해 스코어링 단계가 실행되도록 조정
            new_baseline = max(1, available_windows // 2)
            if new_baseline != args.baseline:
                console.print(
                    f"[yellow]mock 데이터 길이에 맞춰 baseline을 {args.baseline} -> {new_baseline} 으로 조정합니다[/yellow]"
                )
                args.baseline = new_baseline

    # 실제 인터벌 vs. mock 인터벌 결정
    effective_interval = float(args.interval)
    if os.getenv("PE_MOCK_REALTIME", "0") in ("1", "true", "TRUE"):
        # 실시간처럼 동작시키고 싶을 때 강제로 원래 interval 사용
        pass
    elif args.mock:
        effective_interval = float(args.mock_interval)

    console.rule("[bold]POS ECOD anomaly agent (PyOD + ECOD)")
    console.print(
        f"[bold]interval[/]: {args.interval}s (effective: {effective_interval}s), "
        f"[bold]window[/]: {args.window}, [bold]baseline[/]: {args.baseline}, "
        f"[bold]thr_pct[/]: {args.threshold_pct}, [bold]sustain[/]: {sustain_N}/{sustain_M}, "
        f"[bold]mock[/]: {args.mock}"
    )

    prev_io: Optional[IOStatSnapshot] = None
    sample_buf: deque = deque(maxlen=args.window)
    baseline_buf: deque = deque(maxlen=args.baseline)

    feature_names = [
        "cpu_total_pct_p95", "cpu_iowait_pct_p95", "mem_used_pct_p95", "swap_used_pct_p95",
        "load1_max", "cpu_temp_c_max", "disk_await_ms_p95", "fs_used_pct_root_p95",
        "nic_err_rate_max", "io_saturation", "thermal_pressure",
    ]

    det = ECODDetector(feature_names, baseline_size=args.baseline, threshold_pct=args.threshold_pct)
    sustain_buf: deque = deque(maxlen=sustain_M)

    def maybe_inject_demo_noise(feat: Dict[str, float]):
        if not args.demo:
            return
        if np.random.rand() < 0.12:
            feat["disk_await_ms"] *= (2.0 + 2.5 * np.random.rand())
        if np.random.rand() < 0.10:
            feat["cpu_temp_c"] += (8.0 + 10.0 * np.random.rand())

    while True:
        t0 = time.time()
        if mock_df is not None:
            if mock_idx >= len(mock_df):
                console.print("[green]Mock CSV 재생 종료[/green]")
                break
            raw = next_mock_row(mock_df, mock_idx)
            mock_idx += 1
        else:
            raw, prev_io = collect_once(prev_io)

        maybe_inject_demo_noise(raw)
        sample_buf.append(raw)

        if len(sample_buf) == args.window:
            row = aggregate_window(list(sample_buf))

            if not det.ready():
                det.partial_fit_baseline(row)
                baseline_buf.append(row)
                pct = int(100 * len(baseline_buf) / args.baseline)
                tbl = Table(box=box.SIMPLE, title="Baseline Warm-up", show_edge=False)
                for k, v in row.items():
                    tbl.add_row(k, f"{v:.3f}")
                console.print(Panel(tbl, subtitle=f"[cyan]{pct}% 수집 중[/cyan]"))
                if det.ready():
                    det.fit()
                    console.print(f"\n[green]ECOD baseline fitted.[/green] "
                                  f"threshold (p{det.threshold_pct if hasattr(det,'threshold_pct') else 'X'}) "
                                  f"= {det.threshold:.4f}\n")
            else:
                s = det.score(row)
                thr = det.threshold or math.inf
                is_anom = s >= thr
                sustain_buf.append(1 if is_anom else 0)
                sustained = sum(sustain_buf) >= sustain_N and len(sustain_buf) >= sustain_M

                tbl = Table(box=box.SIMPLE, title="Window Features", show_edge=False)
                for k in feature_names:
                    tbl.add_row(k, f"{row[k]:.3f}")
                score_line = f"score={s:.4f}  thr={thr:.4f}  is_anom={is_anom}  sustained={sustained}"

                color = "green"
                if sustained:
                    color = "red"
                elif is_anom:
                    color = "yellow"
                console.print(Panel(tbl, subtitle=f"[{color}]{score_line}[/{color}]"))

                if sustained:
                    # 지속 조건을 만족한 강한 이상 상황 알림 (명확한 메시지 출력)
                    console.print(
                        f"[bold red]ANOMALY SUSTAINED[/bold red] score={s:.4f} thr={thr:.4f} "
                        f"window_ok={len(sustain_buf)}/{sustain_M}"
                    )
                    alert = {
                        "event": "POS_HW_ANOMALY",
                        "score": round(s, 4),
                        "threshold": round(thr, 4),
                        "sustained": f"{sum(sustain_buf)}/{len(sustain_buf)}",
                        "top_features": sorted(
                            feature_names,
                            key=lambda k: abs(row[k]),
                            reverse=True
                        )[:3],
                        "feature_snapshot": {k: float(row[k]) for k in feature_names},
                        "ts": int(time.time()),
                        "mock_path": args.mock,
                    }
                    console.print("[bold red]ALERT[/bold red] " + json.dumps(alert, ensure_ascii=False))
                elif is_anom:
                    # 일시적 이상 감지 알림 (지속 조건은 아직 충족하지 않음)
                    console.print(
                        f"[bold yellow]ANOMALY DETECTED[/bold yellow] score={s:.4f} thr={thr:.4f} "
                        f"sustain={sum(sustain_buf)}/{len(sustain_buf)} (< {sustain_N}/{sustain_M})"
                    )

        dt = time.time() - t0
        # mock 재생 시에는 매우 짧게 혹은 0초로 대기하여 빠르게 결과를 확인
        time.sleep(max(0, effective_interval - dt))

if __name__ == "__main__":
    cli()
