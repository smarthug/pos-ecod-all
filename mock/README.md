# Mock Data for POS ECOD
- `pos_hw_metrics_normal.csv`: 정상 상태 1시간 (1분 간격)
- `pos_hw_metrics_anomaly.csv`: 45분 이후부터 이상(열/디스크/메모리/네트워크 오류율 상승)

Columns:
ts_minute, cpu_total_pct, cpu_iowait_pct, mem_used_pct, swap_used_pct, cpu_temp_c,
disk_await_ms, fs_used_pct_root, nic_err_rate