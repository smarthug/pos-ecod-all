#!/bin/bash
# =============================================
# System Diagnostic Script for PyOD / Deep Learning readiness
# Author: Hosuk's AI Assistant
# =============================================

set -e
mkdir -p ~/diag && cd ~/diag

REPORT="hardware_report.txt"
LOG="quick_bench.log"

echo "===== [1/4] 기본 시스템 정보 수집 ====="
{
  echo "===== OS ====="; date; uname -a; cat /etc/os-release 2>/dev/null
  echo; echo "===== CPU ====="; lscpu || cat /proc/cpuinfo
  echo; echo "===== MEMORY ====="; free -h || vm_stat 2>/dev/null
  echo; echo "===== DISK ====="; lsblk -o NAME,SIZE,TYPE,MOUNTPOINT,FSTYPE; echo; df -h
  echo; echo "===== NETWORK ====="; ip -brief addr || ifconfig 2>/dev/null
} > "$REPORT"

echo "===== [2/4] GPU 및 가속기 상태 확인 ====="
{
  echo "===== NVIDIA (nvidia-smi) ====="; nvidia-smi || echo "nvidia-smi not found"
  echo; echo "===== CUDA toolkit (nvcc) ====="; nvcc --version || echo "nvcc not found"
  echo; echo "===== AMD ROCm ====="; (rocm-smi || rocminfo) 2>/dev/null || echo "ROCm not found"
  echo; echo "===== OpenCL ====="; clinfo 2>/dev/null || echo "clinfo not found"
} >> "$REPORT"

echo "===== [3/4] Python 환경 검사 ====="
{
  echo; echo "===== Python ====="; which python3 || true; python3 --version || true; pip3 --version || true
  echo; echo "===== Key packages ====="
  python3 - << 'PY'
import importlib
pkgs = ["numpy","scipy","scikit-learn","pyod","torch","tensorflow","jax"]
for p in pkgs:
    try:
        m = importlib.import_module(p)
        v = getattr(m, "__version__", "unknown")
        print(f"{p}: {v}")
    except Exception:
        print(f"{p}: NOT INSTALLED")
PY
} >> "$REPORT"

echo "✅ 하드웨어 리포트 완료: $(pwd)/$REPORT"

# Quick benchmark section
echo "===== [4/4] PyOD / 딥러닝 간단 벤치마크 ====="
cat > quick_bench.py << 'PY'
import os, time, numpy as np
np.random.seed(0)

print("=== PyOD / ECOD quick check ===")
from pyod.models.ecod import ECOD
X = np.random.randn(5000, 8)
t0 = time.time()
clf = ECOD().fit(X)
scores = clf.decision_scores_
print(f"ECOD fit+score OK. n={len(scores)}, time={time.time()-t0:.3f}s")

print("\n=== Torch quick matmul (GPU 있으면 사용) ===")
try:
    import torch
    print("torch:", torch.__version__)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    N = 2048
    a = torch.randn(N, N, device=device)
    b = torch.randn(N, N, device=device)
    t0 = time.time()
    c = a @ b
    if device == "cuda": torch.cuda.synchronize()
    dt = time.time()-t0
    flops = 2*(N**3)/dt/1e12
    print(f"MatMul {N}x{N}: {dt:.3f}s  ~{flops:.3f} TFLOPS (rough)")
except Exception as e:
    print("Torch test skipped:", e)
PY

python3 quick_bench.py | tee "$LOG"

echo
echo "✅ 모든 진단 완료!"
echo "📄 리포트: $(pwd)/$REPORT"
echo "📄 벤치 로그: $(pwd)/$LOG"