#!/usr/bin/env bash
# Which GPU profiling tools work on this cluster? Run on a COMPUTE node (never the login node), never loads a
# model, changes no system setting, uses sudo only as `sudo -n true` (a permission check).
#
#   mkdir -p ~/probes/logs && cp scripts/profiling_probe.sh scripts/ncu_smoke.py ~/probes/
#   cp src/s1/gpu_stats.py src/s1/record.py ~/probes/pkg/s1/   (+ empty __init__.py; see docs/profiling.md)
#   sbatch -p gpu --gres=gpu:1 -c 4 --mem=16G -t 00:10:00 -J profiling-probe \
#          -o ~/probes/logs/%j.out ~/probes/profiling_probe.sh
#
# Output: stdout (the slurm log) and $OUT (default ~/probes/out/<jobid>) with the raw ncu/nsys files.
# On Modal (MODAL_TASK_ID set) scripts/modal_probe.py runs this script; the Slurm guard is skipped there.
# Overrides: PROBE_DIR, OUT, PYBIN, SIZES, TOOL_PATH (dirs prepended to PATH, default /usr/local/cuda/bin),
# NCU_CLOCK_CONTROL (passed as `ncu --clock-control`; default: ncu's own (base) on Slurm, `none` on Modal,
# where clock locking fails).
# Sections: 0 node, a nvidia-smi fields and dmon, b DCGM, c ncu, d nsys, e GPU-metrics sampler idle vs busy.
#SBATCH -J profiling-probe
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:10:00
set -u

JOB="${SLURM_JOB_ID:-${MODAL_TASK_ID:-}}"
if [ -z "$JOB" ]; then
    echo "not inside a Slurm allocation: refusing to run on the login node (sbatch me)" >&2
    exit 2
fi
if [ -n "${MODAL_TASK_ID:-}" ] && [ -z "${SLURM_JOB_ID:-}" ]; then
    NCU_CLOCK_CONTROL="${NCU_CLOCK_CONTROL-none}"
fi
NCU_CC=()
[ -n "${NCU_CLOCK_CONTROL:-}" ] && NCU_CC=(--clock-control "$NCU_CLOCK_CONTROL")

# Slurm runs a COPY of this script from its spool dir, so look for ncu_smoke.py where sbatch was invoked
HERE="${PROBE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}"
[ -f "$HERE/ncu_smoke.py" ] || { echo "ncu_smoke.py not found in $HERE (sbatch from the directory holding it)" >&2; exit 2; }
OUT="${OUT:-$HOME/probes/out/$JOB}"
PY="${PYBIN:-$HOME/cs2470-final-project/.venv/bin/python}"   # read-only use of the shared venv's torch
[ -x "$PY" ] || PY="$(command -v python3)"
SIZES="${SIZES:-512,1024,2048,4096}"
mkdir -p "$OUT"
export PYTHONPATH="$HERE/pkg:${PYTHONPATH:-}"
# ncu / nsys / nvcc live in /usr/local/cuda/bin on the compute nodes but are not on PATH
export PATH="${TOOL_PATH:-/usr/local/cuda/bin}:$PATH"

hr() { printf '\n==== %s ====\n' "$*"; }
run() { echo "\$ $*"; timeout "${T:-60}" "$@" 2>&1; echo "[exit $?]"; }

# ---------------------------------------------------------------- 0. node
hr "0. node"
date -Is; hostname; echo "job $JOB partition ${SLURM_JOB_PARTITION:-?}"
run nvidia-smi -L
run nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv
nvidia-smi | head -4 | tail -2    # banner shows "CUDA Version: X" = what the driver supports
echo "--- CUDA toolkit"
for t in nvcc ncu nsys dcgmi nv-hostengine; do printf '%-14s %s\n' "$t" "$(command -v $t || echo MISSING)"; done
ls -d /usr/local/cuda* /opt/nvidia/nsight* /opt/nvidia/hpc_sdk 2>/dev/null
for d in /usr/local/cuda*/bin /opt/nvidia/nsight-compute/* /opt/nvidia/nsight-systems/*/bin /opt/nvidia/nsight-systems/*/target-linux-x64; do
    ls "$d"/ncu "$d"/nsys "$d"/nvcc 2>/dev/null
done
command -v nvcc >/dev/null && nvcc --version | tail -2
command -v module >/dev/null && module avail cuda nsight 2>&1 | head -20
"$PY" - <<'EOF' 2>&1
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("device", p.name, f"cc {p.major}.{p.minor}", f"{p.total_memory / 2**30:.1f} GiB", f"{p.multi_processor_count} SMs")
EOF

# ---------------------------------------------------------------- a. nvidia-smi
hr "a. nvidia-smi query fields"
for f in utilization.gpu utilization.memory clocks.sm clocks.mem clocks.current.sm \
         clocks_throttle_reasons.active clocks_event_reasons.active power.draw temperature.gpu \
         pcie.link.gen.current pcie.link.width.current; do
    printf '%-34s ' "$f"
    nvidia-smi --query-gpu="$f" --format=csv,noheader,nounits 2>&1 | head -1
done
echo "--- listed throttle/event/clock fields"
nvidia-smi --help-query-gpu 2>&1 | grep -iE 'throttle|event_reasons|"clocks\.(sm|mem)"|utilization' | head -20
echo "--- the full detailed query as the sampler issues it (3 rows)"
Q="index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu,utilization.memory,clocks.sm,clocks.mem"
if nvidia-smi --help-query-gpu 2>/dev/null | grep -q '"clocks_event_reasons.active"'; then Q="$Q,clocks_event_reasons.active"
elif nvidia-smi --help-query-gpu 2>/dev/null | grep -q '"clocks_throttle_reasons.active"'; then Q="$Q,clocks_throttle_reasons.active"; fi
echo "query: $Q"
timeout 10 nvidia-smi --query-gpu="$Q" --format=csv,noheader,nounits -lms=500 2>&1 | head -3
echo "--- throttle detail"
nvidia-smi -q -d PERFORMANCE 2>&1 | head -30
echo "--- nvidia-smi dmon -s t, piped, with arrival times (does it stream line by line or in one block?)"
timeout 20 nvidia-smi dmon -s t -d 1 -c 5 2>&1 | while IFS= read -r l; do printf '%s %s\n' "$(date +%S.%N | cut -c1-6)" "$l"; done

# ---------------------------------------------------------------- b. DCGM (under load so values are not all 0)
hr "b. DCGM"
"$PY" "$HERE/ncu_smoke.py" time --sizes 4096 >/dev/null 2>&1   # CUDA/cuBLAS warm so the load below starts fast
"$PY" - >/dev/null 2>&1 <<'EOF' &
import time, torch
a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
end = time.time() + 40
while time.time() < end:
    torch.matmul(a, a)
torch.cuda.synchronize()
EOF
LOAD_PID=$!
sleep 4
if command -v dcgmi >/dev/null; then
    run dcgmi --version
    echo "--- as installed: is a host engine already running? (expected: no)"
    run dcgmi discovery -l
    DM="-e 1002,1003,1004,1005,1009,1010 -c 5 -d 500"
    if command -v nv-hostengine >/dev/null; then
        echo "--- our own user-space host engine on a private port (no daemon, no root, killed at the end)"
        run nv-hostengine --help | head -40
        PORT=$((20000 + ${SLURM_JOB_ID:-$$} % 20000))  # Modal has no Slurm job id
        HE_PID=""
        for flags in "-n -b 127.0.0.1 -p $PORT" "-n -p $PORT"; do
            env -u CUDA_VISIBLE_DEVICES nv-hostengine $flags >"$OUT/hostengine.log" 2>&1 &
            HE_PID=$!
            sleep 4
            if kill -0 "$HE_PID" 2>/dev/null; then echo "nv-hostengine started with: $flags (pid $HE_PID)"; break; fi
            echo "nv-hostengine failed with: $flags"; cat "$OUT/hostengine.log"; HE_PID=""
        done
        if [ -n "$HE_PID" ]; then
            T=30 run env -u CUDA_VISIBLE_DEVICES dcgmi discovery -l --host 127.0.0.1:$PORT
            # shellcheck disable=SC2086
            T=40 run env -u CUDA_VISIBLE_DEVICES dcgmi dmon --host 127.0.0.1:$PORT $DM
            kill "$HE_PID" 2>/dev/null; wait "$HE_PID" 2>/dev/null
            echo "--- host engine log"; tail -20 "$OUT/hostengine.log"
        fi
    else
        echo "nv-hostengine: MISSING"
    fi
else
    echo "dcgmi: MISSING (not on PATH on this node)"
fi
wait "$LOAD_PID" 2>/dev/null

# ---------------------------------------------------------------- c. ncu
hr "c. ncu"
if ! command -v ncu >/dev/null; then
    echo "ncu: MISSING"
else
    run ncu --version
    echo "--- permission"
    grep -i RmProfilingAdminOnly /proc/driver/nvidia/params 2>&1 || echo "(RmProfilingAdminOnly not in /proc/driver/nvidia/params)"
    if sudo -n true 2>/dev/null; then echo "sudo -n true: WORKS (passwordless sudo)"; else echo "sudo -n true: no passwordless sudo"; fi
    ncu --query-metrics > "$OUT/ncu_query_metrics.txt" 2>&1
    echo "--- tensor/dram metrics this ncu knows on this chip (full list: $OUT/ncu_query_metrics.txt)"
    grep -iE 'tensor' "$OUT/ncu_query_metrics.txt" | head -15
    grep -iE '^dram__bytes_(read|write)\.sum ' "$OUT/ncu_query_metrics.txt" | head
    # pick the tensor-pipe metric that exists here; every requested metric is checked against the query list
    TENSOR=""
    for m in sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active \
             sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active \
             sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active; do
        base="${m%%.*}"
        if grep -q "^$base " "$OUT/ncu_query_metrics.txt"; then TENSOR="$m"; break; fi
    done
    METRICS="gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum,dram__bytes_write.sum"
    [ -n "$TENSOR" ] && METRICS="$METRICS,$TENSOR"
    echo "tensor metric used: ${TENSOR:-NONE FOUND}"
    echo "metrics: $METRICS"
    echo "clock control: ${NCU_CLOCK_CONTROL:-ncu default (base)}"
    echo "--- profiling the matmul workload (sizes $SIZES)"
    T=240 run ncu ${NCU_CC[@]+"${NCU_CC[@]}"} --target-processes all --csv --metrics "$METRICS" \
        --log-file "$OUT/ncu.csv" "$PY" "$HERE/ncu_smoke.py" run --sizes "$SIZES"
    if grep -q ERR_NVGPUCTRPERM "$OUT/ncu.csv" 2>/dev/null; then
        echo "!!! ncu: ERR_NVGPUCTRPERM -- exact message:"
        grep -m3 -i -B1 -A2 'ERR_NVGPUCTRPERM' "$OUT/ncu.csv"
        echo "STOPPING ncu attempts (no workaround tried, nothing changed on the node)."
    else
        "$PY" "$HERE/ncu_smoke.py" parse "$OUT/ncu.csv" --sizes "$SIZES" \
            --gpu "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
        echo "--- unprofiled reference (CUDA events)"
        "$PY" "$HERE/ncu_smoke.py" time --sizes "$SIZES"
    fi
fi

# ---------------------------------------------------------------- d. nsys
hr "d. nsys"
if ! command -v nsys >/dev/null; then
    echo "nsys: MISSING"
else
    run nsys --version
    echo "--- timeline: cuda + nvtx + kernel summary"
    T=150 run nsys profile --trace=cuda,nvtx --stats=true -f true -o "$OUT/nsys_basic" \
        "$PY" "$HERE/ncu_smoke.py" run --sizes "$SIZES" | tail -60
    ls -la "$OUT"/nsys_basic* 2>&1
    echo "--- GPU metrics sampling (SM active, tensor active, DRAM bandwidth, PCIe) -- may need counter permission"
    T=150 run nsys profile --trace=cuda --gpu-metrics-devices=cuda-visible -f true -o "$OUT/nsys_gpumetrics" \
        "$PY" "$HERE/ncu_smoke.py" run --sizes "$SIZES" | tail -25
    ls -la "$OUT"/nsys_gpumetrics* 2>&1
    if [ -f "$OUT/nsys_gpumetrics.nsys-rep" ]; then
        nsys stats --report gpu_metrics --force-export=true "$OUT/nsys_gpumetrics.nsys-rep" 2>&1 | head -30 \
            || echo "(no gpu_metrics report)"
    fi
fi

# ---------------------------------------------------------------- e. our sampler, idle vs busy
hr "e. GpuSampler(detailed): idle vs busy, same GPU"
T=120 run "$PY" "$HERE/ncu_smoke.py" sample

hr "done"
echo "raw files in $OUT"
