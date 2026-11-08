"""Which GPU profiling tools work on a Modal GPU? The Modal twin of `profiling_probe.sh` (docs/profiling.md).

    uvx --python 3.12 --from modal==1.6.3 modal run scripts/modal_probe.py --gpu L4
    uvx --python 3.12 --from modal==1.6.3 modal run scripts/modal_probe.py --gpu 'H100!'

One short GPU job per invocation (a few minutes; the first run also builds the image, which is CPU-only).
The container runs `profiling_probe.sh` as on the cluster: node info, nvidia-smi query fields and
`dmon -s t`, DCGM availability, ncu (`--clock-control none`: clock locking fails on Modal), nsys, and the
`ncu_smoke.py` matmul roofline plus the GpuSampler idle/busy check. Before it, the full `nvidia-smi -q` is
saved and summarized. Output goes to the Modal volume `s1-probe-logs` under `<gpu>-<stamp>-<id>/`, and is
downloaded to `logs/modal-probe-<gpu>-<stamp>-<id>/`.

Nsight Compute and Nsight Systems come from NVIDIA's CUDA redist archive (pinned below, sha256-checked) at
image build time, as in the course starter (github.com/kevinhe5/cs2470-modal-starter, `profile_gpu.py`).

`modal` is not a dependency of the s1 package: it is imported only here, so the cluster and the tests never
need it. Never loads a model; changes no system setting.
"""

from __future__ import annotations

import datetime
import hashlib
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import uuid
from pathlib import Path

modal = importlib.import_module("modal")  # only this script needs modal (run it with `uvx --from modal`)

REPO = Path(__file__).resolve().parents[1]
VOLUME = "s1-probe-logs"
# `!` stops Modal upgrading an H100 to an H200; a plain "H100" is refused so a probe never lands on an H200.
GPUS = ("L4", "A10", "L40S", "A100-80GB", "H100!")
TORCH = "torch==2.14.1"  # what the course starter runs on Modal
# CUDA redist 13.4.3 (redistrib_13.4.3.json): relative path and sha256 of each linux-x86_64 archive.
REDIST = "https://developer.download.nvidia.com/compute/cuda/redist/"
TOOLS = {
    "ncu": (
        "nsight_compute/linux-x86_64/nsight_compute-linux-x86_64-2026.3.1.2-archive.tar.xz",
        "a0039be47cfe551f89d82087256080a9f33c6257d99a72c1164a87466ab072aa",
    ),
    "nsys": (
        "nsight_systems/linux-x86_64/nsight_systems-linux-x86_64-2026.3.2.476-archive.tar.xz",
        "af8daa081e7333e3fb4a7fc3505e379593f05a8014a703f268560cec22338aca",
    ),
}
TOOL_BIN = "/opt/nvidia/bin"  # wrapper scripts for ncu/nsys; the probe prepends it to PATH (TOOL_PATH)
PROBE = "/probe"  # profiling_probe.sh, ncu_smoke.py and pkg/s1/{gpu_stats,record}.py
OUT = "/out"  # the volume
SMI_KEYS = (  # lines of `nvidia-smi -q` worth seeing in the log (clocks and PCIe come from the probe itself)
    "Product Name",
    "Product Architecture",
    "Driver Version",
    "CUDA Version",
    "Persistence Mode",
    "Addressing Mode",
    "Virtualization Mode",
    "Host VGPU Mode",
    "Bus Id",
    "Compute Mode",
    "Current Power Limit",
    "Enforced Power Limit",
    "Default Power Limit",
    "Max Power Limit",
)


def install_tools() -> None:
    """Image build step (no GPU rented): fetch, verify and unpack ncu and nsys; wrappers in TOOL_BIN."""
    Path(TOOL_BIN).mkdir(parents=True, exist_ok=True)
    for tool, (rel, sha) in TOOLS.items():
        archive = Path(f"/tmp/{tool}.tar.xz")
        urllib.request.urlretrieve(REDIST + rel, archive)
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        if digest != sha:
            raise RuntimeError(f"{rel}: sha256 {digest} != pinned {sha}")
        dest = Path("/opt/nvidia") / tool
        with tarfile.open(archive) as tar:
            tar.extractall(dest, filter="data")
        archive.unlink()
        # the launchers: ncu at the archive's top level, nsys in bin/ (both find their target/ dirs)
        found = [p for p in dest.rglob(tool) if p.is_file() and os.access(p, os.X_OK)]
        found.sort(key=lambda p: (p.parent.name != "bin", len(p.parts), str(p)))
        if not found:
            raise RuntimeError(f"no {tool} executable in {rel}")
        wrapper = Path(TOOL_BIN) / tool
        wrapper.write_text(f'#!/bin/sh\nexec "{found[0]}" "$@"\n')
        wrapper.chmod(0o755)
        print(f"{tool}: {found[0]}")
    shutil.rmtree("/root/.cache", ignore_errors=True)


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("procps")  # `timeout` is in coreutils; procps for ps/pgrep in debugging
    .pip_install(TORCH)
    .run_function(install_tools)
    .run_commands(f"mkdir -p {PROBE}/pkg/s1 && touch {PROBE}/pkg/s1/__init__.py")
    .add_local_file(REPO / "scripts/profiling_probe.sh", f"{PROBE}/profiling_probe.sh")
    .add_local_file(REPO / "scripts/ncu_smoke.py", f"{PROBE}/ncu_smoke.py")
    .add_local_file(REPO / "src/s1/gpu_stats.py", f"{PROBE}/pkg/s1/gpu_stats.py")
    .add_local_file(REPO / "src/s1/record.py", f"{PROBE}/pkg/s1/record.py")
)
app = modal.App("s1-modal-probe")
volume = modal.Volume.from_name(VOLUME, create_if_missing=True)


def smi_summary(text: str) -> list[str]:
    """The `nvidia-smi -q` lines whose key is in SMI_KEYS, stripped, in order."""
    return [ln.strip() for ln in text.splitlines() if ln.split(":", 1)[0].strip() in SMI_KEYS and ":" in ln]


def query(field: str) -> str:
    r = subprocess.run(
        ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    return r.stdout.strip().splitlines()[0] if r.stdout.strip() else ""


# ty sees modal as Any (imported dynamically: it is not an s1 dependency), hence the two ignores.
@app.function(image=image, gpu="L4", cpu=4, memory=16384, volumes={OUT: volume}, timeout=1200)  # ty: ignore[dynamic-function-decorator-return]
def probe(gpu: str, sizes: str) -> str:
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{gpu.lower().replace('!', '')}-{stamp}-{uuid.uuid4().hex[:8]}"
    out = Path(OUT) / run_id
    out.mkdir(parents=True)
    smi_q = subprocess.run(["nvidia-smi", "-q"], capture_output=True, text=True, check=False).stdout
    (out / "nvidia-smi-q.txt").write_text(smi_q)
    print("==== nvidia-smi -q (summary; full text in nvidia-smi-q.txt) ====")
    print("\n".join(smi_summary(smi_q)), flush=True)

    env = os.environ | {
        "PROBE_DIR": PROBE,
        "OUT": str(out),
        "PYBIN": sys.executable,
        "TOOL_PATH": TOOL_BIN,
        "SIZES": sizes,
        "PYTHONUNBUFFERED": "1",
    }
    log = out / "probe.log"
    with (
        log.open("w") as fh,
        subprocess.Popen(
            ["bash", f"{PROBE}/profiling_probe.sh"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        ) as proc,
    ):
        assert proc.stdout is not None
        for line in proc.stdout:  # stream to the local terminal and keep a copy
            print(line, end="", flush=True)
            fh.write(line)
    text = log.read_text()
    summary = {
        "run_id": run_id,
        "requested_gpu": gpu,
        "modal_task_id": os.environ.get("MODAL_TASK_ID", ""),
        "gpu_name": query("name"),
        "driver": query("driver_version"),
        "compute_cap": query("compute_cap"),
        "torch": TORCH,
        "tools": {t: rel.rsplit("/", 1)[1] for t, (rel, _) in TOOLS.items()},
        "probe_exit": proc.returncode,
        "ncu_ran": "ncu.csv" in {p.name for p in out.iterdir()},
        "ncu_perm_error": "ERR_NVGPUCTRPERM" in text,
        "rm_profiling_admin_only": next(iter(re.findall(r"RmProfilingAdminOnly:\s*(\d)", text)), None),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    volume.commit()
    files = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())
    return json.dumps({"summary": summary, "files": files})


@app.local_entrypoint()  # ty: ignore[dynamic-function-decorator-return]
def main(gpu: str = "L4", sizes: str = "512,1024,2048,4096") -> None:
    if gpu not in GPUS:
        raise ValueError(
            f"--gpu must be one of {GPUS} (use 'H100!', never a plain H100: it may become an H200)"
        )
    result = json.loads(probe.with_options(gpu=gpu).remote(gpu, sizes))
    summary = result["summary"]
    dest = REPO / "logs" / f"modal-probe-{summary['run_id']}"
    for name in result["files"]:
        path = dest / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            for chunk in volume.read_file(f"/{summary['run_id']}/{name}"):
                fh.write(chunk)
    print(json.dumps(summary, indent=2))
    print(f"downloaded {len(result['files'])} files to {dest}")
