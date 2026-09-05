"""Opt-in CUDA microbenchmark; not an end-to-end video speed prediction.

Run with ComfyUI on PYTHONPATH, e.g. from custom_nodes/ComfyUI-VDN-H3:
    PYTHONPATH=../.. python tools/benchmark_performance.py
No model files are loaded. Compilation can take time and reserve GPU memory.
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vdn_h3 import branch


def measure(label, operation, repeats):
    torch.cuda.synchronize()
    initial = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    result = operation()
    torch.cuda.synchronize()
    first_ms = (time.perf_counter() - start) * 1000
    del result
    for _ in range(3):
        operation()
    samples = []
    for _ in range(repeats):
        start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start_event.record()
        operation()
        end_event.record()
        end_event.synchronize()
        samples.append(start_event.elapsed_time(end_event))
    extra_mib = (torch.cuda.max_memory_allocated() - initial) / (1 << 20)
    print(f"{label}: first={first_ms:.2f} ms, warm median={statistics.median(samples):.3f} ms, "
          f"extra allocated peak={extra_mib:.1f} MiB", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--heads", type=int, default=56)
    parser.add_argument("--tokens-per-frame", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    torch.manual_seed(15)
    f, h, s, d = args.frames, args.heads, args.tokens_per_frame, 128
    print(f"{torch.cuda.get_device_name()}; torch {torch.__version__}; F={f} H={h} S={s} D={d}", flush=True)
    k = torch.randn((f, s, h, d), device="cuda", dtype=torch.bfloat16).permute(0, 2, 1, 3)
    v = torch.randn_like(k)
    beta = torch.rand((f, s, h), device="cuda", dtype=torch.bfloat16).permute(0, 2, 1)
    for fused in (False, True):
        measure(f"statistics fuse={fused}", lambda: branch.frame_statistics(k, v, beta, fuse=fused), args.repeats)
    del k, v, beta
    transitions = torch.randn((f, h, d, d), device="cuda") * .01
    injections = torch.randn_like(transitions)
    start = torch.zeros((h, d, d), device="cuda")
    key = ("scan", f, *start.shape, str(start.device), str(start.dtype))
    measure("scan eager", lambda: branch._scan_body(transitions, injections, start), args.repeats)
    measure("scan compiled", lambda: branch._run_compiled(
        key, branch._scan_body, transitions, injections, start, _mode="reduce-overhead"), args.repeats)
    assert not any(key[0] in ("scan", "stats_prep") for key in branch._COMPILED_BROKEN), "Compilation fell back; do not interpret as compiled timings"


if __name__ == "__main__":
    main()
