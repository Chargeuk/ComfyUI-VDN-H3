"""Placement, lifetime and API regressions; no model files required."""
import contextlib
import threading
import weakref
from types import SimpleNamespace

import pytest
import torch

from vdn_h3 import branch, hybrid, nodes, window
from vdn_h3.spec import runtime_memory_policy

GIB = 1 << 30


@pytest.mark.parametrize("dtype", list(branch._NORM_EPSILON))
def test_epsilon_preserves_original_rounding(dtype):
    assert branch._NORM_EPSILON[dtype] == torch.tensor(1e-6, dtype=dtype).item()


@pytest.mark.parametrize("free,unloaded,workspace,expected", [
    (8, 10, 4, (False, True)),    # offloaded base on a small GPU
    (3, 10, 4, (False, False)),   # no speculative copies under pressure
    (12, 0, 4, (True, False)),   # larger GPU, fully loaded base
    (80, 0, 4, (True, False)),   # shared-memory machine, ample available RAM
    (12, 0, 10, (False, True)),  # a larger workload releases the cache
])
def test_memory_capacity_not_gpu_name(free, unloaded, workspace, expected):
    result = runtime_memory_policy(free * GIB, 3 * GIB, .1 * GIB,
                                   workspace * GIB, unloaded * GIB,
                                   base_stream_bytes=2 * GIB)
    assert result[:2] == expected


def test_explicit_placement_controls():
    args = (12 * GIB, 3 * GIB, .1 * GIB, 4 * GIB, 0)
    assert runtime_memory_policy(*args, cache_mode="stream")[:2] == (False, True)
    assert runtime_memory_policy(*args, cache_mode="stream",
                                 prefetch_mode="off")[:2] == (False, False)
    assert runtime_memory_policy(0, 3 * GIB, GIB, 4 * GIB, 0,
                                 cache_mode="cache_gpu")[:2] == (True, False)
    assert runtime_memory_policy(0, 3 * GIB, GIB, 4 * GIB, 0,
                                 prefetch_mode="on")[:2] == (False, False)


def test_runtime_policy_rechecks_and_releases_cache(monkeypatch):
    state = hybrid.VDNState("test", {}, [None], 2, 8)
    state.layout = SimpleNamespace(seq_len=32, num_frames=4)
    state.stage_bytes = 3 * GIB
    state.block_bytes = .1 * GIB
    state.cache_mode = "auto"
    monkeypatch.setattr(hybrid.comfy.model_management, "get_free_memory",
                        lambda device: 12 * GIB)
    state.configure_memory(torch.device("cuda:0"), torch.bfloat16)
    assert state.cache_gpu
    state._gpu_cache["dummy"] = object()
    monkeypatch.setattr(hybrid.comfy.model_management, "get_free_memory",
                        lambda device: 0)
    state.configure_memory(torch.device("cuda:0"), torch.bfloat16)
    assert not state.cache_gpu and not state._gpu_cache
    assert not state.prefetch_enabled


@pytest.mark.parametrize("retain", [False, True])
def test_prefetch_independent_of_retained_scratch(monkeypatch, retain):
    state = hybrid.VDNState("test", {}, [None], 2, 8)
    state.layout = SimpleNamespace(seq_len=32, num_frames=4)
    state.block_bytes = .1 * GIB
    state.retain_buffers = retain
    monkeypatch.setattr(hybrid.comfy.model_management, "get_free_memory",
                        lambda device: 8 * GIB)
    state.configure_memory(torch.device("cuda:0"), torch.bfloat16)
    assert state.prefetch_enabled


def test_prefetch_waits_for_late_copy_and_releases_storage(monkeypatch):
    stream = SimpleNamespace(wait_event=lambda event: None)
    monkeypatch.setattr(torch.cuda, "Stream", lambda **kw: stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "Event", lambda: SimpleNamespace(record=lambda s: None))
    monkeypatch.setattr(torch.cuda, "current_stream", lambda **kw: stream)
    monkeypatch.setattr(hybrid, "_record_stream_needed", lambda: False)
    pf = hybrid._StreamPrefetcher(torch.device("cuda:0"))
    entered, release, consumed = threading.Event(), threading.Event(), threading.Event()
    calls, references, result = [], [], []

    def fetch():
        calls.append(1)
        entered.set()
        assert release.wait(5)
        tensor = torch.ones(4)
        references.append(weakref.ref(tensor))
        return {"w": tensor}

    pf.request(1, fetch)
    assert entered.wait(5)

    def consume():
        result.append(pf.take(1))
        consumed.set()

    consumer = threading.Thread(target=consume)
    consumer.start()
    assert not consumed.wait(.05), "late copy incorrectly treated as a cache miss"
    release.set()
    assert consumed.wait(5)
    consumer.join()
    assert len(calls) == 1 and result[0] is not None
    result.clear()
    # A subsequent request guarantees the worker has left its previous finally.
    pf.request(2, lambda: {})
    assert pf.take(2) == {}
    assert references[0]() is None, "idle worker holds the last GPU weight block"
    pf.reset()
    assert not pf._done and not pf._inflight


def test_qkv_storage_dead_before_branch_readout(monkeypatch):
    heads, dim, seq = 2, 4, 8
    allocation = []

    def project(x):
        qkv = torch.randn(seq, 3 * heads * dim)
        allocation.append(weakref.ref(qkv))
        return qkv

    def readout(w, xv, *args, **kwargs):
        assert allocation[0]() is None, "RoPE alias retained the QKV allocation"
        return torch.zeros_like(xv)

    attn = SimpleNamespace(heads=heads, head_dim=dim, qkv_proj=project,
                           out_proj=lambda x: x,
                           q_norm=torch.nn.RMSNorm(dim), k_norm=torch.nn.RMSNorm(dim))
    state = hybrid.VDNState("test", {"enable_softmax_gate": False,
                                    "anchor_frames": "none"},
                            [SimpleNamespace(enable_text_state=False, readout=readout)],
                            heads, dim)
    state.layout = hybrid.VDNLayout(0, seq, 4, 2, (1, 2), seq, 0, seq, 0, 1, "none")
    state.weights_on = lambda *args: {"to_out_linear.weight": torch.eye(heads * dim)}
    monkeypatch.setattr(hybrid.comfy.quant_ops.ck, "rms_rope_split_half_", lambda *a, **k: None)
    monkeypatch.setattr(window, "window_softmax_grouped", lambda q, *a, **k: q.clone())
    out = hybrid.make_vdn_forward(attn, state, 0)(torch.zeros(seq, heads * dim),
                                                rope_freqs=torch.zeros(1, 1, 1))
    assert out.shape == (seq, heads * dim)


@pytest.mark.parametrize("node", [nodes.ApplyVDNH3, nodes.ApplyVDNH3Advanced])
def test_new_controls_are_optional_and_conservative(node, monkeypatch):
    monkeypatch.setattr(nodes.spec, "list_vdn_checkpoints", lambda: ["test"])
    inputs = node.INPUT_TYPES()
    assert inputs["optional"]["compile_scan"][1]["default"] is False
    assert inputs["optional"]["fuse_statistics"][1]["default"] is False
    assert inputs["optional"]["prefetch"][1]["default"] == "auto"
    assert list(inputs["optional"])[-3:] == ["prefetch", "compile_scan", "fuse_statistics"]


def test_compile_failure_latches_eager_and_logs(monkeypatch, caplog):
    key = ("test_fallback",)
    calls = []

    def broken(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("test compiler unavailable")

    monkeypatch.setattr(torch, "compile", broken)
    try:
        assert branch._run_compiled(key, lambda x: x + 1, 2) == 3
        assert branch._run_compiled(key, lambda x: x + 1, 3) == 4
        assert len(calls) == 1
        assert "using eager" in caplog.text
        assert key not in branch._COMPILED_CACHE
    finally:
        branch._COMPILED_BROKEN.discard(key)
