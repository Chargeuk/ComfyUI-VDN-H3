"""Actual CUDA compilation/parity checks, not silently successful fallbacks."""
import pytest
import torch

from vdn_h3 import branch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(autouse=True)
def isolate_scratch_caches():
    # Legacy tests assert cache-size deltas; don't fill their bounded LRUs.
    banks = branch._SCAN_BANKS.copy()
    scratch = branch._DELTA_SCRATCH.copy()
    branch._SCAN_BANKS.clear()
    branch._DELTA_SCRATCH.clear()
    try:
        yield
    finally:
        branch._SCAN_BANKS.clear()
        branch._SCAN_BANKS.update(banks)
        branch._DELTA_SCRATCH.clear()
        branch._DELTA_SCRATCH.update(scratch)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_compiled_statistics_exact(dtype):
    torch.manual_seed(12)
    shape = (5, 97, 3, 128)
    k = torch.randn(shape, device="cuda", dtype=dtype).permute(0, 2, 1, 3)
    v = torch.randn_like(k)
    beta = torch.rand(shape[:-1], device="cuda", dtype=dtype).permute(0, 2, 1)
    eager = branch.frame_statistics(k, v, beta, fuse=False)
    actual = branch.frame_statistics(k, v, beta, fuse=True)
    assert all(torch.equal(a, b) for a, b in zip(eager, actual))
    key = ("stats_prep", str(k.device), k.dtype, v.dtype, beta.dtype)
    assert key in branch._COMPILED_CACHE and key not in branch._COMPILED_BROKEN


@pytest.mark.parametrize("frames", [5, 17])
def test_compiled_scan_exact(frames):
    torch.manual_seed(13)
    shape = (frames, 4, 128, 128)
    transitions = torch.randn(shape, device="cuda") * .01
    injections = torch.randn(shape, device="cuda")
    start = torch.randn(shape[1:], device="cuda")
    eager = branch._scan_body(transitions, injections, start)
    key = ("scan", frames, *start.shape, str(start.device), str(start.dtype))
    actual = branch._run_compiled(key, branch._scan_body, transitions, injections,
                                  start, _mode="reduce-overhead")
    assert all(torch.equal(a, b) for a, b in zip(eager, actual))
    assert key in branch._COMPILED_CACHE and key not in branch._COMPILED_BROKEN


def test_branch_readout_with_both_options_and_text_state():
    torch.manual_seed(14)
    frames, per_frame, heads, dim, hidden = 5, 4, 2, 128, 256
    channels = heads * dim
    shapes = {
        "beta_proj.weight": (heads, hidden),
        "alpha.down.weight": (dim, hidden),
        "alpha.up.weight": (channels, dim),
        "alpha.dt_bias": (channels,), "alpha.A_log": (heads,),
        "output_gate.down.weight": (dim, hidden),
        "output_gate.up.weight": (channels, dim),
        "output_gate.up.bias": (channels,), "norm.weight": (dim,),
        "short_conv.k_sp.weight": (channels, 1, 5, 5),
        "short_conv.k_tm.weight": (channels, 1, 5),
        "short_conv.v_sp.weight": (channels, 1, 5, 5),
        "short_conv.v_tm.weight": (channels, 1, 5),
    }
    def rand(*shape):
        return torch.randn(shape, device="cuda", dtype=torch.bfloat16) * .1
    w = {key: rand(*shape) for key, shape in shapes.items()}
    model = branch.LinearBranch(w, heads, dim)
    inputs = (w, rand(frames * per_frame, hidden),
              rand(frames * per_frame, heads, dim),
              rand(frames * per_frame, heads, dim),
              rand(frames * per_frame, heads, dim), frames, per_frame,
              [(max(0, i - 1), min(frames - 1, i + 1)) for i in range(frames)])
    kwargs = dict(frame_size=(2, 2), text_x=rand(7, hidden),
                  text_k_raw=rand(7, heads, dim), text_v_raw=rand(7, heads, dim))
    eager = model.readout(*inputs, **kwargs).clone()
    model.compile_scan = model.fuse_statistics = True
    actual = model.readout(*inputs, **kwargs)
    assert torch.equal(eager, actual)
    assert not model.fuse_epilogue
    assert not any(key[0] in ("scan", "stats_prep") for key in branch._COMPILED_BROKEN)
