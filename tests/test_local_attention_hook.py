"""The optional local-window hook must not replace the trained linear branch."""

from types import SimpleNamespace
from unittest.mock import Mock

import torch

from vdn_h3 import hybrid, window


def fixture():
    torch.manual_seed(7)
    heads, dim, seq = 2, 4, 20
    qkv = torch.randn(seq, heads * dim * 3)
    attn = SimpleNamespace(heads=heads, head_dim=dim, qkv_proj=lambda x: qkv,
                           out_proj=lambda x: x,
                           q_norm=torch.nn.RMSNorm(dim, elementwise_affine=False),
                           k_norm=torch.nn.RMSNorm(dim, elementwise_affine=False))
    readout = Mock(side_effect=lambda w, x, *a, **kw: torch.ones_like(x))
    state = hybrid.VDNState("test", {"enable_softmax_gate": False, "anchor_frames": "both"},
                           [SimpleNamespace(enable_text_state=False, readout=readout)], heads, dim)
    state.layout = hybrid.VDNLayout(2, 18, 8, 2, (1, 2), 0, 2, seq, 0, 2, "both")
    state.retain_buffers = False
    state.weights_on = lambda *a: {"to_out_linear.weight": torch.eye(heads * dim)}
    return hybrid.make_vdn_forward(attn, state, 0), state, readout, torch.zeros(seq, heads * dim)


def test_exact_local_hook_preserves_forward_and_branch():
    forward, state, readout, x = fixture()
    plain = forward(x)
    hook = Mock(side_effect=lambda q, k, v, scale, ranges: window._sdpa(q, k, v, scale))
    patched = forward(x, transformer_options={"vdn_local_attention": lambda i: hook})
    torch.testing.assert_close(patched, plain, rtol=0, atol=0)
    assert hook.call_count > 0
    assert readout.call_count == 2
    assert state._act is None
    window.clear_window_state()


def test_custom_local_hook_leaves_branch_contribution_intact():
    forward, state, readout, x = fixture()
    state.softmax_backend = "flex"
    hook = Mock(side_effect=lambda q, k, v, scale, ranges: torch.zeros_like(q))
    result = forward(x, transformer_options={"vdn_local_attention": lambda i: hook})
    # Non-anchor video queries: zero local attention plus the unchanged unit branch.
    torch.testing.assert_close(result[4:16], torch.ones_like(result[4:16]), rtol=0, atol=0)
    assert readout.call_count == 1
    assert hook.call_count > 0
    assert state.softmax_backend == "flex"  # no persistent mutation of backend choice
    assert state._act is None
    window.clear_window_state()


def test_full_cover_and_excluded_layer_do_not_call_hook():
    forward, state, readout, x = fixture()
    plain = forward(x)
    result = forward(x, transformer_options={"vdn_local_attention": lambda i: None})
    torch.testing.assert_close(result, plain, rtol=0, atol=0)
    state.layout.full_cover = True
    selector = Mock(side_effect=AssertionError("full coverage must not select sparse windows"))
    before = readout.call_count
    forward(x, transformer_options={"vdn_local_attention": selector})
    selector.assert_not_called()
    assert readout.call_count == before
    window.clear_window_state()
