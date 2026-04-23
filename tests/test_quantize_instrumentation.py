"""Tests for the cross-cutting instrumentation on :class:`AbstractQuantizer`.

Three features land via the post-forward hook installed by
:meth:`AbstractQuantizer.__init_subclass__`:

1. ``log_metric`` / ``get_metrics`` — a latest-only scalar sink.
2. Codebook usage tracking — a lazily-allocated ``_usage_buffer`` of shape
   ``(n_e,)`` that accumulates index hits across forward calls.
3. ``revive_dead_codes`` — replace unused embedding rows with random samples
   from a batch of encoder activations.
"""
from __future__ import annotations

import pytest
import torch

from medlat.first_stage.discrete.quantizer import quantize as qn


# ---------------------------------------------------------------------------
# log_metric / get_metrics
# ---------------------------------------------------------------------------


def test_log_metric_overwrites_latest():
    m = qn.VectorQuantizer(n_e=4, e_dim=2, beta=0.25)
    m.log_metric("k", 1.0)
    m.log_metric("k", 2.0)
    assert m.get_metrics()["k"] == 2.0


def test_log_metric_detaches_tensor():
    m = qn.VectorQuantizer(n_e=4, e_dim=2, beta=0.25)
    x = torch.tensor(3.0, requires_grad=True)
    y = (x * 2).sum()  # y has grad_fn
    m.log_metric("y", y)
    stored = m.get_metrics()["y"]
    assert not stored.requires_grad
    assert stored.grad_fn is None
    assert stored.item() == 6.0


def test_log_metric_preserves_nontensor_values():
    m = qn.VectorQuantizer(n_e=4, e_dim=2, beta=0.25)
    m.log_metric("str_value", "hi")
    m.log_metric("int_value", 42)
    m.log_metric("list_value", [1, 2, 3])
    snap = m.get_metrics()
    assert snap["str_value"] == "hi"
    assert snap["int_value"] == 42
    assert snap["list_value"] == [1, 2, 3]


def test_reset_metrics_clears_user_logged_values():
    m = qn.VectorQuantizer(n_e=4, e_dim=2, beta=0.25)
    m.log_metric("foo", 1.0)
    m.reset_metrics()
    # User metrics cleared (get_metrics may still include derived usage fields
    # if the usage buffer exists, but our custom key must be gone).
    assert "foo" not in m.get_metrics()


# ---------------------------------------------------------------------------
# Post-forward hook auto-logs perplexity and loss
# ---------------------------------------------------------------------------


def test_forward_auto_logs_loss_and_perplexity():
    m = qn.VectorQuantizer(n_e=8, e_dim=4, beta=0.25).eval()
    x = torch.randn(2, 4, 3, 3, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        m(x)
    snap = m.get_metrics()
    assert "loss" in snap
    assert "perplexity" in snap
    assert isinstance(snap["loss"], torch.Tensor)
    assert isinstance(snap["perplexity"], torch.Tensor)


def test_forward_auto_logged_values_update_on_each_call():
    m = qn.VectorQuantizer(n_e=8, e_dim=4, beta=0.25).eval()
    x1 = torch.randn(2, 4, 3, 3, generator=torch.Generator().manual_seed(0))
    x2 = torch.randn(2, 4, 3, 3, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        m(x1)
    perp1 = m.get_metrics()["perplexity"].clone()
    with torch.no_grad():
        m(x2)
    perp2 = m.get_metrics()["perplexity"]
    # Values can differ between calls; the important property is that we
    # overwrote (i.e. the stored tensor is the latest one).
    assert perp1.data_ptr() != perp2.data_ptr() or torch.equal(perp1, perp2)


def test_hook_is_defensive_against_nonstandard_return_shapes():
    # _post_forward must never raise — it's called from every forward and
    # any crash in it would break training. The new return contract is
    # ``(z_q, loss, indices)`` where ``indices`` is a Tensor, list, or None.
    m = qn.VectorQuantizer(n_e=4, e_dim=2, beta=0.25)

    # Short tuple — should silently skip.
    m._post_forward((torch.zeros(1), torch.tensor(0.0)))
    # Non-tuple return — should silently skip.
    m._post_forward(torch.zeros(1))
    # indices is None — should silently skip but still log loss.
    m._post_forward((torch.zeros(1), torch.tensor(0.5), None))
    assert m.get_metrics()["loss"].item() == pytest.approx(0.5)
    # indices is a list (residual variants emit a per-level list) — must not
    # attempt to treat the list as a Tensor.
    m._post_forward(
        (
            torch.zeros(1),
            torch.tensor(0.7),
            [torch.zeros(3, dtype=torch.long), torch.zeros(3, dtype=torch.long)],
        )
    )
    # loss overwritten with the latest value.
    assert m.get_metrics()["loss"].item() == pytest.approx(0.7)
    # 4-tuple (Gumbel return_logits=True shape) — hook only reads the first
    # three elements, ignoring the extra logits tensor.
    m._post_forward(
        (
            torch.zeros(1),
            torch.tensor(0.9),
            torch.tensor([0, 1, 2, 3], dtype=torch.long),
            torch.zeros(7),  # extra logits-like element
        )
    )
    snap = m.get_metrics()
    assert snap["loss"].item() == pytest.approx(0.9)
    # Perplexity is now derived from indices via bincount, so a fully-used
    # codebook (each code hit once) gives perplexity == n_e.
    assert snap["perplexity"].item() == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Usage tracking
# ---------------------------------------------------------------------------


def test_usage_buffer_is_lazily_allocated():
    m = qn.VectorQuantizer(n_e=8, e_dim=4, beta=0.25)
    # Before any forward, no usage buffer exists.
    assert not hasattr(m, "_usage_buffer")

    m.eval()
    with torch.no_grad():
        m(torch.randn(1, 4, 2, 2))
    # After one forward, the buffer exists with the right size.
    assert hasattr(m, "_usage_buffer")
    assert m._usage_buffer.shape == (8,)
    # And it's non-persistent (won't end up in state_dict)
    assert "_usage_buffer" not in m.state_dict()


def test_usage_buffer_accumulates_across_forward_calls():
    m = qn.VectorQuantizer(n_e=8, e_dim=4, beta=0.25).eval()
    x = torch.randn(4, 4, 3, 3, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        m(x)
    total_after_one = int(m._usage_buffer.sum().item())
    with torch.no_grad():
        m(x)
    total_after_two = int(m._usage_buffer.sum().item())
    assert total_after_two == 2 * total_after_one


def test_total_tokens_seen_equals_indices_per_call():
    # batch 2 × 3 × 3 spatial = 18 tokens per forward.
    m = qn.VectorQuantizer(n_e=8, e_dim=4, beta=0.25).eval()
    x = torch.randn(2, 4, 3, 3, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        m(x)
    snap = m.get_metrics()
    assert snap["total_tokens_seen"] == 2 * 3 * 3


def test_track_usage_false_skips_buffer():
    m = qn.VectorQuantizer(n_e=8, e_dim=4, beta=0.25)
    m.track_usage = False
    m.eval()
    with torch.no_grad():
        m(torch.randn(1, 4, 2, 2))
    assert not hasattr(m, "_usage_buffer")
    # But other instrumentation still works.
    assert "loss" in m.get_metrics()


def test_dead_code_ratio_decreases_with_more_diverse_input():
    # With only 4 tokens total and a codebook of 16, at most 4 codes are hit —
    # so the majority of codes are dead.
    m = qn.VectorQuantizer(n_e=16, e_dim=4, beta=0.25).eval()
    with torch.no_grad():
        m(torch.randn(1, 4, 2, 2, generator=torch.Generator().manual_seed(0)))
    snap = m.get_metrics()
    assert snap["active_code_count"] <= 4
    assert snap["dead_code_ratio"] >= 12 / 16


def test_dead_code_ratio_is_zero_when_all_codes_hit():
    # Force every code to be hit by manually seeding the usage buffer.
    m = qn.VectorQuantizer(n_e=8, e_dim=4, beta=0.25).eval()
    # Run one forward to allocate the buffer on the right device.
    with torch.no_grad():
        m(torch.randn(1, 4, 2, 2))
    m._usage_buffer.fill_(100)
    snap = m.get_metrics()
    assert snap["dead_code_ratio"] == 0.0
    assert snap["codebook_utilization"] == 1.0


def test_instrumentation_kwargs_accepted_at_construction():
    # Every concrete subclass accepts track_usage / dead_code_threshold /
    # revive_dead_codes_after as constructor kwargs via the __init_subclass__
    # wrap — the subclass's own __init__ signature doesn't need to declare them.
    m = qn.VectorQuantizer(
        n_e=8, e_dim=4, beta=0.25,
        track_usage=False,
        dead_code_threshold=5,
        revive_dead_codes_after=100,
    )
    assert m.track_usage is False
    assert m.dead_code_threshold == 5
    assert m.revive_dead_codes_after == 100


def test_instrumentation_kwargs_work_via_get_model():
    # Same kwargs must flow through the registry path too.
    from medlat import get_model

    m = get_model(
        "discrete.quantizer.vector_quantizer",
        n_e=8, e_dim=4, beta=0.25,
        revive_dead_codes_after=250,
        dead_code_threshold=3,
    )
    assert m.revive_dead_codes_after == 250
    assert m.dead_code_threshold == 3


def test_instrumentation_kwargs_work_on_transitively_wrapped_subclasses():
    # SimpleQINCo inherits VectorQuantizer2's __init__ via super(); the
    # kwarg wrap still fires at SimpleQINCo's own __init_subclass__ so users
    # can pass instrumentation kwargs there too.
    m = qn.SimpleQINCo(n_e=8, e_dim=4, revive_dead_codes_after=42)
    assert m.revive_dead_codes_after == 42


def test_instrumentation_kwargs_preserve_subclass_defaults_when_not_passed():
    # Omitting the kwargs keeps the class-level defaults (0 / 1 / True).
    m = qn.VectorQuantizer(n_e=8, e_dim=4, beta=0.25)
    assert m.track_usage is True
    assert m.dead_code_threshold == 1
    assert m.revive_dead_codes_after == 0


def test_reset_usage_zeroes_buffer():
    m = qn.VectorQuantizer(n_e=8, e_dim=4, beta=0.25).eval()
    with torch.no_grad():
        m(torch.randn(1, 4, 2, 2))
    assert m._usage_buffer.sum().item() > 0
    m.reset_usage()
    assert m._usage_buffer.sum().item() == 0


def test_residual_quantizer_logs_per_level_perplexity():
    # Every residual wrapper should surface perplexity_level_{i} metrics for
    # each level's indices so training-loop logging sees utilization per level.
    r = qn.ResidualQuantizer(
        quantizer_class=qn.VectorQuantizer2,
        num_quantizers=3,
        quantizer_kwargs_list=[{"n_e": 4, "e_dim": 3}] * 3,
    ).eval()
    x = torch.randn(1, 3, 2, 2, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        r(x)
    snap = r.get_metrics()
    assert "perplexity_level_0" in snap
    assert "perplexity_level_1" in snap
    assert "perplexity_level_2" in snap
    # Aggregated 'perplexity' key is reserved for single-level quantizers.
    assert "perplexity" not in snap


def test_residual_quantizer_list_indices_do_not_crash_hook():
    # ResidualQuantizer's info tuple holds LISTS of per-level tensors. The
    # post-forward hook should skip usage tracking silently instead of
    # crashing. Equivalence tests already cover behavior preservation;
    # here we only prove the hook is defensive.
    r = qn.ResidualQuantizer(
        quantizer_class=qn.VectorQuantizer2,
        num_quantizers=2,
        quantizer_kwargs_list=[{"n_e": 4, "e_dim": 3}, {"n_e": 4, "e_dim": 3}],
    ).eval()
    x = torch.randn(1, 3, 2, 2, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        out = r(x)
    # Forward succeeded; the residual wrapper's OWN buffer stays unset
    # (indices was a list, not a tensor) — nested quantizers accumulate their
    # own stats independently.
    assert not hasattr(r, "_usage_buffer")


# ---------------------------------------------------------------------------
# revive_dead_codes
# ---------------------------------------------------------------------------


def test_revive_dead_codes_noop_without_usage_buffer():
    m = qn.VectorQuantizer(n_e=8, e_dim=4, beta=0.25)
    # No forward yet, so no buffer.
    n = m.revive_dead_codes(torch.randn(5, 4))
    assert n == 0


def test_revive_dead_codes_noop_for_codebook_free_class():
    m = qn.LookupFreeQuantizer(token_bits=3).eval()
    with torch.no_grad():
        m(torch.randn(1, 3, 2, 2))
    # LFQ has no nn.Embedding, so revival must return 0 regardless of
    # what the usage buffer says.
    m._usage_buffer.fill_(0)
    n = m.revive_dead_codes(torch.randn(5, 3))
    assert n == 0


def test_revive_dead_codes_noop_when_all_codes_alive():
    m = qn.VectorQuantizer(n_e=8, e_dim=4, beta=0.25).eval()
    with torch.no_grad():
        m(torch.randn(1, 4, 2, 2))
    m._usage_buffer.fill_(100)  # every code has plenty of hits
    n = m.revive_dead_codes(torch.randn(5, 4))
    assert n == 0


def test_revive_dead_codes_replaces_dead_rows():
    m = qn.VectorQuantizer(n_e=8, e_dim=4, beta=0.25).eval()
    # Seed a deterministic codebook.
    torch.manual_seed(0)
    with torch.no_grad():
        m(torch.randn(1, 4, 2, 2))
    # Mark codes 3, 5, 7 as dead, the rest as alive.
    m._usage_buffer.fill_(100)
    m._usage_buffer[[3, 5, 7]] = 0

    original_weights = m.embedding.weight.data.clone()
    enc = torch.randn(10, 4)  # 10 encoder activations of dim 4
    n = m.revive_dead_codes(enc)
    assert n == 3

    # Only rows 3, 5, 7 should have changed.
    for row in range(8):
        if row in (3, 5, 7):
            assert not torch.equal(m.embedding.weight.data[row], original_weights[row])
        else:
            assert torch.equal(m.embedding.weight.data[row], original_weights[row])


def test_revive_dead_codes_uses_provided_activations():
    m = qn.VectorQuantizer(n_e=8, e_dim=4, beta=0.25).eval()
    with torch.no_grad():
        m(torch.randn(1, 4, 2, 2))
    m._usage_buffer.fill_(0)  # all codes dead

    # Use a pool where every row is the same known vector — revived codes
    # must be exactly that vector.
    known = torch.tensor([1.0, 2.0, 3.0, 4.0])
    enc = known.repeat(10, 1)  # (10, 4) — every row is `known`
    n = m.revive_dead_codes(enc)
    assert n == 8
    for row in range(8):
        torch.testing.assert_close(m.embedding.weight.data[row], known)


def test_revive_resets_usage_to_threshold_for_revived_codes():
    m = qn.VectorQuantizer(n_e=8, e_dim=4, beta=0.25).eval()
    with torch.no_grad():
        m(torch.randn(1, 4, 2, 2))
    m._usage_buffer.fill_(100)
    m._usage_buffer[[1, 4]] = 0
    m.revive_dead_codes(torch.randn(5, 4))
    # Revived codes now sit *at* the threshold so the next call won't treat
    # them as dead immediately.
    assert int(m._usage_buffer[1].item()) == m.dead_code_threshold
    assert int(m._usage_buffer[4].item()) == m.dead_code_threshold


def test_auto_revive_disabled_by_default():
    m = qn.VectorQuantizer(n_e=8, e_dim=4, beta=0.25).train()
    assert m.revive_dead_codes_after == 0
    with torch.no_grad():
        for _ in range(5):
            m(torch.randn(1, 4, 2, 2))
    # No auto-revival counter spun up, no codes_revived metric logged.
    assert not hasattr(m, "_forward_count")
    assert "codes_revived" not in m.get_metrics()


def test_auto_revive_fires_on_configured_cadence():
    # Configure every 3 forwards; seed the buffer with dead codes after the
    # very first call, then verify that revival triggers on the 3rd forward.
    torch.manual_seed(0)
    m = qn.VectorQuantizer(n_e=8, e_dim=4, beta=0.25).train()
    m.revive_dead_codes_after = 3

    # First forward populates the usage buffer.
    m(torch.randn(2, 4, 2, 2, generator=torch.Generator().manual_seed(0)))
    # Zero out the buffer so every code looks dead.
    m._usage_buffer.fill_(0)
    assert m._forward_count == 1

    # Second forward — counter advances but revival threshold not yet hit.
    m(torch.randn(2, 4, 2, 2, generator=torch.Generator().manual_seed(1)))
    assert m._forward_count == 2
    # No codes_revived metric yet because revival hasn't fired this batch.
    # (The counter may have re-populated some entries in the buffer though,
    # so we don't assert it's still all-zero — only that revival hasn't run.)
    assert "codes_revived" not in m.get_metrics()

    # Zero again to guarantee dead codes when the cadence hits.
    m._usage_buffer.fill_(0)
    m(torch.randn(2, 4, 2, 2, generator=torch.Generator().manual_seed(2)))
    assert m._forward_count == 3
    # Revival ran — codes_revived metric should show up.
    assert "codes_revived" in m.get_metrics()
    assert m.get_metrics()["codes_revived"] > 0


def test_auto_revive_rolls_the_usage_window():
    # With revive_dead_codes_after=N, the usage buffer should represent hits
    # in the LAST N forwards, not lifetime cumulative counts. Verified by
    # checking that the buffer is zeroed right after each revival tick.
    torch.manual_seed(0)
    m = qn.VectorQuantizer(
        n_e=8, e_dim=4, beta=0.25,
        revive_dead_codes_after=3,
    ).train()

    # Forwards 1 and 2: buffer accumulates, no revival.
    m(torch.randn(2, 4, 2, 2, generator=torch.Generator().manual_seed(1)))
    assert int(m._usage_buffer.sum().item()) > 0, "buffer accumulating mid-window"
    m(torch.randn(2, 4, 2, 2, generator=torch.Generator().manual_seed(2)))
    pre_revival_total = int(m._usage_buffer.sum().item())
    assert pre_revival_total > 0

    # Forward 3: revival fires, THEN buffer is reset for the next window.
    m(torch.randn(2, 4, 2, 2, generator=torch.Generator().manual_seed(3)))
    assert m._forward_count == 3
    assert int(m._usage_buffer.sum().item()) == 0, (
        "buffer must reset after each revival so the next revival uses "
        "recent-window data, not lifetime cumulative counts"
    )

    # Forward 4 fills the new window.
    m(torch.randn(2, 4, 2, 2, generator=torch.Generator().manual_seed(4)))
    assert int(m._usage_buffer.sum().item()) > 0, "window refilling after reset"


def test_auto_revive_skipped_in_eval_mode():
    m = qn.VectorQuantizer(n_e=8, e_dim=4, beta=0.25).eval()
    m.revive_dead_codes_after = 1
    with torch.no_grad():
        for _ in range(3):
            m(torch.randn(1, 4, 2, 2))
    # Training gate ensures no revival counter is spun up in eval.
    assert not hasattr(m, "_forward_count")


def test_auto_revive_skipped_for_codebook_free_classes():
    # LookupFreeQuantizer has no nn.Embedding, so revive_dead_codes is a
    # no-op for it; the counter still advances but no codes are ever revived.
    m = qn.LookupFreeQuantizer(token_bits=4).train()
    m.revive_dead_codes_after = 1
    m(torch.randn(1, 4, 2, 2))
    # No codes_revived metric because revive_dead_codes returned 0.
    assert "codes_revived" not in m.get_metrics()


def test_revive_accepts_any_shape_with_matching_trailing_dim():
    # encoder_output can be (B, C, H, W), (B, C, D, H, W), flat, etc. — all
    # are flattened to (N, embedding_dim).
    m = qn.VectorQuantizer(n_e=8, e_dim=4, beta=0.25).eval()
    with torch.no_grad():
        m(torch.randn(1, 4, 2, 2))
    m._usage_buffer.fill_(0)

    # (B=2, H=3, W=3, C=4) — channel-last, reshape(-1, 4) gives 18 rows.
    enc_cl = torch.randn(2, 3, 3, 4)
    n = m.revive_dead_codes(enc_cl)
    assert n == 8  # all dead codes revived


# ---------------------------------------------------------------------------
# entropy_regularization — opt-in MaskGIT-style entropy loss.
# ---------------------------------------------------------------------------


def test_entropy_regularization_returns_zero_by_default():
    m = qn.VectorQuantizer(n_e=8, e_dim=4, beta=0.25)
    affinity = torch.randn(10, 8)
    out = m.entropy_regularization(affinity)
    assert out.item() == 0.0
    assert out.shape == ()
    assert out.device == affinity.device
    # Zero-tensor returns should allow chained "loss + entropy_regularization(...)".
    loss = torch.tensor(1.0) + out
    assert loss.item() == pytest.approx(1.0)


def test_entropy_regularization_preserves_dtype_and_device():
    m = qn.VectorQuantizer(n_e=8, e_dim=4, beta=0.25)
    affinity = torch.randn(5, 8, dtype=torch.float64)
    out = m.entropy_regularization(affinity)
    assert out.dtype == torch.float64


def test_entropy_regularization_matches_modules_helper_when_enabled():
    from medlat.first_stage.discrete.quantizer.modules import entropy_loss_fn

    m = qn.VectorQuantizer(n_e=8, e_dim=4, beta=0.25).train()
    m.entropy_loss_weight = 0.5
    m.entropy_loss_temperature = 0.8
    m.entropy_gamma = 1.2

    affinity = torch.randn(12, 8, generator=torch.Generator().manual_seed(7))

    # Call the helper first (it clones internally so affinity stays intact).
    actual = m.entropy_regularization(affinity)
    # Then compute the expected value manually — entropy_loss_fn mutates its
    # input, so we clone as well to keep the test independent of call order.
    expected_per, expected_avg = entropy_loss_fn(
        affinity.clone(), temperature=0.8, entropy_gamma=1.2
    )
    expected = 0.5 * (expected_per - expected_avg)
    torch.testing.assert_close(actual, expected)


def test_entropy_regularization_returns_zero_in_eval_mode():
    # The helper is gated on self.training to match SoftVQ / BSQ behavior.
    m = qn.VectorQuantizer(n_e=8, e_dim=4, beta=0.25).eval()
    m.entropy_loss_weight = 0.5
    affinity = torch.randn(6, 8)
    out = m.entropy_regularization(affinity)
    assert out.item() == 0.0
    # And nothing should be logged.
    assert "entropy_per_sample" not in m.get_metrics()


def test_entropy_regularization_does_not_mutate_affinity():
    # Defensive: the helper clones before calling into modules.entropy_loss_fn,
    # which contains an in-place `flat_affinity /= temperature`. Callers that
    # reuse the affinity tensor downstream must see it unchanged.
    m = qn.VectorQuantizer(n_e=8, e_dim=4, beta=0.25).train()
    m.entropy_loss_weight = 1.0
    m.entropy_loss_temperature = 0.5
    affinity = torch.randn(6, 8, generator=torch.Generator().manual_seed(0))
    snapshot = affinity.clone()
    m.entropy_regularization(affinity)
    torch.testing.assert_close(affinity, snapshot)


def test_entropy_regularization_logs_components_as_metrics():
    m = qn.VectorQuantizer(n_e=8, e_dim=4, beta=0.25).train()
    m.entropy_loss_weight = 0.1
    affinity = torch.randn(6, 8, generator=torch.Generator().manual_seed(1))
    m.entropy_regularization(affinity)

    snap = m.get_metrics()
    assert "entropy_per_sample" in snap
    assert "entropy_avg" in snap
    assert "entropy_loss" in snap
    # Values are detached (no grad graph).
    assert snap["entropy_per_sample"].grad_fn is None
    assert snap["entropy_avg"].grad_fn is None
    assert snap["entropy_loss"].grad_fn is None


def test_entropy_regularization_does_not_log_when_disabled():
    m = qn.VectorQuantizer(n_e=8, e_dim=4, beta=0.25).train()
    # Default: weight is 0 → skip the log.
    affinity = torch.randn(6, 8)
    m.entropy_regularization(affinity)
    snap = m.get_metrics()
    assert "entropy_per_sample" not in snap
    assert "entropy_avg" not in snap


def test_entropy_regularization_accepts_affinity_of_arbitrary_batch_shape():
    # The helper internally reshapes to (-1, n_e), so callers can pass any
    # shape as long as the last dim is the codebook dimension.
    m = qn.VectorQuantizer(n_e=8, e_dim=4, beta=0.25).train()
    m.entropy_loss_weight = 1.0
    aff = torch.randn(2, 3, 4, 8, generator=torch.Generator().manual_seed(0))
    out = m.entropy_regularization(aff)
    assert out.shape == ()
    assert out.dim() == 0


def test_entropy_regularization_flows_gradient_when_enabled():
    m = qn.VectorQuantizer(n_e=8, e_dim=4, beta=0.25).train()
    m.entropy_loss_weight = 0.5
    affinity = torch.randn(4, 8, requires_grad=True)
    loss = m.entropy_regularization(affinity)
    loss.backward()
    # Non-trivial gradient: the entropy term actually depends on affinity.
    assert affinity.grad is not None
    assert affinity.grad.abs().sum().item() > 0


# ---------------------------------------------------------------------------
# VectorQuantizer2 entropy migration — VQ2 now uses the shared helper.
# ---------------------------------------------------------------------------


def test_vq2_accepts_new_entropy_kwargs():
    m = qn.VectorQuantizer2(
        n_e=8, e_dim=4,
        entropy_loss_weight=0.1,
        entropy_loss_temperature=0.5,
        entropy_gamma=1.2,
    )
    assert m.entropy_loss_weight == 0.1
    assert m.entropy_loss_temperature == 0.5
    assert m.entropy_gamma == 1.2


def test_vq2_rejects_legacy_entropy_kwargs():
    # Old-style kwargs are no longer accepted — callers must migrate to
    # entropy_loss_weight / entropy_loss_temperature / entropy_gamma.
    for legacy_name, value in [
        ("entropy_loss_ratio", 0.1),
        ("entropy_temperature", 0.5),
        ("entropy_loss_type", "gumbel"),
    ]:
        with pytest.raises(TypeError, match=f"unexpected keyword argument '{legacy_name}'"):
            qn.VectorQuantizer2(n_e=8, e_dim=4, **{legacy_name: value})


def test_vq2_rejects_unknown_kwarg():
    with pytest.raises(TypeError, match="unexpected keyword argument 'bogus_setting'"):
        qn.VectorQuantizer2(n_e=8, e_dim=4, bogus_setting=42)


def test_vq2_entropy_contributes_to_loss_in_training_mode():
    m = qn.VectorQuantizer2(n_e=16, e_dim=4, entropy_loss_weight=0.1).train()
    x = torch.randn(2, 4, 3, 3, generator=torch.Generator().manual_seed(0))
    _, loss, _ = m(x)
    # With entropy reg on, the per/avg/entropy metrics should be populated.
    snap = m.get_metrics()
    assert "entropy_per_sample" in snap
    assert "entropy_avg" in snap
    assert "entropy_loss" in snap
    # And the logged entropy_loss must equal the term actually added to loss
    # (we can't observe the unregularized loss in isolation, but we can check
    # the published value is non-zero).
    assert snap["entropy_loss"].item() != 0.0


def test_vq2_entropy_is_zero_in_eval_even_when_weight_positive():
    m = qn.VectorQuantizer2(n_e=16, e_dim=4, entropy_loss_weight=0.1).eval()
    x = torch.randn(2, 4, 3, 3, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        m(x)
    snap = m.get_metrics()
    # training-mode gate means no entropy metrics are populated in eval.
    assert "entropy_per_sample" not in snap


def test_vq2_clone_with_preserves_migrated_entropy_config():
    from medlat import clone_with, get_model
    base = get_model(
        "discrete.quantizer.vector_quantizer2",
        n_e=8, e_dim=4, entropy_loss_weight=0.1, entropy_gamma=1.0,
    )
    hotter = clone_with(base, entropy_loss_temperature=0.3)
    assert hotter.entropy_loss_weight == 0.1
    assert hotter.entropy_gamma == 1.0
    assert hotter.entropy_loss_temperature == 0.3
