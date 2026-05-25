# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical equivalence tests: Triton batched_remask vs PyTorch reference.

Every test asserts exact bit-equality on discrete outputs (draft tokens,
all_done, num_transferred). Confidence values from online softmax may
differ by FP32 accumulation order; if the argmax token and threshold
decision agree, the implementations are functionally equivalent.
"""

import pytest
import torch

from dllm_plugin.sampling.diffusion_sampler import add_gumbel_noise, batched_remask

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton requires CUDA"),
]

MASK_ID = 156895
THRESHOLD = 0.9
VOCAB_SIZE = 157184
BLOCK_SIZE = 32


def _try_import_triton():
    try:
        from dllm_plugin.sampling.triton_kernels import batched_remask_triton

        return batched_remask_triton
    except ImportError:
        pytest.skip("Triton kernel not available")


def _random_inputs(
    batch_size, block_size=BLOCK_SIZE, vocab_size=VOCAB_SIZE, mask_ratio=0.5, seed=42
):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn(
        batch_size, block_size, vocab_size, device="cuda", generator=gen
    )
    draft = torch.full(
        (batch_size, block_size), MASK_ID, dtype=torch.int64, device="cuda"
    )
    n_resolved = int(block_size * (1 - mask_ratio))
    if n_resolved > 0:
        resolved_ids = torch.randint(
            0,
            vocab_size - 1,
            (batch_size, n_resolved),
            device="cuda",
            generator=gen,
        )
        draft[:, :n_resolved] = resolved_ids
    return logits, draft


def _assert_equivalent(ref, tri, name=""):
    draft_r, done_r, transferred_r = ref
    draft_t, done_t, transferred_t = tri
    tag = f" [{name}]" if name else ""
    assert torch.equal(draft_r, draft_t), (
        f"Draft mismatch{tag}: max_diff="
        f"{(draft_r.float() - draft_t.float()).abs().max().item()}"
    )
    assert torch.equal(done_r, done_t), f"all_done mismatch{tag}"
    assert torch.equal(transferred_r, transferred_t), (
        f"num_transferred mismatch{tag}: "
        f"ref={transferred_r.tolist()} tri={transferred_t.tolist()}"
    )


class TestTritonRemaskEquivalence:
    def test_random_inputs(self):
        triton_fn = _try_import_triton()
        for seed in range(20):
            logits, draft = _random_inputs(2, seed=seed)
            ref = batched_remask(logits, draft, MASK_ID, THRESHOLD)
            tri = triton_fn(logits, draft, MASK_ID, THRESHOLD)
            _assert_equivalent(ref, tri, f"seed={seed}")

    def test_all_masked(self):
        triton_fn = _try_import_triton()
        logits = torch.randn(1, BLOCK_SIZE, VOCAB_SIZE, device="cuda")
        draft = torch.full((1, BLOCK_SIZE), MASK_ID, dtype=torch.int64, device="cuda")
        ref = batched_remask(logits, draft, MASK_ID, THRESHOLD)
        tri = triton_fn(logits, draft, MASK_ID, THRESHOLD)
        _assert_equivalent(ref, tri, "all_masked")

    def test_no_masked(self):
        triton_fn = _try_import_triton()
        logits = torch.randn(1, BLOCK_SIZE, VOCAB_SIZE, device="cuda")
        draft = torch.randint(
            0,
            VOCAB_SIZE - 1,
            (1, BLOCK_SIZE),
            dtype=torch.int64,
            device="cuda",
        )
        ref = batched_remask(logits, draft, MASK_ID, THRESHOLD)
        tri = triton_fn(logits, draft, MASK_ID, THRESHOLD)
        _assert_equivalent(ref, tri, "no_masked")

    def test_single_mask(self):
        triton_fn = _try_import_triton()
        logits = torch.randn(1, BLOCK_SIZE, VOCAB_SIZE, device="cuda")
        draft = torch.randint(
            0,
            VOCAB_SIZE - 1,
            (1, BLOCK_SIZE),
            dtype=torch.int64,
            device="cuda",
        )
        draft[0, 15] = MASK_ID
        ref = batched_remask(logits, draft, MASK_ID, THRESHOLD)
        tri = triton_fn(logits, draft, MASK_ID, THRESHOLD)
        _assert_equivalent(ref, tri, "single_mask")

    def test_argmax_is_mask_token(self):
        triton_fn = _try_import_triton()
        logits = torch.randn(1, BLOCK_SIZE, VOCAB_SIZE, device="cuda")
        draft = torch.full((1, BLOCK_SIZE), MASK_ID, dtype=torch.int64, device="cuda")
        logits[:, :, MASK_ID] = 100.0
        ref = batched_remask(logits, draft, MASK_ID, THRESHOLD)
        tri = triton_fn(logits, draft, MASK_ID, THRESHOLD)
        _assert_equivalent(ref, tri, "argmax_is_mask")

    def test_batch_size_8(self):
        triton_fn = _try_import_triton()
        logits, draft = _random_inputs(8, seed=99)
        ref = batched_remask(logits, draft, MASK_ID, THRESHOLD)
        tri = triton_fn(logits, draft, MASK_ID, THRESHOLD)
        _assert_equivalent(ref, tri, "batch8")

    def test_low_threshold(self):
        triton_fn = _try_import_triton()
        logits, draft = _random_inputs(2, mask_ratio=0.8, seed=77)
        ref = batched_remask(logits, draft, MASK_ID, 0.1)
        tri = triton_fn(logits, draft, MASK_ID, 0.1)
        _assert_equivalent(ref, tri, "low_threshold")

    def test_high_threshold(self):
        triton_fn = _try_import_triton()
        logits, draft = _random_inputs(2, mask_ratio=0.8, seed=88)
        ref = batched_remask(logits, draft, MASK_ID, 0.99)
        tri = triton_fn(logits, draft, MASK_ID, 0.99)
        _assert_equivalent(ref, tri, "high_threshold")

    def test_temperature_zero_backward_compat(self):
        """temperature=0.0 must produce identical results to no-temp call."""
        triton_fn = _try_import_triton()
        logits, draft = _random_inputs(2, seed=42)
        ref = batched_remask(logits, draft, MASK_ID, THRESHOLD)
        ref_t0 = batched_remask(logits, draft, MASK_ID, THRESHOLD, temperature=0.0)
        tri_t0 = triton_fn(logits, draft, MASK_ID, THRESHOLD, temperature=0.0)
        _assert_equivalent(ref, ref_t0, "temp0_vs_default")
        _assert_equivalent(ref, tri_t0, "temp0_triton_vs_default")

    def test_temperature_positive_falls_back(self):
        """temperature>0 on Triton wrapper must fall back to PyTorch."""
        triton_fn = _try_import_triton()
        logits, draft = _random_inputs(1, seed=55)
        torch.manual_seed(123)
        ref = batched_remask(logits, draft, MASK_ID, THRESHOLD, temperature=1.0)
        torch.manual_seed(123)
        tri = triton_fn(logits, draft, MASK_ID, THRESHOLD, temperature=1.0)
        _assert_equivalent(ref, tri, "temp1_fallback")


class TestGumbelNoise:
    def test_zero_temperature_is_identity(self):
        logits = torch.randn(2, BLOCK_SIZE, VOCAB_SIZE, device="cuda")
        result = add_gumbel_noise(logits, 0.0)
        assert result is logits

    def test_nonzero_temperature_changes_argmax(self):
        torch.manual_seed(42)
        logits = torch.randn(1, BLOCK_SIZE, VOCAB_SIZE, device="cuda")
        base_argmax = torch.argmax(logits, dim=-1)
        noised = add_gumbel_noise(logits, 1.0)
        noised_argmax = torch.argmax(noised.float(), dim=-1)
        assert not torch.equal(base_argmax, noised_argmax)

    def test_uses_float64(self):
        logits = torch.randn(1, 4, 100, device="cuda")
        result = add_gumbel_noise(logits, 0.5)
        assert result.dtype == torch.float64

    def test_clean_confidence_differs_from_noised(self):
        """Confidence should come from clean logits, not noised."""
        logits = torch.randn(1, BLOCK_SIZE, VOCAB_SIZE, device="cuda")
        draft = torch.full((1, BLOCK_SIZE), MASK_ID, dtype=torch.int64, device="cuda")
        torch.manual_seed(99)
        d1, _, _ = batched_remask(logits, draft, MASK_ID, THRESHOLD, temperature=1.0)
        torch.manual_seed(99)
        d2, _, _ = batched_remask(logits, draft, MASK_ID, THRESHOLD, temperature=0.0)
        assert not torch.equal(d1, d2)
