# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""**Stack-testing only** registered model for Phases 2–6 (milestone issue #24).

This is **not** production LLaDA2 inference. Real HF weights and attention live
in issue #12 (Phase 7). Requires a working ``vllm`` + ``torch`` install.

**Outputs:** ``forward`` returns last-hidden-shaped tensors; ``compute_logits``
returns a **non-normalized** logit vector (zeros plus a 1.0 at index 0)—fine for
shape / device / dtype / argmax-bias checks, not for tests that assume a proper
softmax distribution or diverse sampling.

There is **no** ``make_empty_intermediate_tensors``; PP-shaped executor paths may
fail before ``forward`` until DllmWorker / model parity work (milestone issue
#10) adds factory hooks or an early error when PP > 1.

**HF config:** ``architectures`` must include a name registered in
``register_dllm()`` (see ``dllm_plugin.config`` and ``docs/MOCK_STACK_MODEL.md``).
``hidden_size`` and ``vocab_size`` should be set; defaults below apply if absent.

Pipeline-parallel staging is only **partially** mimicked: non-last ranks return
``IntermediateTensors`` with a **zero** residual (not a real transformer
residual). **PP > 1 is unsupported** for this milestone—use single-GPU /
non-PP bring-up until ``make_empty_intermediate_tensors`` and real residuals are
added.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_pp_group
from vllm.sequence import IntermediateTensors

from dllm_plugin.validation import assert_compatible_stack

try:
    # vLLM 0.18+ (approx): ``Attention`` lives in ``attention.py``, re-exported
    # from ``model_executor.layers.attention``.
    from vllm.model_executor.layers.attention import Attention
except ImportError:  # pragma: no cover - depends on vLLM minor layout.
    try:
        from vllm.model_executor.layers.attention.layer import Attention
    except ImportError:  # pragma: no cover
        from vllm.attention.layer import Attention


class DllmMockLlada2ForCausalLM(nn.Module):
    """Minimal causal-LM-shaped module for plugin stack integration tests.

    No ``make_empty_intermediate_tensors``; see module docstring for PP caveats.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        del prefix
        assert_compatible_stack(
            vllm_config, caller="DllmMockLlada2ForCausalLM.__init__"
        )
        hf = vllm_config.model_config.hf_config
        self.hidden_size = int(getattr(hf, "hidden_size", 32))
        self.vocab_size = int(getattr(hf, "vocab_size", 256))
        # Register one attention layer so vLLM KV-cache discovery yields
        # non-empty specs in full engine initialization paths.
        self.mock_attention = Attention(
            num_heads=1,
            head_size=self.hidden_size,
            scale=1.0,
            num_kv_heads=1,
            prefix="layers.0.attn",
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            input_ids.shape[0],
            self.hidden_size,
            device=input_ids.device,
            dtype=torch.get_default_dtype(),
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        del positions, kwargs
        pp = get_pp_group()
        if pp.is_first_rank:
            if inputs_embeds is not None:
                hidden = inputs_embeds
            else:
                assert input_ids is not None
                hidden = self.embed_input_ids(input_ids)
        else:
            assert intermediate_tensors is not None
            hidden = intermediate_tensors["hidden_states"]

        if not pp.is_last_rank:
            residual = torch.zeros_like(hidden)
            return IntermediateTensors(
                {"hidden_states": hidden, "residual": residual},
            )

        return hidden

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        if not get_pp_group().is_last_rank:
            return None
        # Stub logits: not log-prob / softmax-normalized (shape and argmax bias only).
        logits = torch.zeros(
            hidden_states.shape[0],
            self.vocab_size,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        logits[:, 0] = 1.0
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Weightless mock: empty set means nothing to load (not a load failure).
        del weights
        return set()
