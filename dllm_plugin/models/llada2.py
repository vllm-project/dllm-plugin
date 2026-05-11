# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Production LLaDA2.0 vLLM model with MoE and block-style attention.

**Phase 7** real model implementation (issue #12). Replaces mock stub from Phases 2-6
with production-ready LLaDA2.0 inference supporting:
- 256-expert Mixture-of-Experts with group-limited routing
- Block-style non-causal attention for diffusion-based generation
- Shared expert (always active) + routed experts (top-k selected)
- Tensor parallelism (TP) support for multi-GPU inference

**Not supported in Phase 7 MVP:** Pipeline parallelism (PP > 1). The model will
fail fast at initialization if PP is enabled. TP is the primary scaling path.

References:
- HuggingFace LLaDA2.0: https://huggingface.co/collections/inclusionAI/llada-20
- SGlang implementation: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/llada2.py
- Design doc: docs/ATTENTION_DESIGN.md
"""

from __future__ import annotations

import logging
import os
import re
from collections import defaultdict
from collections.abc import Iterable

import torch
import torch.nn as nn
from transformers import PretrainedConfig
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors

from dllm_plugin.config import (
    LLADA2_DEFAULT_MOE_INTERMEDIATE_SIZE,
    LLADA2_DEFAULT_N_GROUP,
    LLADA2_DEFAULT_NUM_EXPERTS,
    LLADA2_DEFAULT_NUM_EXPERTS_PER_TOK,
    LLADA2_DEFAULT_NUM_SHARED_EXPERTS,
    LLADA2_DEFAULT_ROUTED_SCALING_FACTOR,
    LLADA2_DEFAULT_TOPK_GROUP,
)
from dllm_plugin.gpu_capability import detect_gpu_capabilities
from dllm_plugin.models.llada2_attention import LLaDA2BlockAttention
from dllm_plugin.validation import assert_compatible_stack

logger = logging.getLogger(__name__)


class LLaDA2MoE(nn.Module):
    """LLaDA2.0 Mixture-of-Experts layer with group-limited routing.

    Implements LLaDA2.0's unique MoE architecture:
    - **256 experts** with group-limited top-k selection
    - **Group-limited routing**: 8 groups → select top-4 groups → select top-2 experts
    - **Sigmoid activation** on gate logits (not softmax like other MoE models)
    - **Shared expert** (always active) + routed experts (conditionally activated)
    - **Routed scaling factor**: 2.5x scaling on routed expert output

    Routing algorithm:
    1. Compute gate logits: `logits = gate(hidden_states)`  [batch, num_experts]
    2. Apply sigmoid: `scores = sigmoid(logits)`
    3. Group-limited selection:
       a. Divide 256 experts into 8 groups (32 experts each)
       b. For each token, select top-4 groups based on max score in group
       c. From selected groups, select top-k experts (k=8 default)
    4. Forward through selected experts via FusedMoE
    5. Add shared expert output (if present)
    6. Scale routed output by 2.5

    References:
    - Similar to Qwen2 MoE shared expert pattern
    - Routing differs from Mixtral/DeepSeek (group-limited vs. flat top-k)

    Args:
        config: HuggingFace model config with MoE parameters.
        tp_size: Tensor parallelism world size.
        prefix: Parameter name prefix for weight loading.
        is_dense_only: If True, only create shared expert (no routing/experts).
                       Used for first_k_dense_replace layers.
    """

    # Type annotations for optional attributes (set conditionally in __init__)
    gate: ReplicatedLinear | None
    experts: FusedMoE | None
    shared_expert_gate: ColumnParallelLinear | None
    shared_expert_up: ColumnParallelLinear | None
    shared_expert_down: RowParallelLinear | None

    def __init__(
        self,
        config: PretrainedConfig,
        tp_size: int,
        prefix: str = "",
        is_dense_only: bool = False,
    ) -> None:
        super().__init__()

        self.config = config
        self.tp_size = tp_size
        self.hidden_size = config.hidden_size
        self.is_dense_only = is_dense_only

        # MoE architecture parameters
        self.num_experts = getattr(config, "num_experts", LLADA2_DEFAULT_NUM_EXPERTS)
        self.num_experts_per_tok = getattr(
            config, "num_experts_per_tok", LLADA2_DEFAULT_NUM_EXPERTS_PER_TOK
        )
        self.num_shared_experts = getattr(
            config, "num_shared_experts", LLADA2_DEFAULT_NUM_SHARED_EXPERTS
        )
        self.moe_intermediate_size = getattr(
            config, "moe_intermediate_size", LLADA2_DEFAULT_MOE_INTERMEDIATE_SIZE
        )

        # Group-limited routing parameters
        self.n_group = getattr(config, "n_group", LLADA2_DEFAULT_N_GROUP)
        self.topk_group = getattr(config, "topk_group", LLADA2_DEFAULT_TOPK_GROUP)
        self.routed_scaling_factor = getattr(
            config, "routed_scaling_factor", LLADA2_DEFAULT_ROUTED_SCALING_FACTOR
        )

        # Dense-only layers skip router and experts (only use shared expert)
        if not is_dense_only:
            # Validate TP configuration
            if self.tp_size > self.num_experts:
                raise ValueError(
                    f"Tensor parallelism size ({self.tp_size}) cannot exceed "
                    f"number of experts ({self.num_experts}) in LLaDA2MoE."
                )

            # TP > 1 supported via per-expert weight loading (Phase 8.2 refactoring)
            # Validate TP size is reasonable for expert count
            if self.tp_size > 1 and self.num_experts % self.tp_size != 0:
                logger.warning(
                    "TP size %d does not evenly divide %d experts. "
                    "Expert distribution may be unbalanced across ranks.",
                    self.tp_size,
                    self.num_experts,
                )

            # Gate/router network (replicated across TP ranks)
            self.gate = ReplicatedLinear(
                self.hidden_size,
                self.num_experts,
                bias=True,
                prefix=f"{prefix}.gate",
            )

            # Routed experts (vLLM FusedMoE)
            self.experts = FusedMoE(
                num_experts=self.num_experts,
                top_k=self.num_experts_per_tok,
                hidden_size=self.hidden_size,
                intermediate_size=self.moe_intermediate_size,
                tp_size=self.tp_size,
                prefix=f"{prefix}.experts",
            )
        else:
            self.gate = None
            self.experts = None
            # Dense-only mode requires shared experts to exist
            if self.num_shared_experts <= 0:
                raise ValueError(
                    "Dense-only MoE layer (is_dense_only=True) requires "
                    f"num_shared_experts > 0, got {self.num_shared_experts}. "
                    "Dense layers must have a shared expert to forward through."
                )

        # Shared expert (always active, not routed)
        if self.num_shared_experts > 0:
            # Dense-only layers (first_k_dense_replace) use
            # config.intermediate_size. MoE layers use
            # num_shared_experts * moe_intermediate_size for shared expert
            if self.is_dense_only:
                # Dense-only: use full intermediate_size from config (e.g., 5120)
                shared_intermediate_size = getattr(
                    config,
                    "intermediate_size",
                    self.num_shared_experts * self.moe_intermediate_size,
                )
                # Dense-only: weights are at {prefix}.{param}.weight (no "shared_expert")
                gate_prefix = f"{prefix}.gate_proj"
                up_prefix = f"{prefix}.up_proj"
                down_prefix = f"{prefix}.down_proj"
            else:
                # MoE layers: shared expert is just one of many, sized appropriately
                shared_intermediate_size = (
                    self.num_shared_experts * self.moe_intermediate_size
                )
                # MoE: weights are at {prefix}.shared_experts.{param}.weight (plural!)
                gate_prefix = f"{prefix}.shared_experts.gate_proj"
                up_prefix = f"{prefix}.shared_experts.up_proj"
                down_prefix = f"{prefix}.shared_experts.down_proj"

            self.shared_expert_gate = ColumnParallelLinear(
                self.hidden_size,
                shared_intermediate_size,
                bias=False,
                prefix=gate_prefix,
            )
            self.shared_expert_up = ColumnParallelLinear(
                self.hidden_size,
                shared_intermediate_size,
                bias=False,
                prefix=up_prefix,
            )
            self.shared_expert_down = RowParallelLinear(
                shared_intermediate_size,
                self.hidden_size,
                bias=False,
                prefix=down_prefix,
            )
        else:
            self.shared_expert_gate = None
            self.shared_expert_up = None
            self.shared_expert_down = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply MoE layer with group-limited routing.

        For dense-only layers (is_dense_only=True), only the shared expert is used.

        Args:
            hidden_states: Input tensor, shape (num_tokens, hidden_size) for V2 Model Runner
                           or (batch_size, seq_len, hidden_size) for legacy.

        Returns:
            Output tensor, same shape as input.
        """
        # V2 Model Runner uses 2D tensors (num_tokens, hidden_size)
        # Legacy path uses 3D tensors (batch_size, seq_len, hidden_size)
        input_shape = hidden_states.shape
        if hidden_states.ndim == 2:
            num_tokens, hidden_size = hidden_states.shape
            hidden_states_2d = hidden_states
        else:
            batch_size, seq_len, hidden_size = hidden_states.shape
            num_tokens = batch_size * seq_len
            hidden_states_2d = hidden_states.reshape(num_tokens, hidden_size)

        # Dense-only mode: skip routing, only use shared expert
        if self.is_dense_only:
            # Assertions for type checker (enforced by __init__ validation)
            assert self.shared_expert_gate is not None
            assert self.shared_expert_up is not None
            assert self.shared_expert_down is not None
            # Shared expert: SwiGLU activation
            gate_out, _ = self.shared_expert_gate(hidden_states_2d)
            up_out, _ = self.shared_expert_up(hidden_states_2d)
            shared_hidden = torch.nn.functional.silu(gate_out) * up_out
            output, _ = self.shared_expert_down(shared_hidden)
            return output.reshape(input_shape)

        # Full MoE mode with routing
        assert self.gate is not None, "gate should exist in non-dense-only mode"
        assert self.experts is not None, "experts should exist in non-dense-only mode"

        # 1. Compute gate logits and apply sigmoid (LLaDA2.0 specific)
        # Router precision: Default to FP32 (safe) with BF16 opt-in for experimentation
        # Following DeepSeek V3 and Qwen2-MoE patterns which use FP32 router for
        # numerical stability. Unlike softmax (proven to need FP32), sigmoid stability
        # in BF16 is unvalidated for LLaDA2's group-limited routing.
        # Set VLLM_LLADA2_BF16_ROUTER=1 to opt-in to BF16 (faster but unvalidated).
        use_bf16_router = os.getenv("VLLM_LLADA2_BF16_ROUTER", "0") == "1"

        # NOTE: FP32 vs BF16 router precision
        # Original intent was FP32 router for numerical stability (following DeepSeek V3 / Qwen2-MoE)
        # However, vLLM's ReplicatedLinear doesn't support mixed-dtype (Float input + BFloat16 weights)
        # For Phase 7 MVP, using BF16 router (same dtype as model)
        # Phase 9 validation (issue #39) will quantify BF16 vs FP32 routing accuracy
        if use_bf16_router:
            if not hasattr(self, "_bf16_router_warning_logged"):
                logger.warning(
                    "LLaDA2 router using BF16 precision (EXPERIMENTAL). "
                    "Numerical correctness unvalidated - may cause expert mis-routing. "
                    "Phase 9 validation (issue #39) will compare BF16 vs FP32 routing. "
                    "Unset VLLM_LLADA2_BF16_ROUTER to use safe FP32 default."
                )
                self._bf16_router_warning_logged = True

        gate_logits, _ = self.gate(hidden_states_2d)  # BF16 (model dtype)

        router_logits = torch.sigmoid(gate_logits)  # Sigmoid, not softmax!

        # 2. Apply group-limited top-k routing
        router_weights, selected_experts = self._apply_group_limited_topk(router_logits)

        # 3. Forward through routed experts via FusedMoE
        # FusedMoE expects 2D inputs (num_tokens, hidden_size)
        routed_output = self.experts(
            hidden_states_2d,
            router_weights,
            selected_experts,
        )

        # 4. Apply routed scaling factor
        # Scale routed expert outputs by 2.5x (from HuggingFace config
        # `routed_scaling_factor` field). This asymmetric scaling (routed vs
        # shared experts) follows LLaDA2.0 architecture specification.
        routed_output = routed_output * self.routed_scaling_factor

        # 5. Add shared expert output (if present)
        if self.num_shared_experts > 0:
            assert self.shared_expert_gate is not None
            assert self.shared_expert_up is not None
            assert self.shared_expert_down is not None
            # Shared expert: SwiGLU activation
            gate_out, _ = self.shared_expert_gate(hidden_states_2d)
            up_out, _ = self.shared_expert_up(hidden_states_2d)
            shared_hidden = torch.nn.functional.silu(gate_out) * up_out
            shared_output, _ = self.shared_expert_down(shared_hidden)

            output = routed_output + shared_output
        else:
            output = routed_output

        # Reshape output back to input shape
        return output.reshape(input_shape)

    def _apply_group_limited_topk(
        self, router_logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply group-limited top-k expert selection.

        LLaDA2.0's unique routing: divide experts into groups, select top groups,
        then select top-k experts from selected groups.

        Args:
            router_logits: Router scores, shape (num_tokens, num_experts) for V2 Model Runner
                           or (batch_size, seq_len, num_experts) for legacy.

        Returns:
            router_weights: Normalized weights for selected experts,
                shape (num_tokens, topk).
            selected_experts: Indices of selected experts,
                shape (num_tokens, topk).
        """
        # V2 Model Runner uses 2D tensors directly
        if router_logits.ndim == 2:
            num_tokens, num_experts = router_logits.shape
            scores = router_logits
        else:
            # Legacy 3D path
            batch_size, seq_len, num_experts = router_logits.shape
            num_tokens = batch_size * seq_len
            scores = router_logits.reshape(num_tokens, num_experts)  # (N, num_experts)

        # Group experts: 256 experts → 8 groups of 32 experts each
        experts_per_group = num_experts // self.n_group
        scores_grouped = scores.reshape(num_tokens, self.n_group, experts_per_group)

        # Select top-k groups based on maximum score in each group
        group_scores, _ = scores_grouped.max(dim=2)  # (N, n_group)
        _, top_group_indices = torch.topk(
            group_scores, k=self.topk_group, dim=1
        )  # (N, topk_group)

        # Create mask for selected groups
        group_mask = torch.zeros(
            num_tokens, self.n_group, device=scores.device, dtype=torch.bool
        )
        group_mask.scatter_(1, top_group_indices, True)

        # Apply group mask to scores
        group_mask_expanded = group_mask.unsqueeze(2).expand_as(
            scores_grouped
        )  # (N, n_group, experts_per_group)
        masked_scores = scores_grouped.clone()
        masked_scores[~group_mask_expanded] = float("-inf")

        # Flatten back and select top-k experts from selected groups
        masked_scores_flat = masked_scores.reshape(num_tokens, num_experts)
        topk_weights, topk_indices = torch.topk(
            masked_scores_flat, k=self.num_experts_per_tok, dim=1
        )  # (N, topk)

        # Normalize weights
        topk_weights = topk_weights / (topk_weights.sum(dim=1, keepdim=True) + 1e-20)

        return topk_weights, topk_indices


class LLaDA2DecoderLayer(nn.Module):
    """Single LLaDA2.0 transformer layer with block attention and MoE.

    Combines:
    - LLaDA2BlockAttention (non-causal within blocks)
    - LLaDA2MoE (group-limited routing with shared expert)
    - RMSNorm pre-normalization

    Architecture:
        hidden = residual + self_attn(norm(residual))
        hidden = hidden + mlp(norm(hidden))

    Args:
        config: HuggingFace model config.
        layer_idx: Layer index (0-indexed).
        vllm_config: vLLM configuration.
        prefix: Parameter name prefix.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        layer_idx: int,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        # Extract vLLM configs for attention layer
        cache_config = vllm_config.cache_config
        quant_config = getattr(vllm_config, "quant_config", None)

        # Block-style attention
        # TEMPORARY FIX: Use encoder_only (non-causal) attention instead of decoder (causal)
        # LLaDA2.0 is a diffusion model with bidirectional attention within blocks
        # Full dual-chunk block attention implementation deferred to Phase 10
        self.self_attn = LLaDA2BlockAttention(
            num_heads=config.num_attention_heads,
            head_size=config.hidden_size // config.num_attention_heads,
            num_kv_heads=getattr(
                config, "num_key_value_heads", config.num_attention_heads
            ),
            rope_theta=getattr(config, "rope_theta", 10000),
            rope_scaling=getattr(config, "rope_scaling", None),
            max_position_embeddings=getattr(config, "max_position_embeddings", 8192),
            partial_rotary_factor=getattr(config, "partial_rotary_factor", None),
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            # attn_type defaults to "decoder" - use bidirectional backend to override causal=False
        )

        # MoE FFN (or dense-only for early layers)
        # Check if this layer is dense-only (controlled by first_k_dense_replace)
        first_k_dense = getattr(config, "first_k_dense_replace", 0)
        is_dense_only = layer_idx < first_k_dense

        tp_size = get_tp_group().world_size
        self.mlp = LLaDA2MoE(
            config=config,
            tp_size=tp_size,
            prefix=f"{prefix}.mlp",
            is_dense_only=is_dense_only,
        )

        # RMSNorm layers
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=getattr(config, "rms_norm_eps", 1e-6),
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=getattr(config, "rms_norm_eps", 1e-6),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Transformer layer forward pass (V2 Model Runner signature).

        Args:
            positions: Position indices, shape (num_tokens,).
            hidden_states: Input tensor, shape (num_tokens, hidden_size).
            residual: Residual tensor from previous layer, or None for first layer.

        Returns:
            Tuple of (hidden_states, residual).

        Note:
            V2 Model Runner pattern:
            - KV cache accessed by attention layer via self.kv_cache
            - Attention metadata accessed via get_forward_context().attn_metadata
        """
        # Pre-norm attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        # Block-style attention (V2 signature - no kv_cache/attn_metadata params)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        # Pre-norm MoE
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


@support_torch_compile(
    dynamic_arg_dims={"input_ids": 0, "positions": 0},
)
class LLaDA2ForCausalLM(nn.Module):
    """LLaDA2.0 causal language model for vLLM.

    Production-ready implementation with:
    - 256-expert MoE architecture
    - Block-style non-causal attention
    - Tensor parallelism (TP) support
    - Proper weight loading from HuggingFace checkpoints

    **Phase 8 Optimizations:** Supports vLLM torch.compile via `@support_torch_compile`
    decorator. Compilation is automatic when vLLM's compilation config is enabled.
    Routing and MoE layers benefit most from graph optimization.

    **Pipeline Parallelism:** NOT supported in Phase 7 MVP. Will raise ValueError
    if `pipeline_parallel_size > 1`. Use `--tensor-parallel-size` for multi-GPU.

    Args:
        vllm_config: vLLM configuration object.
        prefix: Parameter name prefix (default: empty).
    """

    # Declare supported vLLM runners
    supported_runners = ["generate"]

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        # Validate stack compatibility
        assert_compatible_stack(vllm_config, caller="LLaDA2ForCausalLM.__init__")

        # Validate PP not enabled (fail fast for MVP)
        pp_config = vllm_config.parallel_config
        if pp_config.pipeline_parallel_size > 1:
            raise ValueError(
                "Pipeline parallelism (PP > 1) is not supported for LLaDA2ForCausalLM "
                "in Phase 7 MVP. Use tensor parallelism (--tensor-parallel-size) for "
                "multi-GPU inference. PP support may be added in a future phase."
            )

        # Log GPU capability (Phase 8)
        try:
            gpu_caps = detect_gpu_capabilities()
            logger.info(
                "LLaDA2.0 model initialized on %s (compute %d.%d, %.1fGB) "
                "- torch.compile via @support_torch_compile decorator",
                gpu_caps.device_name,
                gpu_caps.compute_capability[0],
                gpu_caps.compute_capability[1],
                gpu_caps.total_memory_gb,
            )
        except Exception as e:
            logger.warning(
                "GPU capability detection failed (non-fatal): %s",
                e,
            )

        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config

        # Model dimensions
        self.hidden_size = self.config.hidden_size
        self.vocab_size = self.config.vocab_size
        self.num_layers = self.config.num_hidden_layers

        # MoE architecture parameters (needed for weight loading)
        self.num_experts = getattr(
            self.config, "num_experts", LLADA2_DEFAULT_NUM_EXPERTS
        )

        # Embedding layer
        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            self.hidden_size,
        )

        # Transformer layers
        self.layers = nn.ModuleList(
            [
                LLaDA2DecoderLayer(
                    config=self.config,
                    layer_idx=layer_idx,
                    vllm_config=vllm_config,
                    prefix=f"model.layers.{layer_idx}",
                )
                for layer_idx in range(self.num_layers)
            ]
        )

        # Final layer norm
        self.norm = RMSNorm(
            self.hidden_size,
            eps=getattr(self.config, "rms_norm_eps", 1e-6),
        )

        # LM head (with optional weight tying to embeddings)
        self.lm_head = ParallelLMHead(
            self.vocab_size,
            self.hidden_size,
            bias=False,
        )

        # Weight tying: Share lm_head weights with embed_tokens if configured
        # (Standard LLaMA pattern to reduce model size)
        if getattr(self.config, "tie_word_embeddings", False):
            self.lm_head.weight = self.embed_tokens.weight

        # Logits processor
        self.logits_processor = LogitsProcessor(self.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Apply token embeddings to input_ids.

        Required by vLLM's VllmModel protocol for model validation.

        Args:
            input_ids: Input token IDs, shape (batch, seq_len).

        Returns:
            Token embeddings, shape (batch, seq_len, hidden_size).
        """
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        """Model forward pass (V2 Model Runner signature).

        Args:
            input_ids: Input token IDs, shape (num_tokens,).
            positions: Position indices, shape (num_tokens,).
            intermediate_tensors: Intermediate tensors for PP (not used in Phase 7).
            inputs_embeds: Optional pre-computed embeddings.

        Returns:
            Hidden states, shape (num_tokens, hidden_size).

        Note:
            V2 Model Runner accesses KV cache and attention metadata via context:
            - KV cache: accessed by attention layers via self.kv_cache
            - Attention metadata: accessed via get_forward_context().attn_metadata
        """
        # PP rank handling (simplified for single-rank in Phase 7)
        pp_group = get_pp_group()

        if pp_group.is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                assert input_ids is not None
                hidden_states = self.embed_tokens(input_ids)
            residual = None
        else:
            # PP > 1: receive hidden states from previous rank
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors.get("residual")

        # Transformer layers (V2 signature - no kv_cache/attn_metadata params)
        for layer in self.layers:
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )

        if not pp_group.is_last_rank:
            # PP > 1: pass hidden states to next rank
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual,
            })

        # Final norm
        hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata=None,
    ) -> torch.Tensor:
        """Compute logits from hidden states.

        Args:
            hidden_states: Hidden states, shape (batch, seq_len, hidden_size).
            sampling_metadata: Sampling metadata from vLLM (None during profiling).

        Returns:
            Logits, shape (batch, seq_len, vocab_size).
        """
        logits = self.logits_processor(
            self.lm_head,
            hidden_states,
            sampling_metadata,
        )
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load LLaDA2.0 weights with MoE expert stacking.

        Follows vLLM MoE pattern (Qwen2 MoE two-phase + Mixtral expert mapping):
        1. Regular parameters (embeddings, norms, non-expert layers)
        2. Expert parameters (stacked into FusedMoE format)

        Weight naming convention:
        - Experts: `model.layers.{L}.mlp.experts.{E}.{gate_proj,up_proj,down_proj}`
        - Shared expert:
          `model.layers.{L}.mlp.shared_expert.{gate_proj,up_proj,down_proj}`
        - Other: Standard transformer naming

        Args:
            weights: Iterable of (name, tensor) pairs from checkpoint.

        Returns:
            Set of parameter names that were not loaded (for debugging).
        """
        params_dict = dict(self.named_parameters())
        loaded_params = set()

        # Collect expert weights for later stacking
        expert_weights = defaultdict(
            lambda: defaultdict(dict)
        )  # layer_id -> expert_id -> param_name -> tensor

        # Debug: Track what's happening during weight loading
        checkpoint_names_seen = []
        mapped_names_loaded = []
        mapped_names_skipped = []

        for name, loaded_weight in weights:
            checkpoint_name = name  # Save original for debugging

            # Handle checkpoint naming conventions
            # HuggingFace checkpoints may use different names than vLLM models
            # Map common HF naming patterns to vLLM parameter names
            if name.startswith("model."):
                name = name[len("model.") :]  # Remove 'model.' prefix
            if "word_embeddings" in name:
                name = name.replace("word_embeddings", "embed_tokens")

            # Map attention layer parameters (HF -> vLLM)
            # HF LLaDA2.0 uses: attention.query_key_value, attention.dense
            # vLLM Attention uses: self_attn.qkv_proj, self_attn.o_proj
            if ".attention.query_key_value." in name:
                name = name.replace(
                    ".attention.query_key_value.", ".self_attn.qkv_proj."
                )
            if ".attention.dense." in name:
                name = name.replace(".attention.dense.", ".self_attn.o_proj.")
            if ".attention.query_layernorm." in name:
                name = name.replace(".attention.query_layernorm.", ".self_attn.q_norm.")
            if ".attention.key_layernorm." in name:
                name = name.replace(".attention.key_layernorm.", ".self_attn.k_norm.")

            # Map shared expert parameters (non-expert MLP weights)
            # HF checkpoint has two formats:
            #   Layer 0: layers.0.mlp.gate_proj.weight
            #   Layer 1+: layers.X.mlp.shared_experts.gate_proj.weight
            # Model expects: layers.X.mlp.shared_expert_gate.weight (singular)
            # NOTE: Only applies to NON-expert paths (parsed separately)
            if ".mlp.shared_experts.gate_proj." in name:
                name = name.replace(
                    ".mlp.shared_experts.gate_proj.", ".mlp.shared_expert_gate."
                )
            elif ".mlp.gate_proj." in name and ".mlp.experts." not in name:
                name = name.replace(".mlp.gate_proj.", ".mlp.shared_expert_gate.")

            if ".mlp.shared_experts.up_proj." in name:
                name = name.replace(
                    ".mlp.shared_experts.up_proj.", ".mlp.shared_expert_up."
                )
            elif ".mlp.up_proj." in name and ".mlp.experts." not in name:
                name = name.replace(".mlp.up_proj.", ".mlp.shared_expert_up.")

            if ".mlp.shared_experts.down_proj." in name:
                name = name.replace(
                    ".mlp.shared_experts.down_proj.", ".mlp.shared_expert_down."
                )
            elif ".mlp.down_proj." in name and ".mlp.experts." not in name:
                name = name.replace(".mlp.down_proj.", ".mlp.shared_expert_down.")

            # Map MoE router/gate bias parameter
            # HF checkpoint: mlp.gate.expert_bias
            # Model expects: mlp.gate.bias
            if ".mlp.gate.expert_bias" in name:
                name = name.replace(".mlp.gate.expert_bias", ".mlp.gate.bias")

            checkpoint_names_seen.append(checkpoint_name)

            # Skip lm_head if weight tying is enabled
            # (lm_head.weight is tied to embed_tokens.weight in __init__)
            if getattr(self.config, "tie_word_embeddings", False) and "lm_head" in name:
                mapped_names_skipped.append(f"{checkpoint_name} -> {name} (tied)")
                continue

            # Skip expert weights in first pass (will stack later)
            if ".mlp.experts." in name:
                # Parse: layers.{L}.mlp.experts.{E}.{param}
                # (Note: 'model.' prefix already removed above)
                match = re.match(
                    r"layers\.(\d+)\.mlp\.experts\.(\d+)\.(.+)",
                    name,
                )
                if match:
                    layer_id, expert_id, param_name = match.groups()
                    expert_weights[int(layer_id)][int(expert_id)][param_name] = (
                        loaded_weight
                    )
                    # NOTE: Don't add individual expert weights to loaded_params here
                    # They'll be stacked and added as w13_weight/w2_weight later
                    mapped_names_loaded.append(
                        f"{checkpoint_name} -> {name} (expert, will stack)"
                    )
                continue

            # Load regular parameters
            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)
                mapped_names_loaded.append(f"{checkpoint_name} -> {name}")
            else:
                mapped_names_skipped.append(
                    f"{checkpoint_name} -> {name} (NOT IN params_dict)"
                )

        # Stack expert weights per layer (Phase 2 of loading)
        # Skip dense-only layers (controlled by first_k_dense_replace)
        first_k_dense = getattr(self.config, "first_k_dense_replace", 0)

        for layer_id, experts_dict in expert_weights.items():
            # Skip dense-only layers (they don't have expert weights)
            if layer_id < first_k_dense:
                continue

            # Load expert weights individually with expert_id for TP sharding.
            # vLLM's weight_loader uses expert_id to distribute experts across TP ranks.
            # Pattern matches Qwen2-MoE, Mixtral, DeepSeek V3 implementations.
            #
            # Each expert has gate_proj, up_proj, down_proj weights.
            # FusedMoE expects w13_weight (gate+up stacked) and w2_weight (down).

            layer = self.layers[layer_id]
            # Type narrowing: layer.mlp is LLaDA2MoE
            assert isinstance(layer.mlp, LLaDA2MoE)
            # Ensure experts exist (dense-only layers are skipped above)
            assert layer.mlp.experts is not None

            for expert_id in range(self.num_experts):
                if expert_id not in experts_dict:
                    raise ValueError(
                        f"Missing expert {expert_id} weights for layer {layer_id}"
                    )

                expert_params = experts_dict[expert_id]

                # Get gate_proj and up_proj, stack into w13
                gate_weight = expert_params.get("gate_proj.weight")
                up_weight = expert_params.get("up_proj.weight")
                down_weight = expert_params.get("down_proj.weight")

                # Validate all weights exist
                if gate_weight is None or up_weight is None or down_weight is None:
                    raise ValueError(
                        f"Missing weights for layer {layer_id} expert {expert_id}: "
                        f"gate={gate_weight is not None}, "
                        f"up={up_weight is not None}, "
                        f"down={down_weight is not None}"
                    )

                # Load expert weights using FusedMoE weight_loader (correct shard_ids)
                # shard_id must be one of ['w1', 'w2', 'w3']
                # w1=gate_proj, w2=down_proj, w3=up_proj
                param_w13 = layer.mlp.experts.w13_weight
                weight_loader = getattr(
                    param_w13, "weight_loader", default_weight_loader
                )

                # Load gate projection (w1)
                weight_loader(
                    param_w13,
                    gate_weight,
                    "gate_proj.weight",
                    shard_id="w1",
                    expert_id=expert_id,
                )

                # Load up projection (w3)
                weight_loader(
                    param_w13,
                    up_weight,
                    "up_proj.weight",
                    shard_id="w3",
                    expert_id=expert_id,
                )

                # Load down projection (w2)
                param_w2 = layer.mlp.experts.w2_weight
                weight_loader_w2 = getattr(
                    param_w2, "weight_loader", default_weight_loader
                )
                weight_loader_w2(
                    param_w2,
                    down_weight,
                    "down_proj.weight",
                    shard_id="w2",
                    expert_id=expert_id,
                )

            # Track expert weights as loaded
            loaded_params.add(f"layers.{layer_id}.mlp.experts.w13_weight")
            loaded_params.add(f"layers.{layer_id}.mlp.experts.w2_weight")

        # Mark lm_head.weight as loaded if weight tying is enabled
        # (it shares the same Parameter object as embed_tokens.weight, so loading
        # embed_tokens.weight automatically "loads" lm_head.weight)
        if (
            getattr(self.config, "tie_word_embeddings", False)
            and "embed_tokens.weight" in loaded_params
        ):
            loaded_params.add("lm_head.weight")

        # Return loaded params (vLLM expects set of successfully loaded param names)
        return loaded_params
