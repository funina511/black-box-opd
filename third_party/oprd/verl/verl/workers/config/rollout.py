# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Optional

from omegaconf import MISSING

from verl.base_config import BaseConfig
from verl.utils.profiler import ProfilerConfig

__all__ = [
    "SamplingConfig",
    "MultiTurnConfig",
    "CustomAsyncServerConfig",
    "AgentLoopConfig",
    "TraceConfig",
    "ServerConfig",
    "RolloutConfig",
]


@dataclass
class SamplingConfig(BaseConfig):
    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0
    do_sample: bool = True
    max_tokens: Optional[int] = None
    n: int = 1


@dataclass
class MultiTurnConfig(BaseConfig):
    _mutable_fields = {"max_assistant_turns", "max_user_turns"}

    enable: bool = False
    max_assistant_turns: Optional[int] = None
    tool_config_path: Optional[str] = None
    max_user_turns: Optional[int] = None
    max_parallel_calls: int = 1
    max_tool_response_length: int = 256
    tool_response_truncate_side: str = "middle"
    interaction_config_path: Optional[str] = None
    use_inference_chat_template: bool = False
    tokenization_sanity_check_mode: str = "strict"
    format: str = "hermes"
    num_repeat_rollouts: Optional[int] = None


@dataclass
class CustomAsyncServerConfig(BaseConfig):
    path: Optional[str] = None
    name: Optional[str] = None


@dataclass
class AgentLoopConfig(BaseConfig):
    num_workers: int = 8
    default_agent_loop: str = "single_turn_agent"
    agent_loop_config_path: Optional[str] = None
    custom_async_server: CustomAsyncServerConfig = field(default_factory=CustomAsyncServerConfig)


@dataclass
class TraceConfig(BaseConfig):
    backend: Optional[str] = None
    token2text: bool = False


@dataclass
class ServerConfig(BaseConfig):
    """
    Configuration for SGLang server when running in server mode
    """

    timeout: float = 60.0
    max_attempts: int = 3
    retry_delay: float = 2.0
    max_connections: int = 1000
    max_start_wait_time: float = 300.0


@dataclass
class RolloutConfig(BaseConfig):
    _mutable_fields = {"max_model_len", "load_format"}

    name: Optional[str] = MISSING
    mode: str = "sync"
    skip_tokenizer_init: bool = True

    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    do_sample: bool = True
    n: int = 1

    # Early termination threshold for multi-turn rollout in sglang.
    # Abort remaining requests when (1 - over_sample_rate) * total_requests are completed.
    over_sample_rate: float = 0.0

    prompt_length: int = 512
    response_length: int = 512

    dtype: str = "bfloat16"
    gpu_memory_utilization: float = 0.5
    ignore_eos: bool = False
    enforce_eager: bool = True
    cudagraph_capture_sizes: Optional[list] = None
    free_cache_engine: bool = True
    data_parallel_size: int = 1
    expert_parallel_size: int = 1
    tensor_model_parallel_size: int = 2
    pipeline_model_parallel_size: int = 1
    max_num_batched_tokens: int = 8192

    # TODO: enable train_kwargs
    # train_sampling_config: SamplingConfig = field(default_factory=SamplingConfig)

    val_kwargs: SamplingConfig = field(default_factory=SamplingConfig)

    max_model_len: Optional[int] = None
    max_num_seqs: int = 1024

    # note that the logprob computation should belong to the actor
    log_prob_micro_batch_size: Optional[int] = None
    log_prob_micro_batch_size_per_gpu: Optional[int] = None
    log_prob_use_dynamic_bsz: bool = False
    log_prob_max_token_len_per_gpu: int = 16384
    log_prob_top_k: int = 256
    top_k_strategy: str = "only_stu"  # "only_stu", "only_tch", "intersection", or "union"
    reward_weight_mode: str = "student_p"  # "student_p", "teacher_p", or "none"
    teacher_temperature: float = 1.0  # Temperature for teacher logits (default 1.0, no scaling)
    # If False with OPRD: teacher still provides hidden for RKD, but skip reverse-KL
    # rm_scores so student outcome/format RL drives the policy head.
    use_token_kl_reward: bool = True
    # Surface channel (②): return teacher per-token log-prob on student response
    # tokens so the trainer can build a text-manifold teacher-likelihood reward.
    use_surface_reward: bool = False
    # Cross-vocab surface (①): teacher tokenizer != student. RM worker decodes the
    # student text, re-tokenizes with the teacher tokenizer, and returns a length-
    # normalized scalar seq_ll (teacher token-mean logp) instead of per-token logp.
    surface_reward_cross_vocab: bool = False
    # Cross-vocab surface tuning (previously read from a non-existent reward_model
    # field, so silently pinned to defaults). topk: partition-function top-k width;
    # max_length: teacher-side truncation cap (None -> student attention length);
    # log_tail_gap: measure the S2 top-k denominator bias (one extra full-V logsumexp).
    surface_reward_topk: int = 2048
    surface_reward_max_length: Optional[int] = None
    surface_reward_log_tail_gap: bool = False
    # Cross-vocab surface: use the EXACT full-vocab logsumexp partition function
    # (chunked over span rows) instead of the top-k approximation. Removes the S2
    # denominator bias entirely at the cost of one full-V reduction; affordable in
    # surface-only runs (hidden-repr extraction is off). Ignored same-vocab.
    surface_reward_exact_denom: bool = False
    # Surface entropy term (route A): subtract lambda * detached student per-token
    # logp from the teacher-LL reward. 0.0 = pure surface (E[logp_T]); 1.0 = full
    # sequence-level OPD (E[logp_T] + H(pi), telescoped). Adds back the student
    # entropy term that pure surface drops, reversing low-entropy mode collapse.
    # Same-vocab only (needs aligned per-token student logp); ignored cross-vocab.
    surface_student_entropy_coef: float = 0.0
    # Surface entropy credit granularity (how the -lam*logp_S term enters):
    #   "seq"        : fold into the last-token sequence scalar, GRPO-baselined
    #                  (route A). Same/cross-vocab. lam=1 is NOT token-OPD (loses
    #                  per-token credit -> empirically collapses to low entropy).
    #   "token_raw"  : SAME-VOCAB ONLY. r_t = logp_T(y_t) - lam*logp_S(y_t) spread
    #                  per-token (no norm, no baseline) + token_reward_direct, so the
    #                  PG sum telescopes to logp_T(y) - lam*logp_S(y). lam=1 == OPD
    #                  EXACTLY (both terms same per-token nats scale -> auto-balanced).
    #                  This is the clean same-vocab anchor.
    #   "token_dual" : DUAL-CHANNEL (works cross-vocab). Teacher term rides as the
    #                  GRPO seq scalar (std-normalized -> ~unit RMS); the entropy term
    #                  -lam*logp_S(y_t) is injected PER-TOKEN post-advantage, itself
    #                  per-seq de-meaned and BATCH-std-normalized to ~unit RMS. lam is
    #                  then a dimensionless teacher/entropy pressure ratio (NOT OPD;
    #                  cross-vocab has no per-token teacher term, so exact OPD is out).
    surface_entropy_mode: str = "seq"

    disable_log_stats: bool = True

    multi_stage_wake_up: bool = False
    engine_kwargs: dict = field(default_factory=dict)

    calculate_log_probs: bool = False

    agent: AgentLoopConfig = field(default_factory=AgentLoopConfig)

    trace: TraceConfig = field(default_factory=TraceConfig)

    multi_turn: MultiTurnConfig = field(default_factory=MultiTurnConfig)

    # Server configuration for sglang server mode
    server: ServerConfig = field(default_factory=ServerConfig)

    update_weights_bucket_megabytes: int = 512

    skip_rollout: bool = False

    skip_dump_dir: str = "/tmp/rollout_dump"

    profiler: Optional[ProfilerConfig] = None

    enable_chunked_prefill: bool = True

    enable_prefix_caching: bool = True

    load_format: str = "dummy"

    layered_summon: bool = False

    layer_name_map: dict = field(default_factory=dict)

    sglang_engine_mode: str = "local"

    limit_images: Optional[int] = None

    skip_tokenizer_init: bool = False

    def __post_init__(self):
        """Validate the rollout config"""
        if self.expert_parallel_size > 1:
            assert self.expert_parallel_size == (self.tensor_model_parallel_size * self.data_parallel_size), (
                "expert_parallel_size must be equal to tensor_model_parallel_size * data_parallel_size"
            )

        if self.pipeline_model_parallel_size > 1:
            if self.name == "vllm" or self.name == "sglang":
                raise NotImplementedError(
                    f"Current rollout {self.name=} not implemented pipeline_model_parallel_size > 1 yet."
                )
