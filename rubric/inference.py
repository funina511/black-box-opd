"""Lazy local generation; never truncate a prompt to fit the judge."""
from __future__ import annotations
import time
import atexit


class Generator:
    def __init__(self, path, *, tp, context, utilization, seed, thinking=None):
        from transformers import AutoTokenizer
        from vllm import LLM
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.thinking = thinking
        self.context = context
        self.engine = LLM(model=str(path), tokenizer=str(path), tensor_parallel_size=tp,
                          max_model_len=context, gpu_memory_utilization=utilization,
                          dtype="bfloat16", enforce_eager=True, trust_remote_code=False,
                          seed=seed, disable_log_stats=True, max_num_seqs=16)
        atexit.register(self.close)

    def close(self):
        if self.engine is None:
            return
        client = getattr(self.engine.llm_engine, "engine_core", None)
        if client is not None:
            client.shutdown()
        else:
            self.engine.llm_engine.model_executor.shutdown()
        self.engine = None
        atexit.unregister(self.close)

    def render(self, messages):
        options = {} if self.thinking is None else {"enable_thinking": self.thinking}
        # Deliberately no fallback that discards the thinking switch.
        return self.tokenizer.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=True, **options)

    def generate(self, messages, *, n, temperature, top_p, max_tokens, seed,
                 max_prompt_tokens=None):
        from vllm import SamplingParams
        rendered = [self.render(m) for m in messages]
        sizes = [len(self.tokenizer.encode(p, add_special_tokens=False)) for p in rendered]
        if any(s + max_tokens > self.context for s in sizes):
            raise ValueError("full prompt + output reservation exceeds context; no truncation")
        if max_prompt_tokens and any(s > max_prompt_tokens for s in sizes):
            raise ValueError("prompt exceeds source max_prompt_length")
        started = time.monotonic()
        seeds = seed if isinstance(seed, list) else [seed] * len(rendered)
        outputs = self.engine.generate(
            rendered, [SamplingParams(n=n, temperature=temperature, top_p=top_p,
                                      max_tokens=max_tokens, seed=s) for s in seeds],
            use_tqdm=False)
        return [{"prompt_tokens": len(item.prompt_token_ids),
                 "seconds_batch": time.monotonic() - started,
                 "outputs": [{"text": out.text, "token_ids": list(out.token_ids),
                              "finish_reason": out.finish_reason,
                              "stop_reason": out.stop_reason,
                              "truncated": out.finish_reason == "length"} for out in item.outputs]}
                for item in outputs]
