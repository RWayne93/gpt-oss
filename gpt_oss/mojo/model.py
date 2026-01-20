import json
import math
import os
import time

import torch
import torch.distributed as dist

from gpt_oss.mojo.mxfp4 import try_mxfp4_unpack
from gpt_oss.torch.model import ModelConfig, RMSNorm, build_model_config, swiglu
from gpt_oss.torch.weights import Checkpoint, FP4_VALUES

_LUT_CACHE: dict[tuple[str, int | None, torch.dtype], torch.Tensor] = {}
_UNPACK_BUFFER_CACHE: dict[tuple[str, int | None, torch.dtype], torch.Tensor] = {}
_PROFILE = os.environ.get("GPT_OSS_MOJO_PROFILE", "0").strip().lower() in {"1", "true", "yes", "on"}
_PROFILE_STATS: dict[str, list[float]] = {}


def _get_int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


_MLP_CHUNK_TOKENS = _get_int_env("GPT_OSS_MOJO_MLP_CHUNK_TOKENS", 0)


def _sync(device: torch.device | None) -> None:
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)


def _profile_add(name: str, duration: float) -> None:
    if not _PROFILE:
        return
    stats = _PROFILE_STATS.setdefault(name, [0.0, 0.0])
    stats[0] += duration
    stats[1] += 1.0


def _get_unpack_buffer(device: torch.device, numel: int, dtype: torch.dtype) -> torch.Tensor:
    key = (device.type, device.index, dtype)
    buf = _UNPACK_BUFFER_CACHE.get(key)
    if buf is None or buf.numel() < numel or buf.device != device:
        buf = torch.empty(numel, device=device, dtype=dtype)
        _UNPACK_BUFFER_CACHE[key] = buf
    return buf


def _unpack_mxfp4(
    blocks: torch.Tensor,
    scales: torch.Tensor,
    out: torch.Tensor,
) -> None:
    if _PROFILE:
        _sync(out.device)
        start = time.perf_counter()
    if try_mxfp4_unpack(out, blocks, scales):
        if _PROFILE:
            _sync(out.device)
            _profile_add("mxfp4_unpack_mojo", time.perf_counter() - start)
        return
    if _PROFILE:
        _sync(out.device)
        _profile_add("mxfp4_unpack_mojo", time.perf_counter() - start)
        start = time.perf_counter()
    key = (out.device.type, out.device.index, out.dtype)
    lut = _LUT_CACHE.get(key)
    if lut is None:
        lut = torch.tensor(FP4_VALUES, dtype=out.dtype, device=out.device)
        _LUT_CACHE[key] = lut
    idx_lo = (blocks & 0x0F).to(torch.long)
    idx_hi = (blocks >> 4).to(torch.long)
    out[:, 0::2] = lut[idx_lo]
    out[:, 1::2] = lut[idx_hi]
    torch.ldexp(out, scales[:, None], out=out)
    if _PROFILE:
        _sync(out.device)
        _profile_add("mxfp4_unpack_torch", time.perf_counter() - start)


class Cache:
    def __init__(self, batch_size, n_ctx, n_kv_heads, d_head=64, device: torch.device | None = None):
        self.k = torch.zeros((batch_size, n_ctx, n_kv_heads, d_head), dtype=torch.bfloat16, device=device)
        self.v = torch.zeros((batch_size, n_ctx, n_kv_heads, d_head), dtype=torch.bfloat16, device=device)
        self.offset = torch.zeros((1,), dtype=torch.long, device=device)

    def reset(self):
        self.k.zero_()
        self.v.zero_()
        self.offset.zero_()

    def extend(self, k, v):
        batch_size, n_ctx, *_rest = k.shape
        assert batch_size == self.k.shape[0]
        indices = torch.arange(0, n_ctx, device=k.device, dtype=torch.long) + self.offset
        self.k.index_copy_(1, indices, k)
        self.v.index_copy_(1, indices, v)
        self.offset.add_(n_ctx)
        return self.k, self.v


class RotaryEmbedding(torch.nn.Module):
    def __init__(
        self,
        head_dim: int,
        base: int,
        dtype: torch.dtype,
        initial_context_length: int = 4096,
        max_context_length: int = 131072,
        scaling_factor: float = 1.0,
        ntk_alpha: float = 1.0,
        ntk_beta: float = 32.0,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.dtype = dtype
        self.initial_context_length = initial_context_length
        self.max_context_length = max_context_length
        self.scaling_factor = scaling_factor
        self.ntk_alpha = ntk_alpha
        self.ntk_beta = ntk_beta
        self.device = device
        self.cos, self.sin = self._compute_cos_sin(0, self.max_context_length)

    def _compute_concentration_and_inv_freq(self) -> torch.Tensor:
        freq = self.base ** (
            torch.arange(0, self.head_dim, 2, dtype=torch.float, device=self.device)
            / self.head_dim
        )
        if self.scaling_factor > 1.0:
            concentration = 0.1 * math.log(self.scaling_factor) + 1.0

            d_half = self.head_dim / 2
            low = (
                d_half
                * math.log(self.initial_context_length / (self.ntk_beta * 2 * math.pi))
                / math.log(self.base)
            )
            high = (
                d_half
                * math.log(self.initial_context_length / (self.ntk_alpha * 2 * math.pi))
                / math.log(self.base)
            )
            assert 0 < low < high < d_half - 1

            interpolation = 1.0 / (self.scaling_factor * freq)
            extrapolation = 1.0 / freq

            ramp = (
                torch.arange(d_half, dtype=torch.float32, device=freq.device) - low
            ) / (high - low)
            mask = 1 - ramp.clamp(0, 1)

            inv_freq = interpolation * (1 - mask) + extrapolation * mask
        else:
            concentration = 1.0
            inv_freq = 1.0 / freq

        return concentration, inv_freq

    def _compute_cos_sin(self, start: int, num_tokens: int):
        concentration, inv_freq = self._compute_concentration_and_inv_freq()
        t = torch.arange(start, start + num_tokens, dtype=torch.float32, device=self.device)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        cos = freqs.cos() * concentration
        sin = freqs.sin() * concentration
        return cos, sin

    def _rotate(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        cos = cos[None, :, None, :].to(x.dtype)
        sin = sin[None, :, None, :].to(x.dtype)
        x1, x2 = torch.chunk(x, 2, dim=-1)
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        return torch.cat((o1, o2), dim=-1)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        offset: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_tokens, num_heads, head_dim = query.shape
        batch_size, num_tokens, num_key_value_heads, head_dim = key.shape

        idx = torch.arange(num_tokens, device=query.device, dtype=torch.long) + offset
        idx = idx % self.max_context_length
        cos = self.cos.index_select(0, idx)
        sin = self.sin.index_select(0, idx)

        query = self._rotate(query, cos, sin)
        key = self._rotate(key, cos, sin)
        return query, key


def attention_ref(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sinks: torch.Tensor,
    sm_scale: float = 0.125,
    sliding_window: int | None = None,
    start_q: torch.LongTensor = 0,
):
    batch_size, num_queries, num_key_value_heads, num_key_value_groups, head_dim = query.shape
    batch_size, num_keys, num_key_value_heads, head_dim = key.shape

    sinks = sinks.view(1, num_key_value_heads, num_key_value_groups, 1, 1).float()
    key = key.unsqueeze(3)
    value = value.unsqueeze(3)

    pos_keys = torch.arange(num_keys, device=query.device)
    pos_queries = torch.arange(num_queries, device=query.device) + start_q
    mask = pos_keys[None, :] > pos_queries[:, None]
    mask = mask.float().masked_fill(mask, float("-inf"))

    if sliding_window:
        too_old = pos_keys[None, :] < (pos_queries[:, None] - sliding_window + 1)
        mask.masked_fill_(too_old, float("-inf"))

    logits = torch.einsum("bqhmd,bkhmd->bhmqk", query.float(), key.float()) * sm_scale
    logits = logits + mask[None, None, None, :, :]

    logits_max = torch.max(logits, dim=-1, keepdim=True).values
    logits_or_sinks_max = torch.maximum(sinks, logits_max)
    sinks = torch.exp(sinks - logits_or_sinks_max)
    unnormalized_scores = torch.exp(logits - logits_or_sinks_max)
    normalizer = unnormalized_scores.sum(dim=-1, keepdim=True) + sinks
    scores = unnormalized_scores / normalizer

    output = torch.einsum("bhmqk,bkhmd->bqhmd", scores, value.float())

    output = output.reshape(batch_size, num_queries, num_key_value_heads * num_key_value_groups * head_dim).bfloat16()
    return output


class AttentionBlockMojo(torch.nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int = 0,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.sliding_window = config.sliding_window if layer_idx % 2 == 0 else 0
        self.layer_idx = layer_idx
        self.sinks = torch.nn.Parameter(
            torch.empty(config.num_attention_heads, device=device, dtype=torch.bfloat16)
        )
        self.norm = RMSNorm(config.hidden_size, device=device)
        qkv_dim = config.head_dim * (
            config.num_attention_heads + 2 * config.num_key_value_heads
        )
        self.qkv = torch.nn.Linear(
            config.hidden_size, qkv_dim, device=device, dtype=torch.bfloat16
        )
        self.out = torch.nn.Linear(
            config.head_dim * config.num_attention_heads,
            config.hidden_size,
            device=device,
            dtype=torch.bfloat16,
        )
        self.sm_scale = 1 / math.sqrt(config.head_dim)
        self.rope = RotaryEmbedding(
            config.head_dim,
            config.rope_theta,
            torch.float32,
            initial_context_length=config.initial_context_length,
            scaling_factor=config.rope_scaling_factor,
            ntk_alpha=config.rope_ntk_alpha,
            ntk_beta=config.rope_ntk_beta,
            device=device,
        )

    def forward(self, x: torch.Tensor, cache: Cache | None = None) -> torch.Tensor:
        batch_size, n_ctx, dim = x.shape
        if _PROFILE:
            _sync(x.device)
            start = time.perf_counter()
        t = self.norm(x)
        qkv = self.qkv(t)
        qkv_parts = (
            self.num_attention_heads * self.head_dim,
            self.num_key_value_heads * self.head_dim,
            self.num_key_value_heads * self.head_dim,
        )
        q, k, v = torch.split(qkv, qkv_parts, dim=-1)
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

        if _PROFILE:
            _sync(x.device)
            _profile_add("attn_qkv", time.perf_counter() - start)
        q = q.view(batch_size, n_ctx, self.num_attention_heads, self.head_dim)
        k = k.view(batch_size, n_ctx, self.num_key_value_heads, self.head_dim)
        v = v.view(batch_size, n_ctx, self.num_key_value_heads, self.head_dim)

        if cache is not None:
            offset = cache.offset.clone()
            q, k = self.rope(q, k, offset=offset)
            k, v = cache.extend(k, v)
        else:
            offset = torch.zeros((1,), dtype=torch.long, device=x.device)
            q, k = self.rope(q, k, offset=offset)

        q = q.view(
            batch_size,
            n_ctx,
            self.num_attention_heads // self.num_key_value_heads,
            self.num_key_value_heads,
            self.head_dim,
        )
        if _PROFILE:
            _sync(x.device)
            start = time.perf_counter()
        t = attention_ref(
            q,
            k,
            v,
            self.sinks,
            self.sm_scale,
            self.sliding_window,
            offset,
        )
        if _PROFILE:
            _sync(x.device)
            _profile_add("attn_scores", time.perf_counter() - start)
            start = time.perf_counter()
        t = self.out(t)
        t = x + t
        if _PROFILE:
            _sync(x.device)
            _profile_add("attn_out", time.perf_counter() - start)
        return t

class MLPBlockMojo(torch.nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.num_experts = config.num_experts
        self.experts_per_token = config.experts_per_token
        self.swiglu_limit = config.swiglu_limit
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.norm = RMSNorm(config.hidden_size, device=device)
        self.gate = torch.nn.Linear(
            config.hidden_size, config.num_experts, device=device, dtype=torch.bfloat16
        )
        if self.world_size != 1:
            raise NotImplementedError("Mojo MXFP4 path only supports world_size=1.")

        self.register_buffer("mlp1_blocks", torch.empty(0, device=device, dtype=torch.uint8))
        self.register_buffer("mlp1_scales", torch.empty(0, device=device, dtype=torch.int32))
        self.register_buffer("mlp2_blocks", torch.empty(0, device=device, dtype=torch.uint8))
        self.register_buffer("mlp2_scales", torch.empty(0, device=device, dtype=torch.int32))

        self.mlp1_bias = torch.nn.Parameter(
            torch.empty(
                (config.num_experts, config.intermediate_size * 2),
                device=device,
                dtype=torch.bfloat16,
            )
        )
        self.mlp2_bias = torch.nn.Parameter(
            torch.empty(
                (config.num_experts, config.hidden_size),
                device=device,
                dtype=torch.bfloat16,
            )
        )

    def _unpack_selected(
        self,
        blocks: torch.Tensor,
        scales: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> torch.Tensor:
        if _PROFILE:
            _sync(blocks.device)
            start = time.perf_counter()
        expert_shape = expert_indices.shape
        flat_indices = expert_indices.reshape(-1, expert_shape[-1])
        sel_blocks = blocks[flat_indices]
        sel_scales = scales[flat_indices]
        batch, experts, rows, block_groups, block_bytes = sel_blocks.shape
        flat_blocks = sel_blocks.reshape(batch * experts * rows * block_groups, block_bytes)
        flat_scales = sel_scales.reshape(batch * experts * rows * block_groups)
        out_rows = batch * experts * rows * block_groups
        out_cols = block_bytes * 2
        out_numel = out_rows * out_cols
        out = _get_unpack_buffer(blocks.device, out_numel, torch.bfloat16)[:out_numel].view(
            out_rows, out_cols
        )
        _unpack_mxfp4(flat_blocks, flat_scales, out)
        if _PROFILE:
            _sync(blocks.device)
            _profile_add("mxfp4_unpack_total", time.perf_counter() - start)
        out = out.reshape(batch, experts, rows, block_groups, block_bytes * 2)
        out = out.reshape(batch, experts, rows, block_groups * block_bytes * 2)
        return out.reshape(*expert_shape[:-1], expert_shape[-1], rows, block_groups * block_bytes * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, n_ctx, dim = x.shape
        if _PROFILE:
            _sync(x.device)
            start = time.perf_counter()
        t = self.norm(x).view(batch_size * n_ctx, dim)
        g = self.gate(t)
        experts = torch.topk(g, k=self.experts_per_token, dim=-1, sorted=True)
        expert_weights = torch.nn.functional.softmax(experts.values, dim=1)
        expert_indices = experts.indices
        if _PROFILE:
            _sync(x.device)
            _profile_add("mlp_gate", time.perf_counter() - start)
        chunk_tokens = _MLP_CHUNK_TOKENS
        if chunk_tokens > 0 and t.shape[0] > chunk_tokens:
            out = torch.empty_like(t)
            for start_idx in range(0, t.shape[0], chunk_tokens):
                end_idx = min(start_idx + chunk_tokens, t.shape[0])
                t_chunk = t[start_idx:end_idx]
                idx_chunk = expert_indices[start_idx:end_idx]
                weights_chunk = expert_weights[start_idx:end_idx]

                if _PROFILE:
                    _sync(x.device)
                    start = time.perf_counter()
                mlp1_weight = self._unpack_selected(self.mlp1_blocks, self.mlp1_scales, idx_chunk)
                mlp1_bias = self.mlp1_bias[idx_chunk, ...]
                t_chunk = torch.einsum("beck,bk->bec", mlp1_weight, t_chunk) + mlp1_bias
                t_chunk = swiglu(t_chunk, limit=self.swiglu_limit)
                if _PROFILE:
                    _sync(x.device)
                    _profile_add("mlp_mlp1", time.perf_counter() - start)

                if _PROFILE:
                    _sync(x.device)
                    start = time.perf_counter()
                mlp2_weight = self._unpack_selected(self.mlp2_blocks, self.mlp2_scales, idx_chunk)
                mlp2_bias = self.mlp2_bias[idx_chunk, ...]
                t_chunk = torch.einsum("beck,bek->bec", mlp2_weight, t_chunk)
                t_chunk += mlp2_bias
                if _PROFILE:
                    _sync(x.device)
                    _profile_add("mlp_mlp2", time.perf_counter() - start)

                if _PROFILE:
                    _sync(x.device)
                    start = time.perf_counter()
                t_chunk = torch.einsum("bec,be->bc", t_chunk, weights_chunk)
                if _PROFILE:
                    _sync(x.device)
                    _profile_add("mlp_combine", time.perf_counter() - start)
                out[start_idx:end_idx] = t_chunk
            t = out
        else:
            if _PROFILE:
                _sync(x.device)
                start = time.perf_counter()
            mlp1_weight = self._unpack_selected(self.mlp1_blocks, self.mlp1_scales, expert_indices)
            mlp1_bias = self.mlp1_bias[expert_indices, ...]
            t = torch.einsum("beck,bk->bec", mlp1_weight, t) + mlp1_bias
            t = swiglu(t, limit=self.swiglu_limit)
            if _PROFILE:
                _sync(x.device)
                _profile_add("mlp_mlp1", time.perf_counter() - start)

            if _PROFILE:
                _sync(x.device)
                start = time.perf_counter()
            mlp2_weight = self._unpack_selected(self.mlp2_blocks, self.mlp2_scales, expert_indices)
            mlp2_bias = self.mlp2_bias[expert_indices, ...]
            t = torch.einsum("beck,bek->bec", mlp2_weight, t)
            t += mlp2_bias
            if _PROFILE:
                _sync(x.device)
                _profile_add("mlp_mlp2", time.perf_counter() - start)

            if _PROFILE:
                _sync(x.device)
                start = time.perf_counter()
            t = torch.einsum("bec,be->bc", t, expert_weights)
            if _PROFILE:
                _sync(x.device)
                _profile_add("mlp_combine", time.perf_counter() - start)

        t = t.view(batch_size, n_ctx, dim)
        return x + t


class TransformerBlockMojo(torch.nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int = 0,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.attn = AttentionBlockMojo(config, layer_idx, device)
        self.mlp = MLPBlockMojo(config, device)

    def forward(self, x: torch.Tensor, cache: Cache | None = None) -> torch.Tensor:
        x = self.attn(x, cache=cache)
        x = self.mlp(x)
        return x


class TransformerMojo(torch.nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.config = config
        self.embedding = torch.nn.Embedding(
            config.vocab_size, config.hidden_size, device=device, dtype=torch.bfloat16
        )
        self.block = torch.nn.ModuleList(
            [
                TransformerBlockMojo(config, layer_idx, device)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, device=device)
        self.unembedding = torch.nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            device=device,
            dtype=torch.bfloat16,
        )

    def forward(self, x: torch.Tensor, caches: list[Cache] | None = None) -> torch.Tensor:
        caches = caches or [None] * len(self.block)
        x = self.embedding(x)
        for block, cache in zip(self.block, caches):
            x = block(x, cache=cache)
        x = self.norm(x)
        x = self.unembedding(x)
        return x

    @staticmethod
    def from_checkpoint(
        path: str, device: str | torch.device = "cuda"
    ) -> "TransformerMojo":
        if not isinstance(device, torch.device):
            device = torch.device(device)

        config_path = os.path.join(path, "config.json")
        with open(config_path, "r") as f:
            json_config = json.load(f)
            config = build_model_config(json_config)

        model = TransformerMojo(
            config=config,
            device=device,
        )
        model.eval()

        checkpoint = Checkpoint(path, device)

        for name, param in model.named_parameters():
            if name.endswith("mlp1_weight") or name.endswith("mlp2_weight"):
                continue
            loaded_tensor = checkpoint.get(name)
            try:
                param.data.copy_(loaded_tensor)
            except Exception:
                print(f"{name=} {param.data.shape=} {loaded_tensor.shape=}")
                raise

        for layer_idx, block in enumerate(model.block):
            blocks_name, scales_name = (
                f"block.{layer_idx}.mlp.mlp1_weight.blocks",
                f"block.{layer_idx}.mlp.mlp1_weight.scales",
            )
            blocks, scales = checkpoint.get_mxfp4_blocks_scales(blocks_name, scales_name)
            block.mlp.mlp1_blocks = blocks
            block.mlp.mlp1_scales = scales

            blocks_name, scales_name = (
                f"block.{layer_idx}.mlp.mlp2_weight.blocks",
                f"block.{layer_idx}.mlp.mlp2_weight.scales",
            )
            blocks, scales = checkpoint.get_mxfp4_blocks_scales(blocks_name, scales_name)
            block.mlp.mlp2_blocks = blocks
            block.mlp.mlp2_scales = scales

        return model


class TokenGenerator:
    @torch.inference_mode()
    def __init__(self, checkpoint: str, device: torch.device, context: int | None = None):
        self.device = device
        self.model = TransformerMojo.from_checkpoint(checkpoint, device=self.device)
        if context is None:
            context = self.model.config.initial_context_length
        self.caches = [
            Cache(1, context, self.model.config.num_key_value_heads, device=self.device)
            for _ in range(len(self.model.block))
        ]
        self.input_token = torch.zeros(1, dtype=torch.int32, device=self.device)

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: list[int],
        stop_tokens: list[int],
        temperature: float = 1.0,
        max_tokens: int = 0,
        return_logprobs: bool = False,
    ):
        if max_tokens is None:
            max_tokens = 0
        for cache in self.caches:
            cache.reset()
        tokens = list(prompt_tokens)
        prompt_tokens = torch.as_tensor(prompt_tokens, dtype=torch.int32, device=self.device)
        if prompt_tokens.numel() > 1:
            if _PROFILE:
                _sync(self.device)
                start = time.perf_counter()
            self.model(prompt_tokens[None, :-1], caches=self.caches)
            if _PROFILE:
                _sync(self.device)
                _profile_add("prefill", time.perf_counter() - start)
        predicted_token = prompt_tokens[-1].item()
        num_generated_tokens = 0
        while max_tokens == 0 or num_generated_tokens < max_tokens:
            self.input_token[0] = predicted_token
            if _PROFILE:
                _sync(self.device)
                start = time.perf_counter()
            logits = self.model(self.input_token[None, :], caches=self.caches)[0]
            if _PROFILE:
                _sync(self.device)
                _profile_add("decode_step", time.perf_counter() - start)
            if temperature == 0.0:
                predicted_token = torch.argmax(logits[-1, :], dim=-1).item()
            else:
                probs = torch.softmax(logits * (1.0 / temperature), dim=-1)
                predicted_token = torch.multinomial(probs[-1, :], num_samples=1).item()
            tokens.append(predicted_token)
            num_generated_tokens += 1

            if return_logprobs:
                logprobs = torch.log_softmax(logits[-1, :], dim=-1)
                selected_logprobs = logprobs[predicted_token].item()
                yield predicted_token, selected_logprobs
            else:
                yield predicted_token

            if predicted_token in stop_tokens:
                break
        if _PROFILE and _PROFILE_STATS:
            print("Mojo profile (seconds):")
            for name, (total, count) in sorted(_PROFILE_STATS.items(), key=lambda x: -x[1][0]):
                avg = total / max(count, 1.0)
                print(f"- {name}: total={total:.4f}, count={int(count)}, avg={avg:.6f}")
