"""BC1S/Conv transformer for fixed-shape ANE export of the English Laya model.

Port/adaptation of mizorewww/laya-coreml's
experiments/ane_engineering/model.py (Apache-2.0; source revision and notices
are recorded in the repository NOTICE). This version targets the English
ModernBERT checkpoint and loads its tensors/config directly instead of a source
model object.

Layout follows Apple's ml-ane-transformers principles: activations are
[B, C, 1, L], dense weights become 1x1 conv kernels, attention is per-head
explicit einsum, masks are additive fp16. Weights load straight from
converted-fp16/model.safetensors (identity names); no torch model needed.
"""

from typing import Any, cast

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from safetensors import safe_open


def conv_from_weights(weight: torch.Tensor, bias: torch.Tensor | None = None) -> nn.Conv2d:
    result = nn.Conv2d(weight.shape[1], weight.shape[0], 1, bias=bias is not None)
    result.weight = nn.Parameter(weight.detach()[:, :, None, None])
    if bias is not None:
        result.bias = nn.Parameter(bias.detach())
    return result


class ChannelNorm(nn.Module):
    """LayerNorm over the channel axis of BC1S, original affine ordering."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None = None,
                 eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(weight.detach()[None, :, None, None])
        self.bias: nn.Parameter | None = None
        if bias is not None:
            self.bias = nn.Parameter(bias.detach()[None, :, None, None])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        centered = x - x.mean(dim=1, keepdim=True)
        out = centered * (centered.square().mean(dim=1, keepdim=True) + self.eps).rsqrt()
        out = out * self.weight
        return out if self.bias is None else out + self.bias


class ConvAttention(nn.Module):
    """Per-head explicit attention in BC1S. rope=True applies rotate-half RoPE

    with precomputed [1, head_dim, 1, L] cos/sin buffers."""
    cos: torch.Tensor
    sin: torch.Tensor

    def __init__(self, w_qkv: torch.Tensor, b_qkv: torch.Tensor | None, w_out: torch.Tensor,
                 b_out: torch.Tensor | None, num_heads: int, head_dim: int, *,
                 rope: bool, length: int, theta: float | None = None) -> None:
        super().__init__()
        self.rope = rope
        self.heads, self.dim = num_heads, head_dim
        self.qkv = conv_from_weights(w_qkv, b_qkv)
        self.out = conv_from_weights(w_out, b_out)
        if rope:
            assert theta is not None
            inv = 1.0 / (float(theta) ** (torch.arange(0, head_dim, 2).float() / head_dim))
            freqs = torch.arange(length).float()[:, None] * inv[None, :]  # [L, D/2]
            self.register_buffer("cos", freqs.cos().T[None, :, None, :])  # [1, D/2, 1, L]
            self.register_buffer("sin", freqs.sin().T[None, :, None, :])

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        left, right = x.chunk(2, dim=1)
        return torch.cat(
            (left * self.cos - right * self.sin, right * self.cos + left * self.sin), dim=1
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        q, k, v = self.qkv(x).chunk(3, dim=1)
        output: list[torch.Tensor] = []
        for qi, ki, vi in zip(
            q.split(self.dim, dim=1), k.split(self.dim, dim=1), v.split(self.dim, dim=1)
        ):
            if self.rope:
                qi, ki = self.rotate(qi), self.rotate(ki)
            scores = torch.einsum("bchq,bkhc->bkhq", qi, ki.transpose(1, 3)) * (self.dim**-0.5)
            probabilities = F.softmax(scores + mask, dim=1)
            output.append(torch.einsum("bkhq,bchk->bchq", probabilities, vi))
        return cast(torch.Tensor, self.out(torch.cat(output, dim=1)))


class ConvMLP(nn.Module):
    """GLU MLP: Wi -> split(value, gate) -> gelu(value) * gate -> Wo."""

    def __init__(self, wi: torch.Tensor, wo: torch.Tensor) -> None:
        super().__init__()
        self.Wi, self.Wo = conv_from_weights(wi), conv_from_weights(wo)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.Wi(x).chunk(2, dim=1)
        return cast(torch.Tensor, self.Wo(F.gelu(value) * gate))


class ConvEncoderLayer(nn.Module):
    def __init__(self, tensors: dict[str, torch.Tensor], prefix: str, cfg: dict[str, Any],
                 layer_idx: int, length: int) -> None:
        super().__init__()
        self.kind: str = cfg["layer_types"][layer_idx]
        eps = cfg["norm_eps"]
        attn_norm_w = tensors.get(prefix + "attn_norm.weight")  # layer 0 has none
        self.attn_norm: nn.Module = (
            nn.Identity()
            if attn_norm_w is None
            else ChannelNorm(attn_norm_w, eps=eps)
        )
        theta = cfg["rope_parameters"][self.kind]["rope_theta"]
        self.attn = ConvAttention(
            tensors[prefix + "attn.Wqkv.weight"],
            tensors.get(prefix + "attn.Wqkv.bias"),
            tensors[prefix + "attn.Wo.weight"],
            tensors.get(prefix + "attn.Wo.bias"),
            cfg["num_attention_heads"],
            cfg["hidden_size"] // cfg["num_attention_heads"],
            rope=True,
            length=length,
            theta=theta,
        )
        self.mlp_norm = ChannelNorm(tensors[prefix + "mlp_norm.weight"], eps=eps)
        self.mlp = ConvMLP(tensors[prefix + "mlp.Wi.weight"], tensors[prefix + "mlp.Wo.weight"])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), mask)
        return cast(torch.Tensor, x + self.mlp(self.mlp_norm(x)))


class ConvHeadLayer(nn.Module):
    """torch.nn.TransformerEncoderLayer(norm_first=True): biased MHA (no RoPE),
    ReLU FFN, LayerNorms with bias."""

    def __init__(self, tensors: dict[str, torch.Tensor], prefix: str, cfg: dict[str, Any],
                 length: int) -> None:
        super().__init__()
        d = cfg["hidden_size"]
        nhead = max(1, d // 64)
        eps = 1e-5  # torch.nn.LayerNorm default in the source head
        self.norm1 = ChannelNorm(tensors[prefix + "norm1.weight"], tensors[prefix + "norm1.bias"], eps)
        self.norm2 = ChannelNorm(tensors[prefix + "norm2.weight"], tensors[prefix + "norm2.bias"], eps)
        self.attn = ConvAttention(
            tensors[prefix + "self_attn.in_proj_weight"],
            tensors[prefix + "self_attn.in_proj_bias"],
            tensors[prefix + "self_attn.out_proj.weight"],
            tensors[prefix + "self_attn.out_proj.bias"],
            nhead,
            d // nhead,
            rope=False,
            length=length,
        )
        self.linear1 = conv_from_weights(tensors[prefix + "linear1.weight"], tensors[prefix + "linear1.bias"])
        self.linear2 = conv_from_weights(tensors[prefix + "linear2.weight"], tensors[prefix + "linear2.bias"])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask)
        return cast(torch.Tensor, x + self.linear2(F.relu(self.linear1(self.norm2(x)))))


class ConvBody(nn.Module):
    """Embedding norm + encoder + type addition + decision head + scorer.

    Inputs (all fp16):
      embeddings   [1, D, 1, L]   raw token-embedding rows, channel-first
      full_mask    [1, L, 1, L]   additive key-padding mask (0 / -1e4), [B,K,1,Q]
      local_mask   [1, L, 1, L]   additive sliding-window mask
      type_vectors [1, D, 1, 1]   question-type embedding row
      marker_map   [1, L, 1, 32]  one-hot marker selectors
    Outputs: option logits [1,1,1,32] and CLS hidden [1,D,1,1].
    The small act head stays on the host in fp32.
    """

    MAX_MARKERS = 32

    def __init__(self, model_dir: str, length: int) -> None:
        super().__init__()
        import json
        import os

        cfg: dict[str, Any] = json.load(open(os.path.join(model_dir, "encoder", "config.json")))
        agent_cfg: dict[str, Any] = json.load(open(os.path.join(model_dir, "rl_agent_config.json")))
        tensors: dict[str, torch.Tensor] = {}
        with safe_open(os.path.join(model_dir, "model.safetensors"), framework="pt") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
        eps = cfg["norm_eps"]
        self.embedding_norm = ChannelNorm(tensors["encoder.embeddings.norm.weight"], eps=eps)
        self.layers = nn.ModuleList(
            [
                ConvEncoderLayer(tensors, "encoder.layers.%d." % i, cfg, i, length)
                for i in range(cfg["num_hidden_layers"])
            ]
        )
        self.final_norm = ChannelNorm(tensors["encoder.final_norm.weight"], eps=eps)
        self.head = nn.ModuleList(
            [
                ConvHeadLayer(tensors, "head.layers.%d." % i, cfg, length)
                for i in range(agent_cfg["head_layers"])
            ]
        )
        self.scorer = nn.Sequential(
            ChannelNorm(tensors["scorer.0.weight"], tensors["scorer.0.bias"], eps),
            conv_from_weights(tensors["scorer.1.weight"], tensors["scorer.1.bias"]),
            nn.GELU(),
            conv_from_weights(tensors["scorer.3.weight"], tensors["scorer.3.bias"]),
        )
        self.cfg: dict[str, Any] = cfg

    def forward(self, embeddings: torch.Tensor, full_mask: torch.Tensor, local_mask: torch.Tensor,
                type_vectors: torch.Tensor, marker_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.embedding_norm(embeddings)
        for layer in self.layers:
            x = layer(x, full_mask if layer.kind == "full_attention" else local_mask)
        x = self.final_norm(x) + type_vectors
        for layer in self.head:
            x = layer(x, full_mask)
        markers = torch.einsum("bkhq,bchk->bchq", marker_map, x)
        return self.scorer(markers), x[:, :, :, :1]
