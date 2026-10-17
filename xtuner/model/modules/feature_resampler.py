import math
from typing import List, Optional, Sequence, Tuple, Union, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from xtuner.model.torchscale.model.longnet_context import make_longnet_context


def trunc_normal_(tensor, mean: float = 0.0, std: float = 1.0) -> torch.Tensor:
    """Truncated normal initialisation used for learnable latent queries."""
    with torch.no_grad():
        size = tensor.shape
        tmp = tensor.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)
    return tensor


def get_1d_sincos_pos_embed(embed_dim: int,
                            length: int,
                            device: Optional[torch.device] = None) -> torch.Tensor:
    """Return [length, embed_dim] 1D sin_cos positional embeddings."""
    assert embed_dim % 2 == 0, 'Embed dim must be even for sin/cos split.'
    pos = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    omega = torch.arange(embed_dim // 2,
                         device=device,
                         dtype=torch.float32)[None, :]
    omega = 1.0 / (10000 ** (omega / (embed_dim / 2)))
    out = pos * omega
    return torch.cat([out.sin(), out.cos()], dim=-1)


def fourier_2d_pos(
    xy: torch.Tensor,
    dim: int,
    num_bands: int = 64,
    max_freq: float = 10000.0,
    min_val: Optional[torch.Tensor] = None,
    max_val: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return [*, dim] 2D Fourier positional embeddings (normalized to [0, 1])."""
    assert dim % 2 == 0, 'Embedding dim must be even.'
    if xy.numel() == 0:
        shape = xy.shape[:-1] + (dim,)
        return xy.new_zeros(shape)

    if min_val is None or max_val is None:
        min_val = xy.amin(dim=-2, keepdim=True)
        max_val = xy.amax(dim=-2, keepdim=True)
    denom = (max_val - min_val).clamp(min=eps)
    xy_norm = (xy - min_val) / denom
    xy_norm = xy_norm.clamp(0.0, 1.0)

    freqs = torch.logspace(
        0.,
        math.log10(max_freq),
        steps=num_bands,
        device=xy.device,
        dtype=xy.dtype)
    angles = xy_norm.unsqueeze(-1) * freqs  # [..., 2, F]
    sin = torch.sin(2 * math.pi * angles)
    cos = torch.cos(2 * math.pi * angles)
    enc = torch.cat([sin, cos], dim=-2).flatten(-2)  # [..., 4F]

    if enc.shape[-1] < dim:
        pad = torch.zeros(*enc.shape[:-1],
                          dim - enc.shape[-1],
                          device=enc.device,
                          dtype=enc.dtype)
        enc = torch.cat([enc, pad], dim=-1)
    elif enc.shape[-1] > dim:
        enc = enc[..., :dim]
    return enc


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class MultiheadCrossAttentionTopK(nn.Module):
    """Cross-attention with optional top-k sparsification.

    Positional encodings are injected only on the key pathway as requested.
    """

    def __init__(self,
                 dim: int,
                 num_heads: int = 8,
                 topk: Optional[int] = None,
                 attn_drop: float = 0.0,
                 proj_drop: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.topk = topk

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        q: torch.Tensor,  # [B, Q, C]
        k: torch.Tensor,  # [B, N, C]
        v: torch.Tensor,  # [B, N, C]
        *,
        k_pos: Optional[torch.Tensor] = None,  # [B, N, C]
        key_padding_mask: Optional[torch.Tensor] = None,  # [B, N] bool; True for pad
        need_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, Q, C = q.shape
        N = k.shape[1]
        H = self.num_heads
        Hd = self.head_dim

        if k_pos is not None:
            k = k + k_pos

        q = self.q_proj(q).view(B, Q, H, Hd).permute(0, 2, 1, 3)
        k = self.k_proj(k).view(B, N, H, Hd).permute(0, 2, 1, 3)
        v = self.v_proj(v).view(B, N, H, Hd).permute(0, 2, 1, 3)

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask[:, None, None, :].to(attn_logits.dtype)
            attn_logits = attn_logits.masked_fill(key_padding_mask.bool(), float('-inf'))

        if self.topk is not None and self.topk < N:
            topk_val, topk_idx = torch.topk(attn_logits, k=self.topk, dim=-1)
            fill = torch.full_like(attn_logits, float('-inf'))
            attn_logits = fill.scatter(-1, topk_idx, topk_val)

        attn = torch.softmax(attn_logits.float(), dim=-1).to(attn_logits.dtype)
        # When all keys are masked, softmax receives all -inf and returns NaNs.
        # Convert those NaNs to zeros so the attention output becomes zero.
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, Q, C)
        out = self.proj_drop(self.out_proj(out))
        if need_weights:
            return out, attn
        return out, None


class DepthwiseConvMixer(nn.Module):
    """Depthwise separable conv mixer used as a lightweight context encoder."""

    def __init__(self, dim: int, kernel_size: int = 7, dilation: int = 1):
        super().__init__()
        padding = (kernel_size // 2) * dilation
        self.dw = nn.Conv1d(dim,
                            dim,
                            kernel_size=kernel_size,
                            padding=padding,
                            dilation=dilation,
                            groups=dim)
        self.pw = nn.Conv1d(dim, dim, kernel_size=1)
        self.ln = nn.LayerNorm(dim)
        self.act = nn.GELU()

    def forward(self,
                x: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        mask = None
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(-1).bool()
            x = x.masked_fill(mask, 0.0)
            residual = residual.masked_fill(mask, 0.0)

        y = x.transpose(1, 2)
        y = self.dw(y)
        y = self.pw(y)
        y = y.transpose(1, 2)
        y = self.act(self.ln(y))
        out = residual + y

        if mask is not None:
            out = out.masked_fill(mask, 0.0)
        return out


class LocalSelfAttention(nn.Module):
    """Sliding-window self-attention context mixer."""

    def __init__(self,
                 dim: int,
                 num_heads: int,
                 window: int = 128,
                 dilation: int = 1,
                 dropout: float = 0.0):
        super().__init__()
        assert window > 0, 'window must be positive'
        assert dilation >= 1, 'dilation must be >= 1'
        self.window = window
        self.dilation = dilation
        self.mha = nn.MultiheadAttention(dim,
                                         num_heads,
                                         batch_first=True,
                                         dropout=dropout)
        self.ln = nn.LayerNorm(dim)

    def forward(self,
                x: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        mask = None
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(-1).bool()
            residual = residual.masked_fill(mask, 0.0)
            x = x.masked_fill(mask, 0.0)

        norm_x = self.ln(x)
        B, N, _ = norm_x.shape
        window = self.window
        stride = max(window // 2, 1) * self.dilation

        out = torch.zeros_like(norm_x)
        counts = torch.zeros(B, N, 1, device=norm_x.device, dtype=norm_x.dtype)

        start = 0
        while start < N:
            end = min(start + window, N)
            chunk = norm_x[:, start:end, :]
            chunk_mask = None
            if key_padding_mask is not None:
                chunk_mask = key_padding_mask[:, start:end]
            chunk_out, _ = self.mha(chunk,
                                    chunk,
                                    chunk,
                                    need_weights=False,
                                    key_padding_mask=chunk_mask)
            out[:, start:end, :] += chunk_out
            counts[:, start:end, :] += 1.0
            start += stride

        counts = counts.clamp_min(1.0)
        mixed = out / counts
        out = residual + mixed

        if mask is not None:
            out = out.masked_fill(mask, 0.0)
        return out


class LatentSelfAttnBlock(nn.Module):
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 mlp_ratio: float = 4.0,
                 drop: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.mha = nn.MultiheadAttention(dim,
                                         num_heads,
                                         batch_first=True,
                                         dropout=drop)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.ln1(z)
        x = self.mha(x, x, x, need_weights=False)[0]
        z = z + x
        z = z + self.mlp(self.ln2(z))
        return z


class PerceiverResamplerTopK(nn.Module):
    """Learns latent queries that attend to variable-length feature tokens."""

    def __init__(self,
                 d_model: int = 512,
                 n_latents: int = 256,
                 num_heads: int = 8,
                 n_layers: int = 1,
                 topk: Optional[int] = 64,
                 mlp_ratio: float = 4.0,
                 q_init_std: float = 0.02,
                 attn_drop: float = 0.0,
                 proj_drop: float = 0.0,
                 latent_drop: float = 0.0,
                 use_latent_self_attn: bool = True,
                 context_mode: str = 'none',
                 ctx_layers: int = 0,
                 ctx_kernel_size: int = 7,
                 ctx_window: int = 128,
                 ctx_dilation: int = 1,
                 dense_latents: int = 0,
                 ctx_longnet_segments: Optional[Sequence[int]] = None,
                 ctx_longnet_dilations: Optional[Union[Sequence[int], Sequence[Sequence[int]]]] = None,
                 ctx_longnet_use_xpos: bool = True,
                 ctx_longnet_mlp_ratio: Optional[float] = None):
        super().__init__()
        self.d_model = d_model
        self.n_latents = n_latents
        self.num_heads = num_heads
        self.n_layers = n_layers
        self.use_latent_self_attn = use_latent_self_attn
        self.context_mode = context_mode
        self.dense_latents = dense_latents
        self.ctx_longnet_segments = (list(ctx_longnet_segments)
                                     if ctx_longnet_segments is not None else
                                     [1024, 2048, 4096])
        self.ctx_longnet_use_xpos = ctx_longnet_use_xpos
        self.ctx_longnet_mlp_ratio = ctx_longnet_mlp_ratio
        if ctx_longnet_dilations is None:
            self.ctx_longnet_dilations: List[List[int]] = [[1, 2, 4]]
        elif (len(ctx_longnet_dilations) > 0 and isinstance(
                ctx_longnet_dilations[0], Sequence)
              and not isinstance(ctx_longnet_dilations[0], (int, float, str, bytes))):
            nested = cast(Sequence[Sequence[int]], ctx_longnet_dilations)
            self.ctx_longnet_dilations = [list(seq) for seq in nested]
        else:
            flat = cast(Sequence[int], ctx_longnet_dilations)
            self.ctx_longnet_dilations = [list(flat)]

        if isinstance(topk, Sequence) and not isinstance(topk, (str, bytes)):
            if len(topk) == 0:
                raise ValueError('topk sequence must not be empty.')
            if len(topk) == 1:
                topk_per_layer = list(topk) * n_layers
            elif len(topk) == n_layers:
                topk_per_layer = list(topk)
            else:
                raise ValueError('topk sequence length must be 1 or n_layers.')
        else:
            topk_per_layer = [topk for _ in range(n_layers)]
        self.topk_per_layer = topk_per_layer

        self.query = nn.Parameter(torch.zeros(n_latents, d_model))
        trunc_normal_(self.query, std=q_init_std)

        if context_mode not in {'none', 'conv', 'local_attn', 'longnet'}:
            raise ValueError(f'Unsupported context_mode: {context_mode}')
        if ctx_layers < 0:
            raise ValueError('ctx_layers must be >= 0.')

        self.ctx_encoder: Optional[nn.Module] = None
        ctx_blocks = []
        if context_mode != 'none' and ctx_layers > 0:
            if context_mode == 'longnet':
                layerwise_dilations: List[List[int]] = []
                for layer_idx in range(ctx_layers):
                    if layer_idx < len(self.ctx_longnet_dilations):
                        layerwise_dilations.append(self.ctx_longnet_dilations[layer_idx])
                    else:
                        layerwise_dilations.append(self.ctx_longnet_dilations[-1])
                base_dil = layerwise_dilations[0] if layerwise_dilations else [1, 2, 4]
                mlp_ratio_ctx = self.ctx_longnet_mlp_ratio if self.ctx_longnet_mlp_ratio is not None else mlp_ratio
                self.ctx_encoder = make_longnet_context(
                    d_model=d_model,
                    num_heads=num_heads,
                    n_layers=ctx_layers,
                    mlp_ratio=mlp_ratio_ctx,
                    dropout=proj_drop,
                    drop_path_rate=0.05,
                    seg=self.ctx_longnet_segments,
                    dil=base_dil,
                    use_xpos=self.ctx_longnet_use_xpos,
                    layer_dilations=layerwise_dilations,
                )
            else:
                for layer_idx in range(ctx_layers):
                    if context_mode == 'conv':
                        ctx_blocks.append(
                            DepthwiseConvMixer(d_model,
                                               kernel_size=ctx_kernel_size,
                                               dilation=ctx_dilation))
                    elif context_mode == 'local_attn':
                        ctx_blocks.append(
                            LocalSelfAttention(d_model,
                                               num_heads,
                                               window=ctx_window,
                                               dilation=ctx_dilation,
                                               dropout=proj_drop))
        self.ctx_blocks = nn.ModuleList(ctx_blocks) if ctx_blocks else None

        self.attn_blocks = nn.ModuleList([
            MultiheadCrossAttentionTopK(d_model,
                                        num_heads,
                                        topk=self.topk_per_layer[idx],
                                        attn_drop=attn_drop,
                                        proj_drop=proj_drop)
            for idx in range(n_layers)
        ])
        self.ln_q = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.ffn = nn.ModuleList(
            [MLP(d_model, int(d_model * mlp_ratio), drop=proj_drop) for _ in range(n_layers)])
        self.ln_ffn = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.latent_blocks = (nn.ModuleList([
            LatentSelfAttnBlock(d_model,
                                num_heads,
                                mlp_ratio=mlp_ratio,
                                drop=latent_drop) for _ in range(n_layers)
        ]) if use_latent_self_attn else None)

    def forward(
        self,
        x: torch.Tensor,
        *,
        k_pos: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Args:
        - x:      [B, N, C] visible tokens.
        - k_pos:  [B, N, C] positional encoding (applied to K only).
        - key_padding_mask: [B, N] mask, True for padding tokens.
        """
        B, _, _ = x.shape

        if self.ctx_encoder is not None:
            ctx_out = self.ctx_encoder(
                src_tokens=None,
                token_embeddings=x,
                encoder_padding_mask=key_padding_mask,
                features_only=True,
            )
            x = ctx_out["encoder_out"]
        elif self.ctx_blocks is not None:
            for block in self.ctx_blocks:
                x = block(x, key_padding_mask=key_padding_mask)

        q = self.query.unsqueeze(0).expand(B, -1, -1)

        last_attn = None
        for idx in range(self.n_layers):
            q_norm = self.ln_q[idx](q)
            attn_block = self.attn_blocks[idx]
            attn = None
            dense_count = max(int(self.dense_latents or 0), 0)
            dense_count = min(dense_count, q_norm.size(1))
            if dense_count > 0:
                q_dense = q_norm[:, :dense_count, :]
                q_sparse = q_norm[:, dense_count:, :]
                out_parts: List[torch.Tensor] = []
                attn_parts: List[torch.Tensor] = []
                orig_topk = attn_block.topk
                if q_dense.size(1) > 0:
                    attn_block.topk = None
                    out_dense, attn_dense = attn_block(
                        q_dense,
                        x,
                        x,
                        k_pos=k_pos,
                        key_padding_mask=key_padding_mask,
                        need_weights=need_attn)
                    out_parts.append(out_dense)
                    if attn_dense is not None:
                        attn_parts.append(attn_dense)
                if q_sparse.size(1) > 0:
                    attn_block.topk = orig_topk
                    out_sparse, attn_sparse = attn_block(
                        q_sparse,
                        x,
                        x,
                        k_pos=k_pos,
                        key_padding_mask=key_padding_mask,
                        need_weights=need_attn)
                    out_parts.append(out_sparse)
                    if attn_sparse is not None:
                        attn_parts.append(attn_sparse)
                attn_block.topk = orig_topk
                if len(out_parts) == 1:
                    out = out_parts[0]
                else:
                    out = torch.cat(out_parts, dim=1)
                if need_attn and attn_parts:
                    attn = attn_parts[0] if len(attn_parts) == 1 else torch.cat(attn_parts, dim=2)
            else:
                out, attn = attn_block(q_norm,
                                       x,
                                       x,
                                       k_pos=k_pos,
                                       key_padding_mask=key_padding_mask,
                                       need_weights=need_attn)
            q = q + out
            q = q + self.ffn[idx](self.ln_ffn[idx](q))
            if self.use_latent_self_attn:
                q = self.latent_blocks[idx](q)
            if need_attn:
                last_attn = attn

        return q, last_attn
