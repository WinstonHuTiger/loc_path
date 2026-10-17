import types
from typing import Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine import print_log


def st_topk_mask(scores: torch.Tensor,
                 k: int,
                 tau: float = 1.0,
                 training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """Straight-through top-k mask with soft gradient surrogate.

    Args:
        scores: Tensor of shape [B, N]
        k: Number of elements to select (clamped to [0, N])
        tau: Temperature for the soft surrogate
        training: Whether to use the soft surrogate (otherwise purely hard)

    Returns:
        indices: LongTensor [B, k] of top-k indices (empty if k == 0)
        mask: FloatTensor [B, N] with hard forward values and soft gradients
    """
    if scores.ndim != 2:
        raise ValueError(f"scores must be 2D (B, N); got shape {scores.shape}")
    B, N = scores.shape
    k_eff = max(0, min(int(k), int(N)))

    if training:
        s_dtype = scores.dtype
        soft = torch.softmax((scores.float() / float(tau)), dim=-1).to(s_dtype)
    else:
        soft = torch.zeros_like(scores)

    if k_eff == 0:
        idx = scores.new_zeros((B, 0), dtype=torch.long)
        hard = torch.zeros_like(scores)
        mask = hard + soft - soft.detach()
        return idx, mask

    idx = torch.topk(scores, k=k_eff, dim=-1).indices
    hard = torch.zeros_like(scores)
    hard.scatter_(1, idx, 1.0)
    mask = hard + soft - soft.detach()
    return idx, mask

class VisualProjector(nn.Module):
    """Map visual embedding dimension to the LLM hidden size."""

    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 depth: int = 2,
                 dropout: float = 0.0):
        super().__init__()
        layers = []
        d_in = dim_in
        for _ in range(max(0, depth - 1)):
            layers.append(nn.Linear(d_in, dim_out))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d_in = dim_out
        layers.append(nn.Linear(d_in, dim_out))
        layers.append(nn.LayerNorm(dim_out))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TopMRouter(nn.Module):
    def __init__(self, dim: int, temperature: float = 1.0):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.GELU(),
            nn.Linear(dim, 1, bias=False))
        self.tau = float(temperature)

    @property
    def requires_query(self) -> bool:
        return False

    def forward(self, tokens: torch.Tensor, top_m: int):
        # tokens: [B, N, D]
        raw_scores = self.scorer(tokens).squeeze(-1)         # [B, N]
        idx, mask = st_topk_mask(
            raw_scores,
            k=top_m,
            tau=self.tau,
            training=self.training,
        )
        if idx.numel() > 0:
            gathered = torch.gather(
                tokens,
                1,
                idx.unsqueeze(-1).expand(-1, -1, tokens.size(-1)))
        else:
            gathered = tokens.new_zeros(
                (tokens.size(0), 0, tokens.size(-1)), dtype=tokens.dtype, device=tokens.device)
        scores = raw_scores / self.tau
        return gathered, idx, scores, mask  # 返回scores和mask给aux loss用


class TextCondTopMRouter(nn.Module):
    """Text-conditioned top-M router.

    Supports two fusion modes between token features and a per-sample text query:
    - 'add': MLP on (v + q) after per-branch linear projections and GELU
    - 'concat': Concatenate projected v and q then score with MLP
    - 'dot': Single-head dot-product between projected keys and query
    """

    def __init__(self,
                 dim: int,
                 temperature: float = 1.0,
                 hidden: int = 1024,
                 dropout: float = 0.0,
                 fuse: str = 'add'):
        super().__init__()
        self.tau = float(temperature)
        self.ln_v = nn.LayerNorm(dim)
        self.ln_q = nn.LayerNorm(dim)
        self.fuse = fuse
        if fuse == 'add':
            self.w_v = nn.Linear(dim, hidden, bias=False)
            self.w_q = nn.Linear(dim, hidden, bias=False)
            self.scorer = nn.Sequential(
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1, bias=False),
            )
        elif fuse == 'concat':
            if hidden % 2 != 0:
                raise ValueError(
                    f"For fuse='concat', hidden must be even; got hidden={hidden}")
            h_half = hidden // 2
            self.w_v = nn.Linear(dim, h_half, bias=False)
            self.w_q = nn.Linear(dim, h_half, bias=False)
            self.scorer = nn.Sequential(
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1, bias=False),
            )
        elif fuse == 'dot':
            self.k_proj = nn.Linear(dim, dim, bias=False)
            self.q_proj = nn.Linear(dim, dim, bias=False)
        else:
            raise ValueError(f"Unsupported fuse mode: {fuse}")

    @property
    def requires_query(self) -> bool:
        return True

    def forward(self, tokens: torch.Tensor, top_m: int, query: torch.Tensor):
        """Args:
        - tokens: [B, N, D]
        - top_m: int, number of tokens to select
        - query: [B, D] pooled/condensed text embedding per sample
        Returns: (gathered_tokens, indices, scores)
        """
        B, N, D = tokens.shape
        if self.fuse == 'add':
            v = self.w_v(self.ln_v(tokens))              # [B, N, H]
            q = self.w_q(self.ln_q(query)).unsqueeze(1)  # [B, 1, H]
            h = self.scorer[0](v + q)                    # GELU
            scores = self.scorer[1:](h).squeeze(-1)            # [B, N]
        elif self.fuse == 'concat':
            v = self.w_v(self.ln_v(tokens))              # [B, N, H/2]
            q = self.w_q(self.ln_q(query)).unsqueeze(1)  # [B, 1, H/2]
            h = torch.cat([v, q.expand(-1, N, -1)], dim=-1)  # [B, N, H]
            scores = self.scorer(h).squeeze(-1)          # [B, N]
        elif self.fuse == 'dot':
            k_proj = self.k_proj(self.ln_v(tokens))       # [B, N, D]
            q_proj = self.q_proj(self.ln_q(query)).unsqueeze(-1)  # [B, D, 1]
            scores = (k_proj @ q_proj).squeeze(-1) / (D ** 0.5)  # [B, N]
        else:
            raise ValueError(f"Unsupported fuse mode: {self.fuse}")

        idx, mask = st_topk_mask(
            scores,
            k=top_m,
            tau=self.tau,
            training=self.training,
        )
        if idx.numel() > 0:
            gathered = torch.gather(tokens, 1, idx.unsqueeze(-1).expand(-1, -1, D))
        else:
            gathered = tokens.new_zeros((B, 0, D), dtype=tokens.dtype, device=tokens.device)
        scaled_scores = scores / self.tau
        return gathered, idx, scaled_scores, mask

class CrossAttnAdapter(nn.Module):
    """Flamingo-style cross attention adapter called inside decoder layers."""

    def __init__(self,
                 dim: int,
                 num_heads: int,
                 dropout: float = 0.0,
                 teacher_pool: str = "topk",   # "mean" | "max" | "topk"
                 teacher_topk_frac: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=dropout,
            batch_first=True)
        self._last_importance: Optional[torch.Tensor] = None
        # Learnable gate to modulate residual strength (float32 for init stability)
        self.gate = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        # Teacher pooling strategy for importance aggregation when collecting attention
        self.teacher_pool = teacher_pool
        self.teacher_topk_frac = teacher_topk_frac

    def forward(self,
                hidden_states: torch.Tensor,
                visual_tokens: torch.Tensor,
                collect_attn: bool = False) -> torch.Tensor:
        if visual_tokens is None:
            return hidden_states
        if visual_tokens.requires_grad:
            # Clone to avoid reusing the same autograd graph across multiple
            # checkpointed decoder layers when the router is disabled.
            visual_tokens = visual_tokens.clone()
        x = self.ln(hidden_states)
        if collect_attn:
            attn_out, attn = self.attn(
                x,
                visual_tokens,
                visual_tokens,
                need_weights=True,
                average_attn_weights=False)
            with torch.no_grad():
                # attn shape: [B, H, T_text, M]
                if self.teacher_pool == "max":
                    # Max over text length per head, then mean heads -> [B, M]
                    importance = attn.amax(dim=2).mean(dim=1)
                elif self.teacher_pool == "topk":
                    # Average top-k over text length per head, then mean heads
                    k = max(1, int(attn.size(2) * self.teacher_topk_frac))
                    topk_vals, _ = torch.topk(attn, k=k, dim=2)  # [B, H, k, M]
                    importance = topk_vals.mean(dim=(1, 2))      # [B, M]
                else:  # "mean"
                    importance = attn.mean(dim=(1, 2))           # [B, M]
                denom = importance.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                self._last_importance = importance / denom
        else:
            attn_out, _ = self.attn(
                x,
                visual_tokens,
                visual_tokens,
                need_weights=False)
            self._last_importance = None
        gate = self.gate.to(hidden_states.dtype)
        return hidden_states + gate * attn_out

    def pop_last_importance(self) -> Optional[torch.Tensor]:
        importance = self._last_importance
        self._last_importance = None
        return importance

def _resolve_layers(model: nn.Module) -> Iterable[nn.Module]:
    """Robustly locate decoder layers under various wrappers (PEFT/BNB/Deepspeed).

    Returns:
        nn.ModuleList of decoder layers.
    Raises:
        ValueError if layers cannot be found.
    """
    def _unwrap(m: nn.Module) -> nn.Module:
        # DDP / Deepspeed
        if hasattr(m, "module"):
            try:
                inner = m.module
                if isinstance(inner, nn.Module) and inner is not m:
                    return _unwrap(inner)
            except Exception:
                pass
        # PEFT
        if hasattr(m, "get_base_model"):
            try:
                inner = m.get_base_model()
                if isinstance(inner, nn.Module) and inner is not m:
                    return _unwrap(inner)
            except Exception:
                pass
        if hasattr(m, "base_model"):
            try:
                inner = m.base_model
                if isinstance(inner, nn.Module) and inner is not m:
                    return _unwrap(inner)
            except Exception:
                pass
        return m

    m = _unwrap(model)

    # Candidate roots inside the unwrapped model
    roots = [m]
    for root_name in ("model", "language_model", "transformer", "backbone", "decoder"):
        r = getattr(m, root_name, None)
        if isinstance(r, nn.Module):
            roots.append(r)

    # Try common container names directly
    for r in roots:
        for name in ("layers", "h", "blocks", "decoder_layers"):
            layers = getattr(r, name, None)
            if isinstance(layers, nn.ModuleList) and len(layers) > 0:
                print_log(f"[qwen_vis_adapter] Resolved decoder layers via {type(r).__name__}.{name} "
                          f"(len={len(layers)})", "current")
                return layers
        # also check nested .model inside the root
        inner = getattr(r, "model", None)
        if isinstance(inner, nn.Module):
            for name in ("layers", "h", "blocks", "decoder_layers"):
                layers = getattr(inner, name, None)
                if isinstance(layers, nn.ModuleList) and len(layers) > 0:
                    print_log(f"[qwen_vis_adapter] Resolved decoder layers via "
                              f"{type(r).__name__}.model.{name} (len={len(layers)})", "current")
                    return layers

    # Fallback: deep search for a ModuleList that looks like decoder layers
    for n, mod in m.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) > 0:
            elem = mod[0]
            has_decoder_attrs = any(
                hasattr(elem, k) for k in ("self_attn", "self_attention", "attention", "attn", "mlp", "mixer")
            )
            if has_decoder_attrs:
                print_log(f"[qwen_vis_adapter] Resolved decoder layers via deep search at '{n}' "
                          f"(len={len(mod)})", "current")
                return mod

    # Nothing found — raise a rich error for debugging
    root_types = [type(r).__name__ for r in roots if isinstance(r, nn.Module)]
    raise ValueError(
        "Unable to locate decoder layers for the supplied model. "
        f"Checked roots={root_types} with container names "
        "['layers','h','blocks','decoder_layers'] and deep searched ModuleList."
    )


class VisionAdapterController(nn.Module):
    """Keep visual tokens and route them through patched decoder layers."""

    def __init__(self,
                 d_llm: int,
                 n_heads: int,
                 layer_indices: Sequence[int],
                 dropout: float = 0.0,
                 teacher_pool: str = "topk",
                 teacher_topk_frac: float = 0.1):
        super().__init__()
        self.layer_indices = list(layer_indices)
        self.teacher_pool = teacher_pool
        self.teacher_topk_frac = teacher_topk_frac
        self.adapters = nn.ModuleDict({
            str(i): CrossAttnAdapter(
                d_llm,
                n_heads,
                dropout,
                teacher_pool=self.teacher_pool,
                teacher_topk_frac=self.teacher_topk_frac,
            )
            for i in self.layer_indices
        })

        self._visual_tokens: Optional[torch.Tensor] = None
        self.collect_attn: bool = False
        self._importance: list[torch.Tensor] = []

    def set_visual_tokens(self, tokens: torch.Tensor) -> None:
        self._visual_tokens = tokens

    def clear(self) -> None:
        self._visual_tokens = None
        self._importance.clear()

    def forward(self,
                layer_idx: int,
                hidden_states: torch.Tensor) -> torch.Tensor:
        key = str(layer_idx)
        try: 
            adapter = self.adapters[key]
        except KeyError:
            return hidden_states
        visual_tokens = self._visual_tokens
        if visual_tokens is not None and visual_tokens.size(0) != hidden_states.size(0):
            # HuggingFace's generate() may expand the batch (e.g., sampling/beam).
            # Broadcast vision tokens along the batch dimension so shapes stay aligned.
            b_hidden = hidden_states.size(0)
            b_visual = visual_tokens.size(0)
            if b_hidden % max(1, b_visual) != 0:
                raise ValueError(
                    f"VisionAdapterController batch mismatch: hidden batch={b_hidden} "
                    f"vs visual batch={b_visual}. Cannot broadcast visual tokens"
                )
            repeat = b_hidden // b_visual
            if repeat > 1:
                visual_tokens = visual_tokens.repeat_interleave(repeat, dim=0)
        out = adapter(hidden_states,
                      visual_tokens,
                      collect_attn=self.collect_attn)
        if self.collect_attn:
            importance = adapter.pop_last_importance()
            if importance is not None:
                self._importance.append(importance)
        return out

    def pop_visual_importance(self) -> Optional[torch.Tensor]:
        if not self._importance:
            return None
        stacked = torch.stack(self._importance, dim=0)
        self._importance.clear()
        aggregated = stacked.mean(dim=0)
        denom = aggregated.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return aggregated / denom


def patch_qwen_for_cross_attn(model: nn.Module,
                              controller: VisionAdapterController,
                              layer_indices: Optional[Sequence[int]] = None) -> None:
    """Monkey patch decoder layers to insert cross attention adapters."""
    layers = _resolve_layers(model)
    if layer_indices is not None:
        controller.layer_indices = list(layer_indices)
    patched = set()
    for idx, layer in enumerate(layers):
        if str(idx) not in controller.adapters:
            continue
        if hasattr(layer, '_orig_forward'):
            continue

        layer._orig_forward = layer.forward

        def make_forward(layer_index: int):
            def new_forward(self, *args, **kwargs):
                out = self._orig_forward(*args, **kwargs)
                if isinstance(out, tuple):
                    hidden = out[0]
                    hidden = controller(layer_index, hidden)
                    out = (hidden,) + out[1:]
                else:
                    out = controller(layer_index, out)
                return out

            return new_forward

        layer.forward = types.MethodType(make_forward(idx), layer)
        patched.add(idx)

    missing = set(controller.layer_indices) - patched
    if missing:
        raise ValueError(f'Layer indices {sorted(missing)} not found during patching.')


def prepare_visual_tokens(tokens: torch.Tensor,
                          projector: Optional[VisualProjector] = None,
                          router: Optional[TopMRouter] = None,
                          top_m: Optional[int] = None) -> torch.Tensor:
    """Project and sparsify visual tokens before feeding them to the controller."""
    visual = tokens
    if projector is not None:
        visual = projector(visual)
    if router is not None and top_m is not None:
        out = router(visual, top_m=top_m)
        visual = out[0]  # (tokens, idx, [scores])
    return visual
