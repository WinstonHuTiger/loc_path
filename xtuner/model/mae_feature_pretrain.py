from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModel

from xtuner.registry import BUILDER
from .modules.feature_resampler import (PerceiverResamplerTopK,
                                        fourier_2d_pos,
                                        get_1d_sincos_pos_embed)


def entropy_from_probs(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = p.clamp_min(eps)
    return -(p * p.log()).sum(dim=-1)


def attention_decorrelation_loss(attn_mean: torch.Tensor) -> torch.Tensor:
    a = attn_mean / (attn_mean.norm(dim=-1, keepdim=True) + 1e-6)
    gram = torch.einsum('bqn,bkn->bqk', a, a)
    eye = torch.eye(gram.size(-1), device=gram.device)[None]
    return ((gram - eye)**2).sum(dim=(1, 2)).mean() / (gram.size(-1)**2)


def feature_covariance_decorrelation(z: torch.Tensor) -> torch.Tensor:
    B, Q, D = z.shape
    reshaped = z.reshape(B * Q, D)
    reshaped = reshaped - reshaped.mean(dim=0, keepdim=True)
    cov = (reshaped.T @ reshaped) / max(reshaped.shape[0] - 1, 1)
    cov = cov - torch.diag(torch.diag(cov))
    return (cov**2).sum() / (D * D)


def per_query_peakiness_variance(attn_mean: torch.Tensor) -> torch.Tensor:
    sq = (attn_mean**2).sum(dim=-1)
    return sq.var(dim=1).mean()


def collect_stats(attn_probs: torch.Tensor,
                  taus: List[float],
                  valid_mask: Optional[torch.Tensor] = None) -> Dict[float, float]:
    """
    Mask-aware per-latent coverage:
      coverage_tau = E_{b,q}[ (# {n valid | p_{b,q}(n) > tau}) / Nv(b) ].
    If valid_mask is None, falls back to dense mean (old behavior).
    """
    stats: Dict[float, float] = {}
    if valid_mask is None:
        for tau in taus:
            cov = (attn_probs > tau).float().mean(dim=-1).mean(dim=-1).mean()
            stats[tau] = float(cov.item())
        return stats

    # valid_mask: [B, Nv] -> [B,1,Nv]
    vm = valid_mask.unsqueeze(1).to(attn_probs.dtype)
    Nv = vm.sum(dim=-1).clamp_min(1.0)        # [B,1]
    for tau in taus:
        hit = ((attn_probs > tau).float() * vm).sum(dim=-1) / Nv  # [B,Q]
        stats[tau] = float(hit.mean().item())
    return stats


class FeatureMAEWithResampler(nn.Module):
    """Feature-token MAE tailored for xtuner's training loop."""

    def __init__(
        self,
        d_in: int = 1024,
        d_model: int = 512,
        n_latents: int = 256,
        num_heads: int = 8,
        resampler_layers: int = 1,
        resampler_topk: Union[int, Sequence[Optional[int]], None] = 64,
        use_latent_self_attn: bool = True,
        latent_drop: float = 0.0,
        dec_depth: int = 2,
        dense_latents: int = 0,
        dec_mlp_ratio: float = 4.0,
        mask_token_init_std: float = 0.02,
        use_pos_embed: bool = True,
        pos_embed_type: str = '1d',  # 'none' | '1d' | '2d_fourier'
        fourier_num_bands: int = 64,
        fourier_max_freq: float = 10000.0,
        context_mode: str = 'none',
        ctx_layers: int = 0,
        ctx_kernel_size: int = 7,
        ctx_window: int = 128,
        ctx_dilation: int = 1,
        ctx_longnet_segments: Optional[Sequence[int]] = None,
        ctx_longnet_dilations: Optional[Union[Sequence[int], Sequence[Sequence[int]]]] = None,
        ctx_longnet_use_xpos: bool = True,
        ctx_longnet_mlp_ratio: Optional[float] = None,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.use_pos_embed = use_pos_embed and pos_embed_type != 'none'
        self.pos_embed_type = pos_embed_type
        self.fourier_num_bands = fourier_num_bands
        self.fourier_max_freq = fourier_max_freq

        if self.pos_embed_type not in {'none', '1d', '2d_fourier'}:
            raise ValueError(
                f'Unsupported pos_embed_type: {self.pos_embed_type}')

        self.in_proj = nn.Linear(d_in, d_model)

        self.resampler = PerceiverResamplerTopK(
            d_model=d_model,
            n_latents=n_latents,
            num_heads=num_heads,
            n_layers=resampler_layers,
            topk=resampler_topk,
            use_latent_self_attn=use_latent_self_attn,
            latent_drop=latent_drop,
            context_mode=context_mode,
            ctx_layers=ctx_layers,
            ctx_kernel_size=ctx_kernel_size,
            ctx_window=ctx_window,
            ctx_dilation=ctx_dilation,
            dense_latents=dense_latents,
            ctx_longnet_segments=ctx_longnet_segments,
            ctx_longnet_dilations=ctx_longnet_dilations,
            ctx_longnet_use_xpos=ctx_longnet_use_xpos,
            ctx_longnet_mlp_ratio=ctx_longnet_mlp_ratio,
        )

        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.mask_token, std=mask_token_init_std)
        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=int(d_model * dec_mlp_ratio),
                batch_first=True,
                activation='gelu',
            ) for _ in range(dec_depth)
        ])
        self.dec_out = nn.Linear(d_model, d_in)

    @staticmethod
    def _ensure_mask_counts(nv: int, mask_ratio: float) -> Tuple[int, int]:
        if nv <= 1:
            return nv, 0
        nm = int(round(nv * mask_ratio))
        nm = max(1, nm)
        nm = min(nm, nv - 1)
        nv_keep = nv - nm
        return nv_keep, nm

    def _build_position_table(self, valid_mask: torch.Tensor,
                              device: torch.device) -> torch.Tensor:
        lengths = valid_mask.sum(dim=1)
        max_len = int(lengths.max().item())
        if max_len == 0:
            max_len = 1
        return get_1d_sincos_pos_embed(self.d_model, max_len, device=device)

    def random_masking(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor,
        mask_ratio: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, _ = x.shape
        device = x.device
        vis_tokens, vis_idx_list, mask_idx_list = [], [], []
        for b in range(B):
            idx_valid = torch.arange(N, device=device)[valid_mask[b]]
            nv = idx_valid.numel()
            if nv == 0:
                idx_valid = torch.tensor([0], device=device, dtype=torch.long)
                nv = 1
            nv_keep, nm = self._ensure_mask_counts(nv, mask_ratio)
            noise = torch.rand(nv, device=device)
            order = torch.argsort(noise)
            keep = idx_valid[order[:nv_keep]]
            mask = idx_valid[order[nv_keep:nv_keep + nm]]
            vis_tokens.append(x[b, keep])
            vis_idx_list.append(keep)
            mask_idx_list.append(mask)

        vis_tokens_pad = torch.nn.utils.rnn.pad_sequence(vis_tokens,
                                                         batch_first=True)
        vis_idx_pad = torch.nn.utils.rnn.pad_sequence(
            vis_idx_list, batch_first=True, padding_value=-1)
        mask_idx_pad = torch.nn.utils.rnn.pad_sequence(
            mask_idx_list, batch_first=True, padding_value=-1)
        return vis_tokens_pad, vis_idx_pad, mask_idx_pad

    def _gather_positional(self,
                           idx: torch.Tensor,
                           *,
                           pos_table_1d: Optional[torch.Tensor],
                           coords_b: Optional[torch.Tensor],
                           coord_stats: Optional[Tuple[torch.Tensor,
                                                       torch.Tensor]]) -> Optional[torch.Tensor]:
        if not self.use_pos_embed or idx.numel() == 0:
            return None
        if self.pos_embed_type == '1d':
            assert pos_table_1d is not None
            return pos_table_1d[idx]
        if self.pos_embed_type == '2d_fourier':
            if coords_b is None:
                raise ValueError(
                    '2D Fourier positional encoding requires coords.')
            min_val, max_val = coord_stats if coord_stats is not None else (None,
                                                                            None)
            return fourier_2d_pos(coords_b[idx],
                                  dim=self.d_model,
                                  num_bands=self.fourier_num_bands,
                                  max_freq=self.fourier_max_freq,
                                  min_val=min_val,
                                  max_val=max_val)
        return None

    def forward(self,
                x_in: torch.Tensor,
                valid_mask: torch.Tensor,
                mask_ratio: float,
                *,
                coords: Optional[torch.Tensor] = None,
                need_attn: bool = True) -> Dict[str, torch.Tensor]:
        B, N, _ = x_in.shape
        device = x_in.device
        proj_dtype = self.in_proj.weight.dtype
        x = self.in_proj(x_in.to(proj_dtype))

        vis_tokens, vis_idx, mask_idx = self.random_masking(
            x, valid_mask, mask_ratio)
        key_padding_mask = vis_idx.eq(-1)

        pos_table = None
        coord_stats: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
        if self.use_pos_embed:
            if self.pos_embed_type == '1d':
                pos_table = self._build_position_table(valid_mask, device)
            elif self.pos_embed_type == '2d_fourier':
                if coords is None:
                    raise ValueError(
                        'coords must be provided for 2D Fourier positional encoding.'
                    )
                coord_stats = []
                for b in range(B):
                    valid_coords = coords[b][valid_mask[b]]
                    if valid_coords.numel() == 0:
                        coord_stats.append((None, None))
                    else:
                        min_xy = valid_coords.amin(dim=0, keepdim=True)
                        max_xy = valid_coords.amax(dim=0, keepdim=True)
                        coord_stats.append((min_xy, max_xy))

        k_pos = None
        if self.use_pos_embed:
            k_pos = torch.zeros_like(vis_tokens)
            for b in range(B):
                idx = vis_idx[b]
                valid_idx = idx[idx >= 0]
                if valid_idx.numel() == 0:
                    continue
                pos_embed = self._gather_positional(
                    valid_idx,
                    pos_table_1d=pos_table,
                    coords_b=coords[b] if coords is not None else None,
                    coord_stats=coord_stats[b] if coord_stats is not None else
                    None)
                if pos_embed is not None:
                    k_pos[b, :valid_idx.numel()] = pos_embed.to(k_pos.dtype)
                # else: leave zeros for this sample; do not break the batch

        latents, attn = self.resampler(vis_tokens,
                                       k_pos=k_pos,
                                       key_padding_mask=key_padding_mask,
                                       need_attn=need_attn)

        mask_idx = mask_idx.clone()
        Nm_max = mask_idx.size(1)
        if Nm_max == 0:
            mask_idx = mask_idx.new_full((B, 1), -1)
            Nm_max = 1
        mask_queries = torch.zeros(B,
                                   Nm_max,
                                   self.d_model,
                                   device=device,
                                   dtype=x.dtype)
        mask_pos = torch.zeros_like(mask_queries) if self.use_pos_embed else None
        target = torch.zeros(B, Nm_max, self.d_in, device=device, dtype=x_in.dtype)

        for b in range(B):
            idx_row = mask_idx[b]
            valid_idx = idx_row[idx_row >= 0]
            if valid_idx.numel() == 0:
                fallback = vis_idx[b][0]
                if fallback < 0:
                    fallback = torch.tensor(0, device=device, dtype=torch.long)
                idx_row[0] = int(fallback)
                valid_idx = idx_row[idx_row >= 0]

            q = self.mask_token.expand(1, valid_idx.numel(), -1).clone()
            mask_queries[b, :valid_idx.numel()] = q

            if self.use_pos_embed:
                pos_embed = self._gather_positional(
                    valid_idx,
                    pos_table_1d=pos_table,
                    coords_b=coords[b] if coords is not None else None,
                    coord_stats=coord_stats[b] if coord_stats is not None else
                    None)
                if pos_embed is not None:
                    mask_pos[b, :valid_idx.numel()] = pos_embed.to(mask_pos.dtype)

            target[b, :valid_idx.numel()] = x_in[b, valid_idx]

        if self.use_pos_embed and mask_pos is not None:
            mask_queries = mask_queries + mask_pos

        y = mask_queries
        for layer in self.decoder_layers:
            y = layer(y, latents)
        pred = self.dec_out(y)

        return {
            'pred': pred,
            'target': target,
            'latents': latents,
            'attn': attn,
            'vis_idx': vis_idx,
            'mask_idx': mask_idx,
            'key_padding_mask': key_padding_mask,
        }


@dataclass
class RegularizerItem:
    weight: float = 0.0
    enable: bool = False
    log_only: bool = False

    def should_compute(self) -> bool:
        return self.enable or self.log_only

    def should_apply(self) -> bool:
        return self.enable and not self.log_only and abs(self.weight) > 0.0


def _reg_item_default(weight: float) -> RegularizerItem:
    return RegularizerItem(weight=weight, enable=True, log_only=False)


@dataclass
class RegularizerConfig:
    attn_entropy: RegularizerItem = field(
        default_factory=lambda: _reg_item_default(1e-3))
    attn_decor: RegularizerItem = field(
        default_factory=lambda: _reg_item_default(1e-2))
    feature_decor: RegularizerItem = field(
        default_factory=lambda: _reg_item_default(1e-2))
    peak_var: RegularizerItem = field(
        default_factory=lambda: _reg_item_default(1e-3))
    coverage_stats_enable: bool = True
    cover_thresholds: Tuple[float, ...] = (1e-4, 1e-3, 1e-2)
    cover: RegularizerItem = field(
        default_factory=lambda: RegularizerItem(weight=5e-4,
                                                enable=False,
                                                log_only=False))
    cover_tau: float = 1e-4
    cover_tau_alpha: float = 0.2
    cover_tau_cap: float = 1e-3
    cover_tau_floor: float = 1e-6

    def requires_attention(self) -> bool:
        return (self.attn_entropy.should_compute()
                or self.attn_decor.should_compute()
                or self.peak_var.should_compute()
                or self.cover.should_compute()
                or self.coverage_stats_enable)


def _update_reg_item(item: RegularizerItem, key: str, value) -> None:
    if key == 'weight':
        item.weight = float(value)
    elif key == 'enable':
        item.enable = bool(value)
    elif key == 'log_only':
        item.log_only = bool(value)


def parse_regularizer_config(
        reg_cfg: Union[None, Dict, RegularizerConfig]) -> RegularizerConfig:
    if reg_cfg is None:
        return RegularizerConfig()
    if isinstance(reg_cfg, RegularizerConfig):
        return reg_cfg
    if not isinstance(reg_cfg, dict):
        raise TypeError(f'Unsupported reg_cfg type: {type(reg_cfg)}')

    cfg = RegularizerConfig()
    cfg_dict = dict(reg_cfg)

    if 'coverage_stats_enable' in cfg_dict:
        cfg.coverage_stats_enable = bool(cfg_dict.pop('coverage_stats_enable'))

    if 'cover_thresholds' in cfg_dict:
        thresholds = cfg_dict.pop('cover_thresholds')
        if isinstance(thresholds, (list, tuple)):
            cfg.cover_thresholds = tuple(float(t) for t in thresholds)
        else:
            raise TypeError('cover_thresholds must be a sequence.')
    if 'cover_tau' in cfg_dict:
        cfg.cover_tau = float(cfg_dict.pop('cover_tau'))
    if 'cover_tau_alpha' in cfg_dict:
        cfg.cover_tau_alpha = float(cfg_dict.pop('cover_tau_alpha'))
    if 'cover_tau_cap' in cfg_dict:
        cfg.cover_tau_cap = float(cfg_dict.pop('cover_tau_cap'))
    if 'cover_tau_floor' in cfg_dict:
        cfg.cover_tau_floor = float(cfg_dict.pop('cover_tau_floor'))

    item_map = {
        'attn_entropy': cfg.attn_entropy,
        'attn_decor': cfg.attn_decor,
        'feature_decor': cfg.feature_decor,
        'cover': cfg.cover,
        'peak_var': cfg.peak_var,
    }

    for name, item in item_map.items():
        nested = cfg_dict.pop(name, None)
        if isinstance(nested, dict):
            for key, value in nested.items():
                if key in {'weight', 'enable', 'log_only'}:
                    _update_reg_item(item, key, value)

    for key, value in list(cfg_dict.items()):
        for name, item in item_map.items():
            prefix = f'{name}_'
            if key.startswith(prefix):
                field_name = key[len(prefix):]
                if field_name in {'weight', 'enable', 'log_only'}:
                    _update_reg_item(item, field_name, value)
                    cfg_dict.pop(key, None)
                break

    return cfg


@BUILDER.register_module()
class MAEFeaturePretrain(BaseModel):
    """Wrapper so the MAE training loop plugs into xtuner runners."""

    def __init__(
        self,
        d_in: int,
        d_model: int,
        n_latents: int,
        num_heads: int,
        resampler_layers: int,
        resampler_topk: Union[int, Sequence[Optional[int]], None],
        use_latent_self_attn: bool,
        dec_depth: int,
        dec_mlp_ratio: float,
        resampler_dense_latents: int = 0,
        resampler_latent_drop: float = 0.0,
        mask_ratio: float = 0.75,
        reg_cfg: Optional[Dict] = None,
        use_pos_embed: bool = True,
        pos_embed_type: str = '1d',
        fourier_num_bands: int = 64,
        fourier_max_freq: float = 10000.0,
        resampler_context_mode: str = 'none',
        resampler_ctx_layers: int = 0,
        resampler_ctx_kernel_size: int = 7,
        resampler_ctx_window: int = 128,
        resampler_ctx_dilation: int = 1,
        resampler_ctx_longnet_segments: Optional[Sequence[int]] = None,
        resampler_ctx_longnet_dilations: Optional[Union[Sequence[int], Sequence[Sequence[int]]]] = None,
        resampler_ctx_longnet_use_xpos: bool = True,
        resampler_ctx_longnet_mlp_ratio: Optional[float] = None,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.reg_cfg = parse_regularizer_config(reg_cfg)
        self.mae = FeatureMAEWithResampler(
            d_in=d_in,
            d_model=d_model,
            n_latents=n_latents,
            num_heads=num_heads,
            resampler_layers=resampler_layers,
            resampler_topk=resampler_topk,
            use_latent_self_attn=use_latent_self_attn,
            dec_depth=dec_depth,
            dec_mlp_ratio=dec_mlp_ratio,
            dense_latents=resampler_dense_latents,
            latent_drop=resampler_latent_drop,
            use_pos_embed=use_pos_embed,
            pos_embed_type=pos_embed_type,
            fourier_num_bands=fourier_num_bands,
            fourier_max_freq=fourier_max_freq,
            context_mode=resampler_context_mode,
            ctx_layers=resampler_ctx_layers,
            ctx_kernel_size=resampler_ctx_kernel_size,
            ctx_window=resampler_ctx_window,
            ctx_dilation=resampler_ctx_dilation,
            ctx_longnet_segments=resampler_ctx_longnet_segments,
            ctx_longnet_dilations=resampler_ctx_longnet_dilations,
            ctx_longnet_use_xpos=resampler_ctx_longnet_use_xpos,
            ctx_longnet_mlp_ratio=resampler_ctx_longnet_mlp_ratio,
        )

    def loss(self, data: Dict, data_samples=None) -> Dict[str, torch.Tensor]:
        tokens = data['tokens']
        valid_mask = data['valid_mask']
        coords = data.get('coords')
        need_attn = self.reg_cfg.requires_attention()
        outputs = self.mae(tokens,
                           valid_mask,
                           self.mask_ratio,
                           coords=coords,
                           need_attn=need_attn)
        pred = outputs['pred']
        target = outputs['target']
        if pred.dtype != target.dtype:
            pred = pred.to(target.dtype)
        latents = outputs['latents']
        attn = outputs.get('attn')

        mask_pad = outputs['mask_idx'].eq(-1)
        if mask_pad.any():
            valid = ~mask_pad
            loss_rec = F.mse_loss(pred[valid], target[valid])
        else:
            loss_rec = F.mse_loss(pred, target)

        zero = pred.new_tensor(0.0)
        p_masked: Optional[torch.Tensor] = None
        if need_attn:
            if attn is None:
                raise RuntimeError(
                    'Attention weights are required but were not returned.')
            attn_mean = attn.mean(dim=1)
            kpm = (~outputs['key_padding_mask']).unsqueeze(1).float()
            p = attn_mean * kpm
            denom = p.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            p_masked = p / denom

        loss = loss_rec
        k_est_value: Optional[float] = None
        tau_used_value: Optional[float] = None
        tau_for_loss: Optional[float] = None
        if p_masked is not None:
            with torch.no_grad():
                k_est_tensor = (p_masked > 1e-12).sum(dim=-1).float().mean()
                k_est_value = float(k_est_tensor.item())
                tau_alpha = float(getattr(self.reg_cfg, 'cover_tau_alpha', 0.2))
                tau_cap = float(getattr(self.reg_cfg, 'cover_tau_cap', 1e-3))
                tau_floor = float(getattr(self.reg_cfg, 'cover_tau_floor', 1e-6))
                tau_cfg = float(getattr(self.reg_cfg, 'cover_tau', 1e-4))
                tau_auto = min(tau_cap, tau_alpha / max(1.0, k_est_value))
                tau_used_value = max(tau_floor, max(tau_cfg, tau_auto))
                tau_for_loss = tau_used_value
        if tau_for_loss is None:
            tau_for_loss = float(self.reg_cfg.cover_tau)

        loss_cover = zero
        if self.reg_cfg.cover.should_compute():
            if p_masked is None:
                raise RuntimeError('cover regularizer requires attention probabilities.')
            u = p_masked.amax(dim=1)
            loss_cover = F.relu(tau_for_loss - u).mean()
            if self.reg_cfg.cover.should_apply():
                loss = loss + self.reg_cfg.cover.weight * loss_cover

        loss_entropy = zero
        if self.reg_cfg.attn_entropy.should_compute():
            if p_masked is None:
                raise RuntimeError('attn_entropy regularizer requires attention probabilities.')
            loss_entropy = entropy_from_probs(p_masked).mean()
            if self.reg_cfg.attn_entropy.should_apply():
                loss = loss + self.reg_cfg.attn_entropy.weight * loss_entropy

        loss_attn_decor = zero
        if self.reg_cfg.attn_decor.should_compute():
            if p_masked is None:
                raise RuntimeError('attn_decor regularizer requires attention probabilities.')
            loss_attn_decor = attention_decorrelation_loss(p_masked)
            if self.reg_cfg.attn_decor.should_apply():
                loss = loss + self.reg_cfg.attn_decor.weight * loss_attn_decor

        loss_feat_decor = zero
        if self.reg_cfg.feature_decor.should_compute():
            loss_feat_decor = feature_covariance_decorrelation(latents.float())
            if self.reg_cfg.feature_decor.should_apply():
                loss = loss + self.reg_cfg.feature_decor.weight * loss_feat_decor

        loss_peak_var = zero
        if self.reg_cfg.peak_var.should_compute():
            if p_masked is None:
                raise RuntimeError('peak_var regularizer requires attention probabilities.')
            loss_peak_var = per_query_peakiness_variance(p_masked)
            if self.reg_cfg.peak_var.should_apply():
                loss = loss + self.reg_cfg.peak_var.weight * loss_peak_var

        stats: Dict[float, float] = {}
        if self.reg_cfg.coverage_stats_enable and p_masked is not None:
            stats = collect_stats(p_masked.detach(),
                      list(self.reg_cfg.cover_thresholds),
                      valid_mask=(~outputs['key_padding_mask']))
        
        union_stats: Dict[str, float] = {}
        if self.reg_cfg.coverage_stats_enable and p_masked is not None:
            u = p_masked.detach().amax(dim=1)
            valid_counts = (
                (~outputs['key_padding_mask']).sum(dim=-1).clamp_min(1).float())
            for tau in self.reg_cfg.cover_thresholds:
                tau_val = float(tau)
                covered = (u > tau_val).float().sum(dim=-1) / valid_counts
                union_stats[f'{tau_val:.0e}'] = float(covered.mean().item())
            if tau_used_value is not None:
                tau_val = float(tau_used_value)
                covered = (u > tau_val).float().sum(dim=-1) / valid_counts
                union_stats['auto'] = float(covered.mean().item())

        pair_cos_value: Optional[float] = None
        with torch.no_grad():
            if latents.dim() == 3 and latents.size(1) > 1:
                z = latents
                zf = z / (z.norm(dim=-1, keepdim=True) + 1e-6)
                cos = torch.einsum('bqd,bkd->bqk', zf, zf)
                B, Q, _ = cos.shape
                off_mask = torch.ones_like(cos, dtype=torch.bool)
                idx = torch.arange(Q, device=cos.device)
                off_mask[:, idx, idx] = False
                if off_mask.any():
                    pair_cos = cos[off_mask].mean()
                    pair_cos_value = float(pair_cos.item())
            elif latents.numel() > 0:
                pair_cos_value = 0.0

        losses = {
            'loss': loss,
            'loss_rec': loss_rec.detach(),
            'attn_entropy': loss_entropy.detach(),
            'attn_decor': loss_attn_decor.detach(),
            'feat_decor': loss_feat_decor.detach(),
            'peak_var': loss_peak_var.detach(),
            'cover': loss_cover.detach(),
        }
        if 'key_padding_mask' in outputs:
            n_vis_avg = (~outputs['key_padding_mask']).sum(dim=-1).float().mean().item()
            losses['n_vis_avg'] = latents.new_tensor(n_vis_avg)
        
        for tau, value in stats.items():
            losses[f'coverage_{tau:.0e}'] = latents.new_tensor(value)
        for name, value in union_stats.items():
            losses[f'union_cov_{name}'] = latents.new_tensor(value)
        if k_est_value is not None:
            losses['k_est'] = latents.new_tensor(k_est_value)
        if tau_used_value is not None:
            losses['cover_tau_used'] = latents.new_tensor(tau_used_value)
        if pair_cos_value is not None:
            losses['pair_cos'] = latents.new_tensor(pair_cos_value)
        return losses

    def forward(self,
                data=None,
                data_samples=None,
                mode: str = 'loss',
                *,
                tokens=None,
                valid_mask=None,
                coords=None,
                **kwargs):
        if data is None:
            data_dict = {}
        else:
            data_dict = dict(data)
        if tokens is not None:
            data_dict['tokens'] = tokens
        if valid_mask is not None:
            data_dict['valid_mask'] = valid_mask
        if coords is not None:
            data_dict['coords'] = coords
        if kwargs:
            data_dict.update(kwargs)
        if 'tokens' not in data_dict or 'valid_mask' not in data_dict:
            raise TypeError('Expected tokens and valid_mask in forward inputs.')
        if mode == 'loss':
            return self.loss(data_dict, data_samples)
        if mode == 'predict':
            with torch.no_grad():
                outputs = self.mae(data_dict['tokens'],
                                   data_dict['valid_mask'],
                                   self.mask_ratio,
                                   coords=data_dict.get('coords'),
                                   need_attn=False)
            return outputs
        raise NotImplementedError(f'Unsupported mode: {mode}')
