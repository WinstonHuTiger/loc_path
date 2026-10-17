import math
import os
import os.path as osp
from collections import OrderedDict
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.config import Config, ConfigDict
from mmengine.model import BaseModel
from mmengine import print_log
from peft import get_peft_model, prepare_model_for_kbit_training
from safetensors.torch import load_file, save_file
from transformers import AutoConfig
from transformers.integrations import is_deepspeed_zero3_enabled

from xtuner.registry import BUILDER
from .modules import dispatch_modules
from .modules.qwen_vis_adapter import (TopMRouter, TextCondTopMRouter, VisualProjector,
                                       VisionAdapterController,
                                       patch_qwen_for_cross_attn)
from .modules.feature_resampler import (PerceiverResamplerTopK, fourier_2d_pos)
from .utils import (IMAGE_TOKEN_INDEX, IGNORE_INDEX, LoadWoInit,
                    find_all_linear_names, get_peft_model_state_dict,
                    guess_load_checkpoint, prepare_inputs_labels_for_multimodal,
                    traverse_dict)
from .sparse_token_merge import SparsePatchMerging

# --- safety helper: align teacher distribution to student top-K space ---
def _align_teacher_probs(teacher: torch.Tensor,
                         router_scores: torch.Tensor,
                         router_indices: torch.Tensor,
                         tau_t: float) -> torch.Tensor | None:
    """
    Align Teacher probs to the same event space as student:
      - If teacher is [B, K] (already routed order): use as-is
      - If teacher is [B, N] (pre-routing): gather by router_indices -> [B, K]
      - Else: return None (skip aux loss for safety)
    Then apply optional temperature by power transform, and renormalize.
    """
    if teacher is None or router_indices is None:
        return None
    B, N = router_scores.shape
    B2, L = teacher.shape
    if B2 != B:
        return None
    K = router_indices.size(1)

    # Case 1: teacher already on routed tokens [B, K]
    if L == K:
        t = teacher.to(torch.float32)
    # Case 2: teacher on full tokens [B, N] -> gather to [B, K]
    elif L == N:
        # indices must be within [0, N-1]
        if router_indices.max() >= N or router_indices.min() < 0:
            return None
        t = torch.gather(teacher.to(torch.float32), 1, router_indices)
    else:
        # Unexpected teacher length; skip safely
        return None

    # ensure strictly positive, apply temperature, and renormalize
    t = t.clamp_min(1e-12)
    if tau_t != 1.0:
        # power transform equals softmax(log(p)/tau)
        t = t.pow(1.0 / tau_t)
    t = t / t.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return t


def _feature_covariance_decorrelation(z: torch.Tensor) -> torch.Tensor:
    """Feature decorrelation regularizer used during resampler pretraining."""
    if z.ndim != 3 or z.numel() == 0:
        return z.new_zeros(())
    bq, d = z.shape[0] * z.shape[1], z.shape[2]
    if bq <= 1:
        return z.new_zeros(())
    z_flat = z.reshape(bq, d)
    z_centered = z_flat - z_flat.mean(dim=0, keepdim=True)
    denom = max(z_centered.size(0) - 1, 1)
    cov = torch.matmul(z_centered.T, z_centered) / denom
    off_diag = cov - torch.diag(torch.diagonal(cov))
    return off_diag.pow(2).sum() / (d * d)

class LlavaVisAdapterModel(BaseModel):
    """LLaVA variant that uses sparse merger + MAE resampler + Qwen visual adapter."""

    def __init__(
        self,
        llm,
        freeze_llm: bool = True,
        llm_lora: Optional[dict] = None,
        train_stage: str = '2',
        router_temperature_s: float = 1.0,
        router_temperature_t: float = 1.0,
        router_diag: bool = True,
        max_position_embeddings: Optional[int] = None,
        visual_dim: int = 512,
        merger_cfg: Optional[dict] = None,
        resampler_cfg: Optional[dict] = None,
        resampler_ckpt: Optional[str] = None,
        pretrained_pth: Optional[str] = None,
        projector_depth: int = 2,
        projector_dropout: float = 0.0,
        router_top_m: Optional[int] = None,
        router_jitter: int = 0,
        router_aux_weight: float = 0.0,
        router_fuse: str = 'add',
        router_hidden: int = 1024,
        router_dropout: float = 0.0,
        router_use_text: bool = True,
        router_enable_capture_mass: bool = True,
        adapter_layer_indices: Sequence[int] = (0, 2, 4, 6),
        adapter_dropout: float = 0.0,
        use_activation_checkpointing: bool = True,
        vision_to_llm_scale: float = 1.0,
        use_token_merger: bool = True,
        token_merger_cfg: Optional[dict] = None,
        freeze_resampler: bool = False,
        tune_resampler_last_latents_inproj: bool = False,
        # Optional: if None, disable sequence insertion when adapters are present.
        insert_visual_into_seq: Optional[bool] = None,
        # cross-attn adapter teacher pooling controls
        teacher_pool: str = "topk",
        teacher_topk_frac: float = 0.1,
        coord_pe_cfg: Optional[dict] = None,
        coord_debug: bool = False,
        feat_decor_weight: float = 0.0,
        
        router_rank_lambda: float = -1,     # 新增：排序损失相对 KL 的权重
        router_rank_pairs: int = 8,          # 新增：每个样本抽多少对 (i,j)
        router_rank_margin: float = 0.0,     # 新增：pairwise hinge 的 margin
    ):
        super().__init__()
        self.freeze_llm = freeze_llm
        self.train_stage = train_stage
        self.use_activation_checkpointing = use_activation_checkpointing
        self.vision_to_llm_scale = vision_to_llm_scale
        self.use_token_merger = use_token_merger
        self.router_aux_weight = router_aux_weight
        self.freeze_resampler = freeze_resampler
        self.tune_resampler_last_latents_inproj = bool(tune_resampler_last_latents_inproj)
        self.insert_visual_into_seq = insert_visual_into_seq
        self.router_temperature_s = float(router_temperature_s)
        self.router_temperature_t = float(router_temperature_t)
        self.router_diag = bool(router_diag)
        self.router_use_text = bool(router_use_text)
        self.router_enable_capture_mass = bool(router_enable_capture_mass)
        self.router_jitter = int(router_jitter)
        self._last_router_state: Optional[Dict[str, torch.Tensor]] = None
        self._last_label_check: Optional[dict] = None
        self._resampler_teacher: Optional[torch.Tensor] = None
        self._last_latents: Optional[torch.Tensor] = None
        self._last_coords_state: Optional[Dict[str, torch.Tensor]] = None
        self.feat_decor_weight = float(feat_decor_weight)
        self._feature_dim_warned = False
        
        self.router_rank_lambda = float(router_rank_lambda)
        self.router_rank_pairs  = int(router_rank_pairs)
        self.router_rank_margin = float(router_rank_margin)

        with LoadWoInit():
            if isinstance(llm, dict):
                llm = self._dispatch_lm_model_cfg(llm, max_position_embeddings)
            self.llm = self._build_from_cfg_or_module(llm)

        self.llm.config.use_cache = False
        dispatch_modules(self.llm)

        if freeze_llm:
            self.llm.requires_grad_(False)

        self.use_llm_lora = llm_lora is not None
        if self.use_llm_lora:
            self._prepare_llm_for_lora(llm_lora, use_activation_checkpointing)

        self.visual_dim = visual_dim
        self.token_merger = None
        if self.use_token_merger:
            merger_cfg = merger_cfg or {}
            tm_cfg = token_merger_cfg or {}
            self.token_merger = SparsePatchMerging(
                embed_dim=visual_dim,
                layernorm_eps=tm_cfg.get('layernorm_eps', 1e-6),
                merge_size=tm_cfg.get('merge_size', 2),
                mode=tm_cfg.get('mode', merger_cfg.get('mode', 'sum')),
                pre_norm=tm_cfg.get('pre_norm', True),
                post_norm=tm_cfg.get('post_norm', True),
                fp32_reduce=tm_cfg.get('fp32_reduce', True),
                clear_after_forward=tm_cfg.get('clear_after_forward', False),
                clear_on_oom=tm_cfg.get('clear_on_oom', True),
                
                # make sure that we use c first
                coord_order = 'cr'
            )
            self.token_merger = self.token_merger.to(self.llm.dtype)

        resampler_cfg = resampler_cfg or {}
        resampler_dim = resampler_cfg.get('d_model', visual_dim)
        resampler_kwargs = dict(
            d_model=resampler_dim,
            n_latents=resampler_cfg.get('n_latents', 256),
            num_heads=resampler_cfg.get('num_heads', self.llm.config.num_attention_heads),
            n_layers=resampler_cfg.get('n_layers', 2),
            topk=resampler_cfg.get('topk'),
            mlp_ratio=resampler_cfg.get('mlp_ratio', 4.0),
            use_latent_self_attn=resampler_cfg.get('use_latent_self_attn', True),
            attn_drop=resampler_cfg.get('attn_drop', 0.0),
            proj_drop=resampler_cfg.get('proj_drop', 0.0),
            context_mode=resampler_cfg.get('context_mode', 'none'),
            
            
            ctx_layers=resampler_cfg.get('ctx_layers', 0),
            ctx_kernel_size=resampler_cfg.get('ctx_kernel_size', 7),
            ctx_window=resampler_cfg.get('ctx_window', 128),
            ctx_dilation=resampler_cfg.get('ctx_dilation', 1),
        
            ctx_longnet_mlp_ratio = resampler_cfg.get('ctx_longnet_mlp_ratio', 2.0),
            ctx_longnet_use_xpos = resampler_cfg.get('ctx_longnet_use_xpos', True),
            ctx_longnet_segments = resampler_cfg.get('ctx_longnet_segments', [1024, 2048, 4096]),
            ctx_longnet_dilations = resampler_cfg.get('ctx_longnet_dilations', [[1, 2, 4], [4, 8, 16]])
        )
        self.resampler = PerceiverResamplerTopK(**resampler_kwargs)
        if self.freeze_resampler:
            self.resampler.requires_grad_(False)

        self.feature_in_proj = nn.Linear(visual_dim, resampler_dim)

        self.feature_in_proj = self.feature_in_proj.to(self.llm.dtype)
        if self.freeze_resampler:
            self.feature_in_proj.requires_grad_(False)

        self.projector = VisualProjector(
            dim_in=resampler_dim,
            dim_out=self.llm.config.hidden_size,
            depth=projector_depth,
            dropout=projector_dropout,
        ).to(self.llm.dtype)

        self.router_top_m = router_top_m
        self.router = None
        if router_top_m is not None:
            if self.router_use_text:
                # Use text-conditioned MLP router by default
                self.router = TextCondTopMRouter(
                    dim=self.llm.config.hidden_size,
                    temperature=1.0,
                    hidden=router_hidden,
                    dropout=router_dropout,
                    fuse=router_fuse,
                )
            else:
                self.router = TopMRouter(
                    dim=self.llm.config.hidden_size,
                    temperature=1.0,
                )
            self.router = self.router.to(self.llm.dtype)

        self.controller = VisionAdapterController(
            self.llm.config.hidden_size,
            self.llm.config.num_attention_heads,
            layer_indices=adapter_layer_indices,
            dropout=adapter_dropout,
            teacher_pool=teacher_pool,
            teacher_topk_frac=teacher_topk_frac,
        )
        self.controller = self.controller.to(self.llm.dtype)
        patch_qwen_for_cross_attn(self.llm, self.controller)

        if self.insert_visual_into_seq is None:
            self.insert_visual_into_seq = (len(adapter_layer_indices) == 0)

        coord_pe_cfg = coord_pe_cfg or {}
        self.coord_pe_num_bands = coord_pe_cfg.get('num_bands', 64)
        self.coord_pe_max_freq = coord_pe_cfg.get('max_freq', 10000.0)
        self.coord_pe_eps = coord_pe_cfg.get('eps', 1e-6)
        self.coord_debug = bool(coord_pe_cfg.get('debug', coord_debug))

        
        if train_stage == "1":
            # if self.router is not None and router_top_m is not None:
            print_log("router is trainable", "current")
            if self.router is not None:
                self.router.requires_grad_(True)
            self.projector.requires_grad_(True)
            
            # keep adapters trainable too
            if hasattr(self.controller, 'adapters'):
                self.controller.adapters.requires_grad_(True)
        elif train_stage == "2":
            # if self.router is not None and router_top_m is not None:
            print_log("router is trainable", "current")
            if self.router is not None:
                self.router.requires_grad_(True)
            self.projector.requires_grad_(True)
            # keep adapters trainable too
            if hasattr(self.controller, 'adapters'):
                self.controller.adapters.requires_grad_(True)
            
        
        if resampler_ckpt:
            self._load_resampler_checkpoint(resampler_ckpt)

        
        if self.freeze_resampler and self.tune_resampler_last_latents_inproj:
            # Re-enable gradients for the last resampler layer + learnable latents + input projection
            if hasattr(self.resampler, 'query') and isinstance(self.resampler.query, nn.Parameter):
                self.resampler.query.requires_grad = True
            last_modules = []
            if hasattr(self.resampler, 'attn_blocks') and len(self.resampler.attn_blocks) > 0:
                last_modules.append(self.resampler.attn_blocks[-1])
            if hasattr(self.resampler, 'ffn') and len(self.resampler.ffn) > 0:
                last_modules.append(self.resampler.ffn[-1])
            if hasattr(self.resampler, 'ln_q') and len(self.resampler.ln_q) > 0:
                last_modules.append(self.resampler.ln_q[-1])
            if hasattr(self.resampler, 'ln_ffn') and len(self.resampler.ln_ffn) > 0:
                last_modules.append(self.resampler.ln_ffn[-1])
            if hasattr(self.resampler, 'latent_blocks') and self.resampler.latent_blocks:
                last_modules.append(self.resampler.latent_blocks[-1])
            for module in last_modules:
                module.requires_grad_(True)
            self.feature_in_proj.requires_grad_(True)
            print_log('Fine-tuning resampler last layer, latent query, and feature_in_proj.', 'current')

        # Optionally load a previously saved/finetuned checkpoint (pth/safetensors)
        if pretrained_pth is not None:
            try:
                sd = guess_load_checkpoint(pretrained_pth)
                current = super().state_dict()  # use full module keys for filtering
                filtered = {k: v for k, v in sd.items()
                            if k in current and current[k].shape == v.shape}
                missing, unexpected = self.load_state_dict(filtered, strict=False)
                print_log(f'Loaded pretrained checkpoint from {pretrained_pth}', 'current')
                if missing:
                    print_log(f'  missing keys: {list(missing)[:8]}... (total {len(missing)})', 'current')
                if unexpected:
                    print_log(f'  unexpected keys: {list(unexpected)[:8]}... (total {len(unexpected)})', 'current')
            except Exception as exc:
                raise RuntimeError(f'Failed to load pretrained_pth: {pretrained_pth}') from exc

        need_visual_grad = (
            any(p.requires_grad for p in self.feature_in_proj.parameters())
            or any(p.requires_grad for p in self.resampler.parameters())
            or any(p.requires_grad for p in self.projector.parameters())
            or (self.router_top_m is not None)
        )
        if hasattr(self.llm, 'gradient_checkpointing_enable'):
            if use_activation_checkpointing and not need_visual_grad:
                self.llm.gradient_checkpointing_enable()
                self.use_activation_checkpointing = True
            else:
                if hasattr(self.llm, 'gradient_checkpointing_disable'):
                    self.llm.gradient_checkpointing_disable()
                self.use_activation_checkpointing = False
                print_log(
                    'Gradient checkpointing disabled: visual tokens with grad are reused across layers; '
                    'this avoids "backward through the graph a second time".',
                    'current')

        self.is_first_iter = True
        self._visual_pad_value = None

    # mmengine calls this before training; the HF backbone already carries
    # initialised weights, so skip the default dtype-sensitive routine.
    def init_weights(self):
        return

    # ---------------------------------------------------------------------- #
    # Helper utilities (mostly adapted from llava_no_longnet)
    # ---------------------------------------------------------------------- #
    @staticmethod
    def _prepare_for_long_context_training(cfg, llm_cfg, max_position_embeddings):
        orig_rope_scaling = getattr(llm_cfg, 'rope_scaling', None)
        if orig_rope_scaling is None:
            orig_rope_scaling = {'factor': 1}

        orig_rope_scaling_factor = orig_rope_scaling.get('factor', 1)
        orig_ctx_len = getattr(llm_cfg, 'max_position_embeddings', None)
        if orig_ctx_len:
            orig_ctx_len *= orig_rope_scaling_factor
            if max_position_embeddings > orig_ctx_len:
                scaling_factor = float(math.ceil(max_position_embeddings / orig_ctx_len))
                llm_cfg.rope_scaling = {'type': 'linear', 'factor': scaling_factor}

        llm_cfg.attn_implementation = 'flash_attention_2'
        cfg.config = llm_cfg
        return cfg, llm_cfg

    @staticmethod
    def _prepare_for_flash_attn(cfg, llm_cfg):
        cls_name = type(llm_cfg).__name__
        support_sdpa = ('LlamaConfig', 'GemmaConfig', 'MistralConfig', 'MixtralConfig',
                        'Qwen2Config', 'Qwen2MoeConfig', 'Starcoder2Config', 'Phi3Config')
        support_fa2 = ('InternLM2Config',) + support_sdpa

        torch_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) \
            else torch.float16

        if getattr(cfg, 'attn_implementation', None) is not None:
            if cfg.attn_implementation == 'flash_attention_2':
                cfg.torch_dtype = torch_dtype
        elif cls_name in support_fa2:
            cfg.torch_dtype = torch_dtype
            cfg.attn_implementation = 'flash_attention_2'
        elif cls_name in support_sdpa:
            cfg.attn_implementation = 'sdpa'
        return cfg, llm_cfg

    @staticmethod
    def _prepare_for_qlora_zero3(cfg):
        if (not is_deepspeed_zero3_enabled()) or (not hasattr(cfg, 'quantization_config')):
            return cfg
        torch_dtype = torch.bfloat16 if (torch.cuda.is_available()
                                         and torch.cuda.is_bf16_supported()) else torch.float16
        cfg.torch_dtype = torch_dtype
        quant_config = cfg.quantization_config
        quant_config.bnb_4bit_compute_dtype = torch_dtype
        quant_config.bnb_4bit_quant_storage = torch_dtype
        return cfg

    def _dispatch_lm_model_cfg(self, cfg, max_position_embeddings=None):
        cfg = self._prepare_for_qlora_zero3(cfg)
        pretrained = cfg.pretrained_model_name_or_path
        llm_cfg = AutoConfig.from_pretrained(pretrained, trust_remote_code=True)
        cfg, llm_cfg = self._prepare_for_flash_attn(cfg, llm_cfg)
        if max_position_embeddings is not None:
            cfg, llm_cfg = self._prepare_for_long_context_training(cfg, llm_cfg, max_position_embeddings)
        return cfg

    def _build_from_cfg_or_module(self, cfg_or_mod):
        if isinstance(cfg_or_mod, nn.Module):
            return cfg_or_mod
        if isinstance(cfg_or_mod, dict):
            traverse_dict(cfg_or_mod)
            return BUILDER.build(cfg_or_mod)
        raise TypeError(f'Unsupported llm spec type: {type(cfg_or_mod)}')

    def _parse_lora_config(self, lora_config):
        if isinstance(lora_config, (dict, Config, ConfigDict)):
            return BUILDER.build(lora_config)
        return lora_config

    def _prepare_llm_for_lora(self, lora_config, use_activation_checkpointing=True):
        lora_config = self._parse_lora_config(lora_config)
        self.llm = prepare_model_for_kbit_training(self.llm, use_activation_checkpointing)
        if lora_config.target_modules is None:
            modules = find_all_linear_names(self.llm)
            lora_config.target_modules = modules
        self.llm = get_peft_model(self.llm, lora_config)

    # ---------------------------------------------------------------------- #
    # Resampler checkpoint loading
    # ---------------------------------------------------------------------- #
    def _load_resampler_checkpoint(self, ckpt_path: str) -> None:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f'Resampler checkpoint not found: {ckpt_path}')
        try:
            if ckpt_path.endswith('.safetensors'):
                state = load_file(ckpt_path, device='cpu')
            else:
                state = guess_load_checkpoint(ckpt_path)
            state_raw = dict(state)
            print_log('loaded the checkpoint of the resampler succeed!', 'current')
        except Exception as exc:
            raise RuntimeError(f'Failed to load resampler checkpoint: {ckpt_path}') from exc

        def _strip_prefix(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
            return {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}

        if any(key.startswith('mae.resampler') for key in state.keys()):
            state = _strip_prefix(state, 'mae.resampler.')
        elif any(key.startswith('resampler.') for key in state.keys()):
            state = _strip_prefix(state, 'resampler.')
        missing, unexpected = self.resampler.load_state_dict(state, strict=False)
        print_log(f'Loaded Resampler checkpoint from {ckpt_path}', 'current')
        if missing:
            print_log(f'  missing keys: {list(missing)[:8]}... (total {len(missing)})', 'current')
        if unexpected:
            print_log(f'  unexpected keys: {list(unexpected)[:8]}... (total {len(unexpected)})', 'current')
        
        self.resampler.to(self.llm.dtype)

        if not isinstance(self.feature_in_proj, nn.Identity) and hasattr(self.feature_in_proj, 'weight'):
            def _try_load_feature_in_proj(weights_dict: Dict[str, torch.Tensor]) -> bool:
                candidates = [
                    k for k in weights_dict.keys()
                    if k.endswith('in_proj.weight')
                ]
                for weight_key in candidates:
                    weight_val = weights_dict[weight_key]
                    if weight_val.shape != self.feature_in_proj.weight.shape:
                        continue
                    bias_key = weight_key[:-6] + 'bias'
                    state_dict = {'weight': weight_val}
                    if self.feature_in_proj.bias is not None and bias_key in weights_dict:
                        bias_val = weights_dict[bias_key]
                        if bias_val.shape == self.feature_in_proj.bias.shape:
                            state_dict['bias'] = bias_val
                    try:
                        missing_fp, unexpected_fp = self.feature_in_proj.load_state_dict(state_dict, strict=False)
                        if missing_fp or unexpected_fp:
                            print_log(
                                f'feature_in_proj load had missing={missing_fp}, unexpected={unexpected_fp}',
                                'current')
                        return True
                    except Exception:
                        continue
                return False

            loaded_feature_proj = _try_load_feature_in_proj(state_raw)
            if not loaded_feature_proj:
                loaded_feature_proj = _try_load_feature_in_proj({k.split('.', 1)[-1]: v for k, v in state_raw.items()})
            if loaded_feature_proj:
                print_log('Loaded feature_in_proj from resampler checkpoint.', 'current')
                self.feature_in_proj.to(self.llm.dtype)
                if self.freeze_resampler:
                    self.feature_in_proj.requires_grad_(False)
            else:
                print_log('feature_in_proj weights not found in resampler checkpoint; using existing init.', 'current')

    # ---------------------------------------------------------------------- #
    # Forward / loss / predict
    # ---------------------------------------------------------------------- #
    def _prepare_multimodal_inputs(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Always clear controller state before a new forward; safe with gradient checkpointing.
        self.controller.clear()
        self.controller.collect_attn = False
        self._last_router_state = None
        self._resampler_teacher = None
        self._last_latents = None
        self._last_coords_state = None
        pixel_values = batch.get('pixel_values')
        pad_mask = batch.get('vision_pad_mask')
        coords = batch.get('coords')
        if self.freeze_llm and batch.get('pixel_values') is None:
            # You can either raise a friendly error or keep the zero-anchor fallback.
            # raise RuntimeError('freeze_llm=True but batch has no visual features; '
            #                    'filter text-only samples in epoch 0.')
            pass

        visual_tokens = None
        if pixel_values is not None:
            # print('llm dtype', self.llm.dtype)
            feats = pixel_values.to(self.llm.dtype)
            expected_dim = int(self.visual_dim)
            current_dim = feats.size(-1)
            if current_dim != expected_dim:
                if current_dim < expected_dim:
                    pad = expected_dim - current_dim
                    feats = F.pad(feats, (0, pad))
                    action = f'zero-padded last dim by {pad}'
                else:
                    feats = feats[..., :expected_dim]
                    action = f'truncated last dim from {current_dim} to {expected_dim}'
                if not self._feature_dim_warned:
                    print_log(
                        f'Adjusted visual feature dimension in _prepare_multimodal_inputs: '
                        f'expected {expected_dim}, got {current_dim}; {action}.',
                        logger='current')
                    self._feature_dim_warned = True
            mask = None
            if pad_mask is not None:
                mask = pad_mask.to(dtype=torch.bool, device=feats.device)
            coords_rc = None

            def _compute_coord_stats(
                    coords_tensor: torch.Tensor,
                    padding_mask: Optional[torch.Tensor]
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                valid_mask = (~padding_mask) if padding_mask is not None else torch.ones_like(
                    coords_tensor[..., 0], dtype=torch.bool)
                valid_expanded = valid_mask.unsqueeze(-1)
                pos_inf = torch.full_like(coords_tensor, float('inf'))
                neg_inf = torch.full_like(coords_tensor, float('-inf'))
                mins = torch.where(valid_expanded, coords_tensor, pos_inf).amin(dim=1, keepdim=True)
                maxs = torch.where(valid_expanded, coords_tensor, neg_inf).amax(dim=1, keepdim=True)
                counts = valid_mask.sum(dim=1)
                has_valid = counts > 0
                safe_mins = torch.where(has_valid.view(-1, 1, 1), mins, torch.zeros_like(mins))
                safe_maxs = torch.where(has_valid.view(-1, 1, 1), maxs, torch.ones_like(maxs))
                return valid_mask, safe_mins, safe_maxs, counts

            def _debug_coord_stats(
                    tag: str,
                    coords_tensor: torch.Tensor,
                    padding_mask: Optional[torch.Tensor]
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                valid_mask, min_val, max_val, counts = _compute_coord_stats(coords_tensor, padding_mask)
                counts_list = counts.tolist()

                def _format(vals: torch.Tensor) -> list:
                    squeezed = vals.squeeze(1).tolist()
                    out = []
                    for cnt, item in zip(counts_list, squeezed):
                        out.append(item if cnt > 0 else None)
                    return out

                coord_min = _format(min_val)
                coord_max = _format(max_val)
                print(f"{tag} coords int-range per sample: min={coord_min} max={coord_max}  "
                      f"G={counts_list}")

                valid_expanded = valid_mask.unsqueeze(-1)
                pos_inf = torch.full_like(coords_tensor, float('inf'))
                neg_inf = torch.full_like(coords_tensor, float('-inf'))
                denom = (max_val - min_val).clamp_min(self.coord_pe_eps)
                norm_valid = torch.where(
                    valid_expanded,
                    (coords_tensor - min_val) / denom,
                    torch.zeros_like(coords_tensor))
                mins_n = torch.where(valid_expanded, norm_valid, pos_inf).amin(dim=1)
                maxs_n = torch.where(valid_expanded, norm_valid, neg_inf).amax(dim=1)

                def _format_norm(vals: torch.Tensor) -> list:
                    vals_list = vals.tolist()
                    out = []
                    for cnt, item in zip(counts_list, vals_list):
                        out.append(item if cnt > 0 else None)
                    return out

                print(f"{tag} norm-range per sample: min={_format_norm(mins_n)} "
                      f"max={_format_norm(maxs_n)}")
                return valid_mask, min_val, max_val, counts
            
            def _axis_sanity(name, coords_tensor, padmask=None, topk=3):
                vmask = (~padmask) if padmask is not None else torch.ones_like(coords_tensor[...,0], dtype=torch.bool)
                xs = []
                ys = []
                for b in range(coords_tensor.size(0)):
                    v = vmask[b]
                    if v.any():
                        x = torch.unique(coords_tensor[b, v, 0]).numel()
                        y = torch.unique(coords_tensor[b, v, 1]).numel()
                        xs.append(int(x)); ys.append(int(y))
                if xs and ys:
                    import numpy as np
                    rx = np.array(xs); ry = np.array(ys)
                    # 打印几个极端样本看下横纵向 token 数差异
                    idx = np.argsort(-(rx*ry))[:topk]
                    print(f"[{name}] unique_x/unique_y (top{topk} by area):",
                        [(xs[i], ys[i]) for i in idx])
            
             # 调试：合并前
            
            def _check_mod2(tag, coords, padmask=None):
                vmask = (~padmask) if padmask is not None else torch.ones_like(coords[...,0], dtype=torch.bool)
                v = coords[vmask]                  # [K, 2]
                if v.numel() == 0: return
                xmod = torch.bincount((v[:,0].to(torch.int32) & 1), minlength=2)
                ymod = torch.bincount((v[:,1].to(torch.int32) & 1), minlength=2)
                print(f"{tag} mod2: x={xmod.tolist()} y={ymod.tolist()}")
            
            if coords is not None:
                coords_rc = coords.to(device=feats.device)
                if self.coord_debug:
                    _debug_coord_stats("Before [TM]", coords_rc.to(torch.float32), mask)
                    _axis_sanity("Before [TM]", coords_rc, mask)  # 可选
                    _check_mod2("Before [TM]", coords_rc, mask)

            # 进入合并前，确保是整型网格坐标
            if self.token_merger is not None and coords_rc is not None:
                coords_int = coords_rc.to(torch.int32)
                merged, coords_int, mask = self.token_merger(x=feats,
                                                             coords_rc=coords_int,
                                                             padmask=mask)
                feats = merged
                coords_rc = coords_int.to(torch.float32)
            elif coords_rc is not None:
                coords_rc = coords_rc.to(torch.float32)

            if coords_rc is not None:
                self._last_coords_state = {
                    'coords': coords_rc.detach(),
                    'mask': mask.detach() if mask is not None else None
                }
            else:
                self._last_coords_state = None

            feats = self.feature_in_proj(feats)
            if feats.size(-1) != getattr(self.resampler, 'd_model', feats.size(-1)):
                raise RuntimeError(
                    f'Resampler expects feature dim {self.resampler.d_model}, '
                    f'but got {feats.size(-1)}. Check feature_in_proj.')

            # 合并后再调试 + PE
            k_pos = None
            stats_post = None
            if coords_rc is not None:
                if self.coord_debug:
                    stats_post = _debug_coord_stats("[TM]", coords_rc, mask)
                    _axis_sanity("[TM]", coords_rc.to(torch.int32), mask)  # 可选
                else:
                    stats_post = _compute_coord_stats(coords_rc, mask)

                _, min_val, max_val, _ = stats_post
                k_pos = fourier_2d_pos(
                    coords_rc, dim=feats.size(-1),
                    num_bands=self.coord_pe_num_bands,
                    max_freq=self.coord_pe_max_freq,
                    min_val=min_val, max_val=max_val,
                    eps=self.coord_pe_eps
                ).to(feats.dtype)
            need_resampler_attn = (
                self.router_enable_capture_mass
                and self.router is not None
                and self.router_top_m is not None
            )
            resampler_trainable = any(p.requires_grad for p in self.resampler.parameters())
            if resampler_trainable:
                resampled, resampler_attn = self.resampler(
                    feats,
                    k_pos=k_pos,
                    key_padding_mask=mask,
                    need_attn=need_resampler_attn)
            else:
                with torch.no_grad():
                    # print('feature dtype', feats.dtype)
                    resampled, resampler_attn = self.resampler(
                        feats,
                        k_pos=k_pos,
                        key_padding_mask=mask,
                        need_attn=need_resampler_attn)
            if need_resampler_attn and resampler_attn is not None:
                teacher = resampler_attn.to(torch.float32)
                teacher = teacher.mean(dim=(1, 2))  # [B, N]
                if mask is not None:
                    teacher = teacher.masked_fill(mask, 0.0)
                denom = teacher.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                teacher = teacher / denom
                self._resampler_teacher = teacher.detach()
            else:
                self._resampler_teacher = None
            
            if self.feat_decor_weight > 0.0:
                self._last_latents = resampled.to(torch.float32)
            resampled = resampled.to(self.llm.dtype) * self.vision_to_llm_scale
            projected = self.projector(resampled)
            collect_router_stats = (
                self.router_aux_weight > 0
                and self.training
                and self.router is not None
                and self.router_top_m is not None
            )
            # Optionally compute text-conditioned routing query
            query = None
            if (self.router is not None and self.router_top_m is not None
                    and self.router_use_text
                    and getattr(self.router, 'requires_query', False)):
                query = self._pool_question_embedding(
                    batch.get('input_ids'), batch.get('attention_mask'))

            if self.router is not None and self.router_top_m is not None:
                top_m_eff = int(self.router_top_m)
                if self.training and self.router_jitter > 0:
                    max_m = int(projected.size(1))
                    if max_m > 0:
                        delta = int(torch.randint(
                            -self.router_jitter,
                            self.router_jitter + 1,
                            (1,),
                            device=projected.device).item())
                        top_m_eff = max(1, min(max_m, top_m_eff + delta))
                mask = None
                if self.router_use_text and query is not None:
                    routed, indices, scores, mask = self.router(
                        projected, top_m=top_m_eff, query=query)
                else:
                    routed, indices, scores, mask = self.router(
                        projected, top_m=top_m_eff)
                visual_tokens = routed
                self._last_router_state = {
                    'indices': indices,
                    'scores': scores,
                    'mask': mask,
                }
            else:
                visual_tokens = projected
            self.controller.collect_attn = bool(collect_router_stats and visual_tokens is not None)
        else:
            visual_tokens = None

        if visual_tokens is not None:
            self.controller.set_visual_tokens(visual_tokens)
            if not self.insert_visual_into_seq:
                batch['pixel_values'] = None
                # print_log("是不是这里", "current")
                return self._prepare_text_only_inputs(
                    input_ids=batch.get('input_ids'),
                    position_ids=batch.get('position_ids'),
                    attention_mask=batch.get('attention_mask'),
                    past_key_values=batch.get('past_key_values'),
                    labels=batch.get('labels'),
                )
            total_sequences = []
            batch_size = visual_tokens.size(0)
            ids = batch.get('input_ids')
            masks = batch.get('attention_mask')
            for idx in range(batch_size):
                count = 1
                if ids is not None:
                    token_row = ids[idx]
                    if masks is not None:
                        token_row = token_row[masks[idx].bool()]
                    count = int((token_row == IMAGE_TOKEN_INDEX).sum().item())
                    if count <= 0:
                        count = 1
                for _ in range(count):
                    total_sequences.append(visual_tokens[idx])
            if len(total_sequences) == 0:
                batch['pixel_values'] = None
                self.controller.collect_attn = False
                self.controller.clear()
            else:
                batch['pixel_values'] = total_sequences
        else:
            batch['pixel_values'] = None

        inputs = prepare_inputs_labels_for_multimodal(
            self.llm,
            input_ids=batch.get('input_ids'),
            position_ids=batch.get('position_ids'),
            attention_mask=batch.get('attention_mask'),
            past_key_values=batch.get('past_key_values'),
            labels=batch.get('labels'),
            pixel_values=batch.get('pixel_values'),
            text_features=batch.get('text_features'),
        )
        self._last_label_check = None
        return inputs

    def _pool_question_embedding(self,
                                  input_ids: Optional[torch.Tensor],
                                  attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Pool a per-sample text embedding to condition the router.

        Strategy:
        - keep only tokens indicated by attention_mask
        - remove <image> placeholders
        - mean-pool token embeddings; if empty, return zeros
        Embeddings are cast to the LLM compute dtype and detached.
        """
        if input_ids is None:
            # Fallback: zeros query
            return torch.zeros(
                (1, self.llm.config.hidden_size),
                dtype=self.llm.dtype,
                device=next(self.llm.parameters()).device)
        ids = input_ids.clone()
        mask = attention_mask.bool() if attention_mask is not None else torch.ones_like(ids, dtype=torch.bool)
        emb = self.llm.get_input_embeddings()
        q_list = []
        for row_ids, row_mask in zip(ids, mask):
            row_ids = row_ids[row_mask]
            keep = (row_ids != IMAGE_TOKEN_INDEX)
            row_ids = row_ids[keep]
            if row_ids.numel() == 0:
                q_list.append(torch.zeros(self.llm.config.hidden_size, device=ids.device, dtype=self.llm.dtype))
                continue
            # Clamp to avoid negative/special placeholders breaking embedding lookup
            e = emb(row_ids.clamp(min=0))  # [L, D]
            q_list.append(e.mean(dim=0))
        return torch.stack(q_list, dim=0).to(self.llm.dtype).detach()

    def _prepare_text_only_inputs(self,
                                  input_ids: torch.Tensor,
                                  position_ids: Optional[torch.Tensor],
                                  attention_mask: Optional[torch.Tensor],
                                  past_key_values: Optional[list],
                                  labels: Optional[torch.Tensor]):
        """Strip <image> placeholders and keep pure text embeds when adapters handle vision."""
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        input_ids_list = [
            cur_ids[cur_mask] for cur_ids, cur_mask in zip(input_ids, attention_mask)
        ]
        labels_list = [
            cur_labels[cur_mask] for cur_labels, cur_mask in zip(labels, attention_mask)
        ]

        filtered_ids = []
        filtered_labels = []
        expected_lengths = []
        for cur_ids, cur_labels in zip(input_ids_list, labels_list):
            keep_mask = cur_ids != IMAGE_TOKEN_INDEX
            filtered_ids.append(cur_ids[keep_mask])
            filtered_labels.append(cur_labels[keep_mask])
            expected_lengths.append(
                int(cur_ids.size(0) - (cur_ids == IMAGE_TOKEN_INDEX).sum().item()))

        text_embeds = [self.llm.get_input_embeddings()(ids) for ids in filtered_ids]
        max_len = max((emb.size(0) for emb in text_embeds), default=0)
        batch_size = len(text_embeds)

        if max_len == 0:
            hidden_size = self.llm.config.hidden_size
            inputs_embeds = torch.zeros(
                (batch_size, 1, hidden_size),
                dtype=self.llm.dtype,
                device=input_ids.device)
            new_labels = torch.full(
                (batch_size, 1),
                IGNORE_INDEX,
                dtype=labels.dtype,
                device=labels.device)
            new_attention_mask = torch.ones(
                (batch_size, 1), dtype=torch.bool, device=input_ids.device)
            new_position_ids = torch.zeros(
                (batch_size, 1), dtype=torch.long, device=input_ids.device)
        else:
            hidden_size = text_embeds[0].size(-1) if text_embeds else self.llm.config.hidden_size
            inputs_embeds = torch.zeros(
                (batch_size, max_len, hidden_size),
                dtype=text_embeds[0].dtype if text_embeds else self.llm.dtype,
                device=input_ids.device)
            new_labels = torch.full(
                (batch_size, max_len),
                IGNORE_INDEX,
                dtype=labels.dtype,
                device=labels.device)
            new_attention_mask = torch.zeros(
                (batch_size, max_len),
                dtype=torch.bool,
                device=input_ids.device)
            new_position_ids = torch.zeros(
                (batch_size, max_len),
                dtype=torch.long,
                device=input_ids.device)
            for idx, (embeds, cur_labels) in enumerate(zip(text_embeds, filtered_labels)):
                length = embeds.size(0)
                if length == 0:
                    continue
                inputs_embeds[idx, :length] = embeds
                new_labels[idx, :length] = cur_labels
                new_attention_mask[idx, :length] = True
                new_position_ids[idx, :length] = torch.arange(
                    0, length, dtype=torch.long, device=input_ids.device)

        self._last_label_check = {
            'expected_lengths': expected_lengths,
            'filtered_lengths': [int(x.size(0)) for x in filtered_ids],
        }
        return {
            'input_ids': None,
            'position_ids': new_position_ids,
            'attention_mask': new_attention_mask,
            'past_key_values': past_key_values,
            'inputs_embeds': inputs_embeds,
            'labels': new_labels,
        }
    def _pairwise_rank_loss(
        self,
        s_scores_topk: torch.Tensor,   # [B, K]  学生在 top-K 上的 raw 分数（未 log-softmax）
        t_prob_topk: torch.Tensor,     # [B, K]  老师在 top-K 上的“概率”分布
        pairs: int = 8,
        margin: float = 0.0
    ) -> tuple[torch.Tensor, dict]:
        """
        以老师概率为参考，构造若干 (高,低) 成对样本，约束学生的分差 s_hi - s_lo >= margin。
        用 (t_hi - t_lo) 作为权重，强调老师更确定的对。
        """
        B, K = s_scores_topk.shape
        if K < 2 or pairs <= 0:
            return s_scores_topk.new_zeros([]), {}

        # 排序后把上半段当做“高”，下半段当做“低”
        t_sorted, idx_sorted = torch.sort(t_prob_topk, dim=1, descending=True)  # [B, K]
        hi_idx = idx_sorted[:, : K // 2]
        lo_idx = idx_sorted[:, K // 2 :]

        n = min(pairs, hi_idx.size(1), lo_idx.size(1))
        if n <= 0:
            return s_scores_topk.new_zeros([]), {}

        hi_idx = hi_idx[:, :n]
        lo_idx = lo_idx[:, :n]

        s_hi = torch.gather(s_scores_topk, 1, hi_idx)
        s_lo = torch.gather(s_scores_topk, 1, lo_idx)
        t_hi = torch.gather(t_prob_topk, 1, hi_idx)
        t_lo = torch.gather(t_prob_topk, 1, lo_idx)

        # 老师越“拉开”的 pair 权重越大
        w = (t_hi - t_lo).clamp_min(0).detach()  # [B, n]
        w = w / (w.mean() + 1e-12)

        # hinge / logistic 都行；先用 hinge，数值稳定
        logits = s_hi - s_lo                       # 理想应该 >= margin
        loss = F.relu(margin - logits) * w
        loss = loss.mean()

        stats = {
            'router_rank_gap': logits.mean().detach().to(s_scores_topk.dtype)
        }
        return loss, stats

    def loss(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        inputs = self._prepare_multimodal_inputs(data)
        outputs = self.llm(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            position_ids=inputs['position_ids'],
            past_key_values=inputs['past_key_values'],
            inputs_embeds=inputs['inputs_embeds'],
            labels=inputs['labels'],
            use_cache=False)
        loss = outputs.loss
        router_loss = None
        feat_decor_metric = None

        if self.feat_decor_weight > 0.0 and self._last_latents is not None:
            latents = self._last_latents.to(torch.float32)
            feat_decor_val = _feature_covariance_decorrelation(latents)
            feat_decor_val = feat_decor_val.to(loss.dtype)
            loss = loss + self.feat_decor_weight * feat_decor_val
            feat_decor_metric = feat_decor_val.detach()
        self._last_latents = None

        # ---------------- inside your loss(...) ----------------
        diagnostics = {}
        visual_importance = self.controller.pop_visual_importance()

        if (self.router_aux_weight > 0
                and self._last_router_state is not None
                and visual_importance is not None):

            router_scores = self._last_router_state['scores']     # [B, N]
            router_indices = self._last_router_state['indices']   # [B, K] (top-k indices into N)
            router_mask = self._last_router_state.get('mask')     # [B, N] if you compute ST-TopK mask

            aux_losses = []

            if router_indices is not None and router_indices.numel() > 0:
                B, N = router_scores.shape
                K = router_indices.size(1)

                # --- student on top-K ---
                if router_indices.max() >= N or router_indices.min() < 0:
                    # avoid device assert; skip aux for this batch
                    if self.router_diag:
                        diagnostics['router_idx_oob'] = torch.tensor(
                            1.0, device=router_scores.device, dtype=loss.dtype)
                else:
                    gathered_scores = torch.gather(router_scores, 1, router_indices)  # [B, K]
                    s_logprob = F.log_softmax(
                        gathered_scores.to(torch.float32) / self.router_temperature_s, dim=-1
                    )  # [B, K], float32 for stability

                    # --- teacher aligned to top-K ---
                    t_prob = _align_teacher_probs(
                        teacher=visual_importance,                   # [B, K] or [B, N] depending on your controller
                        router_scores=router_scores,
                        router_indices=router_indices,
                        tau_t=getattr(self, 'router_temperature_t', 1.0)
                    )
                    if t_prob is not None:
                        # KL(student || teacher): input must be log-probs, target probs
                        kl_loss = F.kl_div(s_logprob, t_prob, reduction='batchmean').to(loss.dtype)
                        aux_losses.append(kl_loss)

                        if self.router_diag:
                            diagnostics['router_kl_loss'] = kl_loss.detach().to(loss.dtype)

                        if self.router_diag:
                            p_s = s_logprob.exp()
                            H_s = (-(p_s * s_logprob).sum(dim=-1)).mean().detach()
                            log_t = (t_prob.clamp_min(1e-12)).log()
                            H_t = (-(t_prob * log_t).sum(dim=-1)).mean().detach()

                            # Spearman on same top-K space
                            def _spearman(a, b):
                                ra = torch.argsort(torch.argsort(a, dim=-1), dim=-1).float()
                                rb = torch.argsort(torch.argsort(b, dim=-1), dim=-1).float()
                                ra = ra - ra.mean(dim=-1, keepdim=True)
                                rb = rb - rb.mean(dim=-1, keepdim=True)
                                num = (ra * rb).sum(dim=-1)
                                den = (ra.norm(dim=-1) * rb.norm(dim=-1)).clamp_min(1e-12)
                                return (num / den).mean().detach()

                            rho = _spearman(gathered_scores.detach(), t_prob.detach())

                            # robust param norm for linear/MLP routers
                            w_norm = 0.0
                            if getattr(self, 'router', None) is not None:
                                for p in self.router.parameters():
                                    if p.requires_grad:
                                        w_norm += p.detach().float().norm().item()

                            diagnostics.update({
                                'router_H_student': H_s.to(loss.dtype),
                                'router_H_teacher': H_t.to(loss.dtype),
                                'router_spearman': rho.to(loss.dtype),
                                'router_w_norm': torch.tensor(
                                    w_norm, device=router_scores.device, dtype=loss.dtype),
                                'router_idx_oob': torch.tensor(
                                    0.0, device=router_scores.device, dtype=loss.dtype)
                            })
                            if self.router_rank_lambda > 0:
                                rank_loss, rank_stats = self._pairwise_rank_loss(
                                    s_scores_topk=gathered_scores.to(torch.float32),   # [B, K]
                                    t_prob_topk=t_prob.to(torch.float32),              # [B, K]
                                    pairs=self.router_rank_pairs,
                                    margin=self.router_rank_margin,
                                )
                                # 排序损失在路由器 AUX 内部按 lambda 混入
                                aux_losses.append(self.router_rank_lambda * rank_loss.to(loss.dtype))
                                diagnostics['router_rank_loss'] = rank_loss.detach().to(loss.dtype)
                                for k, v in rank_stats.items():
                                    diagnostics[k] = v
                    else:
                        if self.router_diag:
                            diagnostics['router_teacher_mismatch'] = torch.tensor(
                                1.0, device=router_scores.device, dtype=loss.dtype)

                        # === 新增：pairwise ranking（与 KL 互补，稳住秩一致性） ===
            
            # --- ST-TopK capture mass (optional but useful) ---
            if self.router_enable_capture_mass and router_mask is not None:
                teacher_full: Optional[torch.Tensor] = None
                if self._resampler_teacher is not None:
                    teacher_full = self._resampler_teacher.to(
                        router_scores.device, dtype=router_scores.dtype)
                    if teacher_full.shape[-1] != router_scores.shape[-1]:
                        teacher_full = None
                if teacher_full is None and visual_importance is not None:
                    teacher_full = torch.zeros_like(router_scores, dtype=visual_importance.dtype)
                    if router_indices is not None and router_indices.numel() > 0:
                        K = router_indices.size(1)
                        if visual_importance.size(1) == K:
                            teacher_full.scatter_(1, router_indices,
                                                  visual_importance.to(teacher_full.dtype))
                        elif visual_importance.size(1) == router_scores.size(1):
                            teacher_full = visual_importance.to(teacher_full.dtype)
                        else:
                            teacher_full = None
                if teacher_full is not None:
                    mask_float = router_mask.to(teacher_full.dtype)
                    denom = mask_float.sum(dim=-1).clamp_min(1e-6)
                    mass = (teacher_full * mask_float).sum(dim=-1) / denom     # average captured mass
                    capture_loss = -mass.mean().to(loss.dtype)
                    aux_losses.append(capture_loss)
                    if self.router_diag:
                        diagnostics.update({
                            'router_capture_loss': capture_loss.detach().to(loss.dtype),
                            'router_capture_mass': mass.mean().detach().to(loss.dtype),
                            'router_selected_tokens': denom.mean().detach().to(loss.dtype)
                        })

            if aux_losses:
                # Be tolerant if any aux is scalar tensor with shape []
                router_loss = torch.stack([x if x.ndim == 0 else x.mean() for x in aux_losses]).sum()
                loss = loss + self.router_aux_weight * router_loss

        if (not self.insert_visual_into_seq
                and inputs['inputs_embeds'] is not None
                and self._last_label_check is not None):
            expected = self._last_label_check.get('expected_lengths', [])
            filtered = self._last_label_check.get('filtered_lengths', [])
            assert filtered == expected, (
                'Label alignment mismatch after removing <image> tokens: '
                f'filtered={filtered} expected={expected}')
            self._last_label_check = None
        self._last_router_state = None

        if self.freeze_llm and not loss.requires_grad:
            anchor = None
            for p in self.controller.parameters():
                if p.requires_grad:
                    anchor = p.sum() * 0.0 if anchor is None else anchor + p.sum() * 0.0
            if anchor is not None:
                loss = loss + anchor
        result = {'loss': loss}
        if router_loss is not None:
            result['router_loss'] = router_loss.detach()
            if diagnostics:
                result.update(diagnostics)
        if feat_decor_metric is not None:
            result['feat_decor'] = feat_decor_metric
        return result

    def predict(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        inputs = self._prepare_multimodal_inputs(data)
        outputs = self.llm(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            position_ids=inputs['position_ids'],
            past_key_values=inputs['past_key_values'],
            inputs_embeds=inputs['inputs_embeds'],
            return_dict=True,
            use_cache=False)
        logits = outputs.logits
        return {'logits': logits}

    def forward(self, data, data_samples=None, mode='loss'):
        
        if self.is_first_iter and 'input_ids' in data:
            self.to(data['input_ids'].device)
            self.is_first_iter = False
        if mode == 'loss':
            return self.loss(data)
        if mode == 'predict':
            return self.predict(data)
        raise NotImplementedError

    # ---------------------------------------------------------------------- #
    # State dict / checkpoints
    # ---------------------------------------------------------------------- #
    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        to_return = OrderedDict()
        if self.use_llm_lora:
            to_return.update(get_peft_model_state_dict(self.llm, state_dict=state_dict))
        elif not self.freeze_llm:
            to_return.update({k: v for k, v in state_dict.items() if k.startswith('llm.')})
        if self.token_merger is not None:
            to_return.update({k: v for k, v in state_dict.items() if k.startswith('token_merger.')})
            
        if getattr(self, 'router', None) is not None:
            to_return.update({k: v for k, v in state_dict.items() if k.startswith('router.')})
            
        to_return.update({k: v for k, v in state_dict.items() if k.startswith('resampler.')})
        to_return.update({k: v for k, v in state_dict.items() if k.startswith('feature_in_proj.')})
        to_return.update({k: v for k, v in state_dict.items() if k.startswith('projector.')})
    
        to_return.update({k: v for k, v in state_dict.items() if k.startswith('controller.')})
        return to_return

    def to_xtuner_llava(self, cfg, save_dir, fp32=False, save_pretrained_kwargs=None):
        save_pretrained_kwargs = save_pretrained_kwargs or {}
        self.llm.config.use_cache = True
        if not fp32:
            self.llm.half()
        if self.use_llm_lora:
            adapter_dir = osp.join(save_dir, 'llm_adapter')
            os.makedirs(adapter_dir, exist_ok=True)
            self.llm.save_pretrained(adapter_dir, **save_pretrained_kwargs)
        elif not self.freeze_llm:
            tokenizer = BUILDER.build(cfg.tokenizer)
            tokenizer.save_pretrained(save_dir, **save_pretrained_kwargs)
            self.llm.save_pretrained(save_dir, **save_pretrained_kwargs)
        self.llm.config.use_cache = False

        proj_dir = osp.join(save_dir, 'projector')
        os.makedirs(proj_dir, exist_ok=True)
        save_file(self.projector.state_dict(),
                  osp.join(proj_dir, 'projector.safetensors'))

        merger_dir = osp.join(save_dir, 'token_merger')
        os.makedirs(merger_dir, exist_ok=True)
        if self.token_merger is not None:
            save_file(self.token_merger.state_dict(),
                      osp.join(merger_dir, 'merger.safetensors'))

        resampler_dir = osp.join(save_dir, 'resampler')
        os.makedirs(resampler_dir, exist_ok=True)
        save_file(self.resampler.state_dict(),
                  osp.join(resampler_dir, 'resampler.safetensors'))

        in_proj_state = self.feature_in_proj.state_dict()
        if in_proj_state:
            in_proj_dir = osp.join(save_dir, 'feature_in_proj')
            os.makedirs(in_proj_dir, exist_ok=True)
            save_file(in_proj_state,
                      osp.join(in_proj_dir, 'feature_in_proj.safetensors'))

        controller_dir = osp.join(save_dir, 'vision_adapter')
        os.makedirs(controller_dir, exist_ok=True)
        save_file(self.controller.state_dict(),
                  osp.join(controller_dir, 'controller.safetensors'))

    # ---------------------------------------------------------------------- #
    # Generation helper
    # ---------------------------------------------------------------------- #
    @torch.no_grad()
    def generate(self,
                 pixel_values: Optional[torch.Tensor] = None,
                 vision_pad_mask: Optional[torch.Tensor] = None,
                 tokenizer=None,
                 prompt: Optional[str] = None,
                 **generate_kwargs):
        device = next(self.llm.parameters()).device
        if tokenizer is None:
            raise ValueError('Tokenizer is required for generation.')

        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        batch = dict(inputs)
        if pixel_values is not None:
            batch['pixel_values'] = pixel_values.to(device)
        if vision_pad_mask is not None:
            batch['vision_pad_mask'] = vision_pad_mask.to(device)
        inputs_prepared = self._prepare_multimodal_inputs(batch)

        generated = self.llm.generate(
            input_ids=inputs_prepared['input_ids'],
            attention_mask=inputs_prepared['attention_mask'],
            inputs_embeds=inputs_prepared['inputs_embeds'],
            **generate_kwargs)
        return tokenizer.batch_decode(generated, skip_special_tokens=True)

    def to_hf(self,
              cfg,
              save_dir,
              fp32=False,
              save_pretrained_kwargs=None,
              save_format='xtuner',
              **kwargs):
        if save_format != 'xtuner':
            raise NotImplementedError('Only xtuner save_format is implemented.')
        self.to_xtuner_llava(cfg, save_dir, fp32, save_pretrained_kwargs)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.llm, name)
