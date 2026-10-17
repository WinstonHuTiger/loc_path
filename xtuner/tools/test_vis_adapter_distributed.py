"""
Multi-GPU distributed inference script for LlavaVisAdapterModel.

Reference: xtuner/tools/test_for_json_files_distributed.py

This script scans a folder of *_test.json files, loads pre-extracted
WSI patch features from .h5 files (features + coords), builds prompts
with image tokens, and runs generation in distributed fashion.

Key differences for LlavaVisAdapterModel:
- Visual feature processing (token merge, resampler, projector) happens
  inside the model via _prepare_multimodal_inputs, using coords for
  positional encoding. We provide raw feats and coords as inputs.
"""

import argparse
import copy
import glob
import json
import os
import os.path as osp
import time
from contextlib import nullcontext
from types import FunctionType
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist

from mmengine.config import Config, DictAction
from mmengine import print_log
from mmengine.dist import init_dist, get_dist_info
from mmengine.registry import MODELS as BUILDER

from transformers import AutoTokenizer, GenerationConfig, StoppingCriteriaList

from xtuner.configs import cfgs_name_path
from xtuner.configs.loader import load_config
from xtuner.dataset.utils import load_slide_features
from xtuner.registry import MAP_FUNC
from xtuner.model.utils import guess_load_checkpoint
from xtuner.utils import (
    DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX,
    StopWordStoppingCriteria, PROMPT_TEMPLATE
)


TORCH_DTYPE_MAP = dict(fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32, auto='auto')


def _flatten_path(value) -> Sequence[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        paths: List[str] = []
        for item in value:
            paths.extend(_flatten_path(item))
        return paths
    return []


def _estimate_axis_step(values: np.ndarray) -> float:
    uniq = np.unique(values)
    if uniq.size <= 1:
        return 1.0
    diffs = np.diff(np.sort(uniq))
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 1.0
    step = float(np.median(diffs))
    if step <= 0.0:
        step = float(np.max(diffs))
    return step if step > 0.0 else 1.0


def _gridify_coords(coords: torch.Tensor,
                    *,
                    eps: float = 1e-6,
                    report_tag: Optional[str] = None) -> torch.Tensor:
    if coords.numel() == 0:
        return coords

    device = coords.device
    arr = coords.to(dtype=torch.float64, device='cpu').numpy()
    base = arr.min(axis=0, keepdims=True)
    rel = arr - base

    grid = np.empty_like(rel, dtype=np.float64)
    steps: Tuple[float, float] = (1.0, 1.0)
    for axis in range(rel.shape[1]):
        pitch = _estimate_axis_step(rel[:, axis])
        pitch = max(pitch, eps)
        scaled = np.round(rel[:, axis] / pitch).astype(np.int64, copy=False)
        grid[:, axis] = scaled
        steps = steps[:axis] + (pitch,) + steps[axis + 1:]

    if report_tag:
        mins = grid.min(axis=0)
        maxs = grid.max(axis=0)
        print_log(
            f'{report_tag}: stride≈({steps[0]:.3f}, {steps[1]:.3f}) '
            f'range=({mins[0]}-{maxs[0]}, {mins[1]}-{maxs[1]})',
            logger='current')

    return torch.from_numpy(grid.astype(np.float32)).to(device=device)


_DINOV2_TUMOR_CACHE: Dict[str, Dict[str, str]] = {}


def parse_args():
    ap = argparse.ArgumentParser("Distributed folder JSON inference for LlavaVisAdapterModel")
    ap.add_argument('config', help='xtuner/mmengine config (file path or cfgs_name_path key)')
    ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--data_dir', required=True, help='Folder with many *_test.json')
    ap.add_argument('--feature_root', required=True,
                    help='Root like .../conch_v1 (contains <tumor><identifier>/h5_files/<case>.h5)')
    ap.add_argument('--feature-backend', default='auto', choices=['auto', 'conch', 'dinov2'],
                    help='Choose feature layout; auto will try Conch first then DINOv2.')
    ap.add_argument('--identifier', default='_224x224_b20_t15', help='Suffix directory appended to tumor folder')
    ap.add_argument('--image_feature_suffix', default='.h5', choices=['.h5', '.npy'],
                    help='Feature file suffix; `.h5` for Conch, `.npy` for DINOv2.')
    ap.add_argument('--dinov2-feature-filename', default=None,
                    help='Fixed filename to look for within each DINOv2 case directory. '
                         'If omitted, built-in defaults will be tried.')
    ap.add_argument('--dinov2-case-suffix', default=None,
                    help='Optional suffix appended to the case id when locating DINOv2 files '
                         '(e.g. "_features.npy"). If set, the resolver will try '
                         'both "<case><suffix>" and --dinov2-feature-filename when searching.')
    ap.add_argument('--dinov2-fallback-filenames', nargs='*', default=None,
                    help='Additional candidate filenames to try inside each DINOv2 folder. '
                         'Use space-separated values. These are tried after the primary name.')
    ap.add_argument('--dinov2-no-coords', dest='dinov2_expect_coords', action='store_false',
                    help='Treat DINOv2 feature files as features-only (no coords). '
                         'Synthesizes zero coords per token.')
    ap.add_argument('--dinov2-expect-coords', dest='dinov2_expect_coords', action='store_true',
                    help='Expect coords to be present within DINOv2 features (default).')
    ap.set_defaults(dinov2_expect_coords=True)
    ap.set_defaults(dinov2_enforce_unit_stride=True)
    ap.add_argument('--dinov2-enforce-unit-stride', dest='dinov2_enforce_unit_stride', action='store_true',
                    help='Force DINOv2 coords onto a unit lattice (default).')
    ap.add_argument('--dinov2-allow-nonunit-stride', dest='dinov2_enforce_unit_stride', action='store_false',
                    help='Disable coord grid snapping for DINOv2 features.')
    ap.add_argument('--dinov2-stride-report', action='store_true',
                    help='Print stride estimation stats for each DINOv2 slide.')
    ap.add_argument('--conch-snap-coords', dest='conch_snap_coords', action='store_true',
                    help='Snap CONCH coords to an integer grid before inference (default).')
    ap.add_argument('--no-conch-snap-coords', dest='conch_snap_coords', action='store_false',
                    help='Disable CONCH coord snapping.')
    ap.set_defaults(conch_snap_coords=True)
    ap.add_argument('--conch-stride-report', action='store_true',
                    help='Log inferred CONCH coordinate stride during snapping.')
    ap.add_argument('--out_csv', required=True)
    ap.add_argument('--out_jsonl', default=None)
    ap.add_argument('--torch-dtype', default='bf16', choices=TORCH_DTYPE_MAP.keys())
    ap.add_argument('--router-top-m', type=int, default=None,
                    help='Override `model.router_top_m` during inference.')
    ap.add_argument('--router-top-m-auto', action='store_true',
                    help='Automatically choose router_top_m based on question type (close/open).')
    ap.add_argument('--router-top-m-close', type=int, default=None,
                    help='router_top_m to use for close questions when --router-top-m-auto is set.')
    ap.add_argument('--router-top-m-open', type=int, default=None,
                    help='router_top_m to use for open questions when --router-top-m-auto is set.')
    ap.add_argument('--work-dir', default=None)
    ap.add_argument('--cfg-options', nargs='+', action=DictAction)
    ap.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none',
                    help='Job launcher for distributed running.')
    ap.add_argument('--sample_num', type=int, default=10240)
    ap.add_argument('--sample_strategy', default='linspace', choices=['linspace', 'random', 'random_full'])
    ap.add_argument('--debug_max_samples', type=int, default=None)
    ap.add_argument('--max_new_tokens', type=int, default=512)

    ap.add_argument('--decoding', default='greedy',
                    choices=['greedy', 'sampling', 'beam', 'contrastive'])
    ap.add_argument('--temperature', type=float, default=0.7)
    ap.add_argument('--top_p', type=float, default=0.9)
    ap.add_argument('--top_k', type=int, default=50)
    ap.add_argument('--typical_p', type=float, default=1.0)
    ap.add_argument('--repetition_penalty', type=float, default=1.0)
    ap.add_argument('--no_repeat_ngram_size', type=int, default=0)
    ap.add_argument('--num_beams', type=int, default=1)
    ap.add_argument('--length_penalty', type=float, default=1.0)
    ap.add_argument('--early_stopping', action='store_true')
    ap.add_argument('--num_return_sequences', type=int, default=1)
    ap.add_argument('--penalty_alpha', type=float, default=None)

    ap.add_argument('--gen_config_json', type=str, default=None,
                    help='JSON string to override HF GenerationConfig fields')

    # Open-question multi-sample + rerank
    ap.add_argument('--open_multi_num_return_sequences', type=int, default=3,
                    help='For open questions, sample K candidates and re-rank (K<=1 disables).')
    ap.add_argument('--save_all_candidates', action='store_true',
                    help='If set, save all candidates and chosen index in outputs.')

    # Prompt/template control
    ap.add_argument('--prompt-template', default='auto', choices=['auto', 'vicuna', 'qwen_chat'],
                    help='Force prompt template; auto = infer from LLM name/arch.')

    # Logging controls
    ap.add_argument('--quiet', action='store_true')
    ap.add_argument('--all-ranks-log', action='store_true')
    ap.add_argument('--log-interval', type=int, default=200)
    ap.add_argument('--log-interval-sec', type=float, default=15.0)
    ap.add_argument('--per-sample-log', action='store_true')
    ap.add_argument('--profile-infer', action='store_true',
                    help='Log inference wall time and GPU memory per sample.')
    ap.add_argument('--profile-trace', action='store_true',
                    help='Capture a torch.profiler trace for the first profiled sample.')
    ap.add_argument('--profile-out', type=str, default=None,
                    help='Optional path to save aggregated profiling stats as JSON.')
    ap.add_argument('--profile-flops', action='store_true',
                    help='Estimate FLOPs via torch.profiler (best-effort).')
    return ap.parse_args()


def gather_results(data, world_size):
    if world_size == 1:
        return data
    gathered_data = [None] * world_size
    dist.all_gather_object(gathered_data, data)
    combined_data = []
    for sublist in gathered_data:
        combined_data.extend(sublist)
    return combined_data


def register_function(cfg_dict):
    if isinstance(cfg_dict, dict):
        for k, v in cfg_dict.items():
            if isinstance(v, FunctionType):
                vs = str(v)
                if vs not in MAP_FUNC:
                    MAP_FUNC.register_module(module=v, name=vs)
                cfg_dict[k] = vs
            else:
                register_function(v)
    elif isinstance(cfg_dict, (list, tuple)):
        for v in cfg_dict:
            register_function(v)


def _normalize_tumor_name(raw: str) -> str:
    token = raw.lower().strip()
    for prefix in ('tcga-', 'tcga_', 'tcga '):
        if token.startswith(prefix):
            token = token[len(prefix):]
    if token.startswith('tcga'):
        token = token[4:]
    token = token.replace('tcga-', '').replace('tcga_', '')
    token = token.strip('_- ')
    return token or 'unknown'


def _case_candidates(paths: Sequence[str],
                     sample_id: Optional[str] = None) -> List[str]:
    candidates: List[str] = []

    def _append(value: Optional[str]):
        if value and value not in candidates:
            candidates.append(value)

    for p in paths:
        base = os.path.splitext(os.path.basename(p))[0]
        _append(base)
        if '.' in base:
            _append(base.split('.', 1)[0])
        parts = base.rsplit('-', 1)
        if len(parts) == 2 and parts[1].isdigit():
            _append(parts[0])

    if sample_id:
        sid = sample_id.split('.', 1)[0]
        _append(sid)
        parts = sid.rsplit('-', 1)
        if len(parts) == 2 and parts[1].isdigit():
            _append(parts[0])

    return candidates


def _parse_stub_conch(image_path: str) -> Tuple[str, str]:
    norm = os.path.normpath(image_path)
    parts = norm.split(os.sep)
    if len(parts) >= 2:
        parent = parts[-2]
    else:
        parent = parts[0] if parts else 'unknown'
    tumor_abbrev = _normalize_tumor_name(parent)
    stem = os.path.splitext(parts[-1])[0] if parts else ''
    case = stem.split('.', 1)[0] if stem else stem
    return tumor_abbrev, case


def _parse_stub_dinov2(image_path: str) -> Tuple[str, str]:
    tumor, case = _parse_stub_conch(image_path)
    return tumor, case


def _build_feature_path_conch(feature_root: str, tumor_abbrev: str, case_name: str,
                              identifier: str, suffix: str) -> str:
    base = os.path.join(feature_root, f"{tumor_abbrev}{identifier}")
    if suffix == ".h5":
        return os.path.join(base, "h5_files", case_name + suffix)
    if suffix == ".npy":
        return os.path.join(base, "npy_files", f"{case_name}_features.npy")
    raise ValueError(f"Unsupported CONCH suffix: {suffix}")


def _find_dinov2_tumor_root(feature_root: str, tumor_norm: str) -> str:
    direct = os.path.join(feature_root, tumor_norm)
    if os.path.isdir(direct):
        return direct
    for entry in os.listdir(feature_root):
        if entry.lower() == tumor_norm:
            return os.path.join(feature_root, entry)
    raise FileNotFoundError(
        f'DINOv2 feature directory not found: {os.path.join(feature_root, tumor_norm)}')


def _dinov2_case_lookup(feature_root: str, tumor: str) -> Dict[str, str]:
    tumor_norm = _normalize_tumor_name(tumor)
    cached = _DINOV2_TUMOR_CACHE.get(tumor_norm)
    if cached is not None:
        return cached

    root = _find_dinov2_tumor_root(feature_root, tumor_norm)

    mapping: Dict[str, str] = {}
    for entry in os.listdir(root):
        path = os.path.join(root, entry)
        if not os.path.isdir(path):
            continue
        key_full = entry.lower()
        mapping[key_full] = path
        prefix = entry.split('.')[0].lower()
        mapping.setdefault(prefix, path)
    if not mapping:
        raise FileNotFoundError(f'No DINOv2 feature folders found under {root}')
    _DINOV2_TUMOR_CACHE[tumor_norm] = mapping
    return mapping


DEFAULT_DINOV2_FILENAMES = [
    'tile_embeddings_features.npy',
    'tile_embeddings.npy',
    'slide_embeddings.tokens_last.npy',
    'slide_embeddings.npy',
]


def _resolve_dinov2_feature_path(feature_root: str, tumor: str, case: str,
                                 filename: Optional[str],
                                 case_suffix: Optional[str] = None,
                                 fallback_filenames: Optional[Sequence[str]] = None,
                                 *,
                                 case_candidates: Optional[Sequence[str]] = None) -> str:
    mapping = _dinov2_case_lookup(feature_root, tumor)
    tried_paths: List[str] = []
    search_list = case_candidates or [case]
    base_filenames: List[str] = []
    if filename:
        base_filenames.append(filename)
    else:
        base_filenames.extend(DEFAULT_DINOV2_FILENAMES)
    if fallback_filenames:
        for item in fallback_filenames:
            if item and item not in base_filenames:
                base_filenames.append(item)
    for candidate in search_list:
        key = candidate.lower()
        folder = mapping.get(key)
        folders_to_try: List[str] = []
        if folder is not None:
            folders_to_try.append(folder)
        for mkey, mfolder in mapping.items():
            if mkey.startswith(key) and mfolder not in folders_to_try:
                folders_to_try.append(mfolder)
        candidate_filenames: List[str] = []
        if case_suffix is not None:
            candidate_filenames.append(candidate + case_suffix)
        candidate_filenames.extend(base_filenames)
        for folder_path in folders_to_try:
            for name in candidate_filenames:
                resolved = os.path.join(folder_path, name)
                tried_paths.append(resolved)
                if osp.exists(resolved):
                    return resolved
    raise FileNotFoundError(
        f'No DINOv2 features for {(tumor, case)} under {feature_root} '
        f'(tried paths: {tried_paths})')


def _resolve_feature_backend_and_path(sample: Dict,
                                      args) -> Tuple[str, str]:
    explicit = sample.get('feature_path')
    if explicit:
        feat_path = explicit if os.path.isabs(explicit) else os.path.join(args.feature_root, explicit)
        if not osp.exists(feat_path):
            raise FileNotFoundError(f'Explicit feature_path not found: {feat_path}')
        lower = feat_path.lower()
        if lower.endswith('.csv'):
            backend = 'bcnb_csv'
        elif lower.endswith('.npy'):
            backend = 'dinov2'
        else:
            backend = 'conch'
        return backend, feat_path

    image_field = sample.get('image') or sample.get('images')
    paths = _flatten_path(image_field)
    if not paths:
        raise KeyError('Sample does not contain image paths for feature resolution.')
    stub_path = paths[0]
    case_candidates = _case_candidates(paths, sample.get('id'))

    errors: List[str] = []

    if args.feature_backend in ('auto', 'conch'):
        tumor, case = _parse_stub_conch(stub_path)
        try:
            suffix_candidates = []
            preferred = args.image_feature_suffix
            if preferred in ('.h5', '.npy'):
                suffix_candidates.append(preferred)
            else:
                suffix_candidates.append('.h5')
            if '.npy' not in suffix_candidates:
                suffix_candidates.append('.npy')
            if '.h5' not in suffix_candidates:
                suffix_candidates.append('.h5')

            for suffix in suffix_candidates:
                feat_path = _build_feature_path_conch(
                    args.feature_root, tumor, case, args.identifier, suffix)
                if osp.exists(feat_path):
                    return 'conch', feat_path
                errors.append(f'Conch path not found ({suffix}): {feat_path}')
        except Exception as exc:  # pragma: no cover - defensive
            errors.append(f'Conch resolve failed: {exc}')
        if args.feature_backend == 'conch':
            raise FileNotFoundError('; '.join(errors))

    if args.feature_backend in ('auto', 'dinov2'):
        tumor, case = _parse_stub_dinov2(stub_path)
        try:
            feat_path = _resolve_dinov2_feature_path(
                args.feature_root, tumor, case, args.dinov2_feature_filename,
                case_suffix=args.dinov2_case_suffix,
                fallback_filenames=args.dinov2_fallback_filenames,
                case_candidates=case_candidates)
            if osp.exists(feat_path):
                return 'dinov2', feat_path
            errors.append(f'DINOv2 path not found: {feat_path}')
        except Exception as exc:
            errors.append(f'DINOv2 resolve failed: {exc}')
        if args.feature_backend == 'dinov2':
            raise FileNotFoundError('; '.join(errors))

    raise FileNotFoundError('; '.join(errors) if errors else
                            'Unable to resolve feature path for sample.')


def _load_bcnb_csv_features(csv_path: str) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float]]:
    df = pd.read_csv(csv_path)

    patch_col = None
    for col in df.columns:
        if str(col).lower() == 'patch_name':
            patch_col = col
            break
    if patch_col is None:
        raise KeyError(f"'patch_name' column not found in {csv_path}")

    feat_cols = [col for col in df.columns if col != patch_col]
    if not feat_cols:
        raise ValueError(f'No feature columns detected in {csv_path}')

    feats = df[feat_cols].to_numpy(dtype=np.float32, copy=False)
    patch_names = df[patch_col].astype(str).tolist()
    if len(patch_names) != feats.shape[0]:
        raise ValueError(f'Feature/patch count mismatch in {csv_path}')

    coords = np.zeros((len(patch_names), 2), dtype=np.float64)
    for idx, name in enumerate(patch_names):
        stem = os.path.splitext(name)[0]
        parts = stem.split('_')
        if len(parts) < 2:
            raise ValueError(f'Cannot parse coords from patch_name="{name}" in {csv_path}')
        try:
            x_val = float(parts[-2])
            y_val = float(parts[-1])
        except ValueError as exc:
            raise ValueError(f'Invalid numeric coords in patch_name="{name}" ({csv_path})') from exc
        coords[idx, 0] = x_val
        coords[idx, 1] = y_val

    stride_x = _estimate_axis_step(coords[:, 0])
    stride_y = _estimate_axis_step(coords[:, 1])

    coords[:, 0] -= coords[:, 0].min()
    coords[:, 1] -= coords[:, 1].min()

    coords_out = np.ascontiguousarray(coords.astype(np.float32))
    feats_out = np.ascontiguousarray(feats.astype(np.float32))
    return feats_out, coords_out, (stride_x, stride_y)


def _choose_indices(total_rows: int, k: int, strategy: str, rng: np.random.Generator):
    if total_rows <= 0:
        return np.array([], dtype=int)
    if strategy == "random_full":
        replace = total_rows < k
        idx = rng.choice(total_rows, size=k, replace=replace)
        return np.sort(idx.astype(int))
    if strategy == "random":
        if total_rows <= k:
            return np.arange(total_rows, dtype=int)
        idx = rng.choice(total_rows, size=k, replace=False)
        return np.sort(idx.astype(int))
    if total_rows <= k:
        return np.arange(total_rows, dtype=int)
    step = total_rows / k
    start = int(rng.integers(0, max(1, int(step))))
    indices = (np.floor(np.arange(k) * step + start)).astype(int)
    return np.clip(indices, 0, total_rows - 1)


def _rng(seed: int = 3407):
    base = seed & 0xFFFFFFFF
    seed_mix = (base ^ (torch.initial_seed() & 0xFFFFFFFF)) & 0xFFFFFFFF
    return np.random.default_rng(seed_mix)


def build_generation_config(args, tokenizer):
    cfg = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=(tokenizer.pad_token_id
                      if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    )
    cfg.repetition_penalty = args.repetition_penalty
    cfg.no_repeat_ngram_size = args.no_repeat_ngram_size
    if args.decoding == 'greedy':
        cfg.do_sample = False
        cfg.num_beams = 1
    elif args.decoding == 'beam':
        cfg.do_sample = False
        cfg.num_beams = max(2, args.num_beams)
        cfg.length_penalty = args.length_penalty
        cfg.early_stopping = args.early_stopping
    elif args.decoding == 'sampling':
        cfg.do_sample = True
        cfg.temperature = (args.temperature if args.temperature > 0 else 0.7)
        cfg.top_p = args.top_p
        cfg.top_k = args.top_k
        cfg.typical_p = args.typical_p
        cfg.num_return_sequences = max(1, args.num_return_sequences)
    elif args.decoding == 'contrastive':
        cfg.do_sample = False
        cfg.top_k = max(1, args.top_k)
        cfg.penalty_alpha = args.penalty_alpha if args.penalty_alpha is not None else 0.6
    if args.gen_config_json:
        import json as _json
        for k, v in _json.loads(args.gen_config_json).items():
            setattr(cfg, k, v)
    return cfg


def make_open_multi_config(base_cfg: GenerationConfig, args) -> GenerationConfig:
    """Clone and override a base GenerationConfig to enable multi-sample for open Qs."""
    cfg = copy.deepcopy(base_cfg)
    # Ensure sampling mode to allow num_return_sequences > 1
    cfg.do_sample = True
    cfg.num_beams = 1
    cfg.temperature = (args.temperature if args.temperature > 0 else 0.7)
    cfg.top_p = args.top_p
    cfg.top_k = args.top_k
    cfg.typical_p = args.typical_p
    cfg.num_return_sequences = max(1, int(args.open_multi_num_return_sequences))
    return cfg


def _clean_text_tail(s: str, use_vicuna: bool) -> str:
    s = s or ""
    if use_vicuna:
        while s.endswith("</s>"):
            s = s[:-4]
    else:
        if s.endswith('<|im_end|>'):
            s = s[:-10]
    return s


def _repetition_rate_by_token(tokenizer, text: str) -> float:
    """Lower is better (less repetition). Uses tokenizer-level tokens for language-agnostic behavior."""
    toks = [t for t in tokenizer.tokenize(text) if t.strip()]
    n = len(toks)
    if n == 0:
        return 1.0
    uniq = len(set(toks))
    return 1.0 - (uniq / n)


def _unique_token_count(tokenizer, text: str) -> int:
    toks = [t for t in tokenizer.tokenize(text) if t.strip()]
    return len(set(toks))


def rerank_candidates(tokenizer, cands: list) -> int:
    """Return index of best candidate by (least repetition, longest unique coverage, longest length)."""
    scored = []
    for idx, t in enumerate(cands):
        rep = _repetition_rate_by_token(tokenizer, t)
        uniq = _unique_token_count(tokenizer, t)
        clen = len(t)
        scored.append((rep, -uniq, -clen, idx))  # sort ascending
    scored.sort()
    return scored[0][-1] if scored else 0


def load_features_and_coords(
    sample: Dict,
    args,
    rng: np.random.Generator,
    logger=None,
):
    backend, feat_path = _resolve_feature_backend_and_path(sample, args)
    if logger is not None:
        logger.info(f'Using {backend} features: {feat_path}')

    sample_num = args.sample_num
    sample_strategy = args.sample_strategy

    if backend == 'conch':
        conch_snap = args.conch_snap_coords and feat_path.lower().endswith(('.h5', '.hdf5'))
        feats_np, coords_np, stride = load_slide_features(
            feat_path,
            feature_key='features',
            coord_key='coords',
            expect_coords=True,
            conch_snap_coords=conch_snap,
            zero_anchor=True)
        if args.conch_stride_report and stride is not None and logger is not None:
            logger.info(f'CONCH stride≈({stride[0]}, {stride[1]}) for {feat_path}')

        total = feats_np.shape[0]
        idx = _choose_indices(total, sample_num, sample_strategy, rng)
        feats_sel = feats_np[idx] if idx.size else feats_np[:0]
        coords_sel = coords_np[idx] if idx.size else coords_np[:0]
        feats = torch.from_numpy(np.ascontiguousarray(feats_sel))
        coords = torch.from_numpy(np.ascontiguousarray(coords_sel))
        return feats, coords, feat_path, backend

    if backend == 'bcnb_csv':
        feats_np, coords_np, stride = _load_bcnb_csv_features(feat_path)

        total = feats_np.shape[0]
        idx = _choose_indices(total, sample_num, sample_strategy, rng)
        feats_sel = feats_np[idx] if idx.size else feats_np[:0]
        coords_sel = coords_np[idx] if idx.size else coords_np[:0]

        feats = torch.from_numpy(np.ascontiguousarray(feats_sel))
        coords = torch.from_numpy(np.ascontiguousarray(coords_sel))
        if logger is not None and getattr(args, 'conch_stride_report', False):
            logger.info(f'BCNB stride≈({stride[0]}, {stride[1]}) for {feat_path}')
        return feats, coords, feat_path, backend

    if backend != 'dinov2':
        raise ValueError(f'Unsupported feature backend: {backend}')

    if args.dinov2_expect_coords:
        feats_np, coords_np, _ = load_slide_features(
            feat_path,
            feature_key='features',
            coord_key='coords',
            expect_coords=True,
            conch_snap_coords=False)
    else:
        feats_np = np.load(feat_path, allow_pickle=False)
        if feats_np.ndim != 2:
            raise ValueError(f'Expected 2D feature array in {feat_path}, got {feats_np.shape}')
        coords_np = np.zeros((feats_np.shape[0], 2), dtype=np.float32)

    total = feats_np.shape[0]
    idx = _choose_indices(total, sample_num, sample_strategy, rng)
    feats_sel = feats_np[idx] if idx.size else feats_np[:0]
    coords_sel = coords_np[idx] if idx.size else coords_np[:0]

    feats = torch.from_numpy(np.ascontiguousarray(feats_sel))
    coords = torch.from_numpy(np.ascontiguousarray(coords_sel))
    if args.dinov2_enforce_unit_stride and args.dinov2_expect_coords:
        report_tag = feat_path if args.dinov2_stride_report else None
        coords = _gridify_coords(coords, report_tag=report_tag)
    return feats, coords, feat_path, backend


def build_input_ids_from_prompt(tokenizer, raw_prompt: str, prompt_template):
    SYSTEM = ''
    instruction = prompt_template.get('INSTRUCTION', '{input}')
    if DEFAULT_IMAGE_TOKEN in raw_prompt or "<image>" in raw_prompt:
        raw_prompt = raw_prompt.replace("<image>", DEFAULT_IMAGE_TOKEN)
    else:
        raw_prompt = DEFAULT_IMAGE_TOKEN + '\n' + raw_prompt
    inputs = (SYSTEM + instruction).format(input=raw_prompt, round=1)
    chunk_encode = []
    for idx, chunk in enumerate(inputs.split(DEFAULT_IMAGE_TOKEN)):
        if idx == 0:
            cur_encode = tokenizer.encode(chunk)
        else:
            cur_encode = tokenizer.encode(chunk, add_special_tokens=False)
        chunk_encode.append(cur_encode)
    assert len(chunk_encode) == 2
    input_ids = []
    for idx, cur_chunk_encode in enumerate(chunk_encode):
        input_ids.extend(cur_chunk_encode)
        if idx != len(chunk_encode) - 1:
            input_ids.append(IMAGE_TOKEN_INDEX)
    return torch.tensor(input_ids, dtype=torch.long)


class ThrottledLogger:
    def __init__(self, rank: int, args):
        self.rank = rank
        self.args = args
        self._last_t = 0.0
        self._count = 0

    def _should_emit(self, force: bool = False) -> bool:
        if not self.args.all_ranks_log and self.rank != 0:
            return False
        if force:
            return True
        if self.args.quiet:
            return False
        now = time.time()
        by_count = (self._count % max(1, int(self.args.log_interval)) == 0)
        by_time = (now - self._last_t) >= float(self.args.log_interval_sec)
        if by_count or by_time:
            self._last_t = now
            return True
        return False

    def info(self, msg: str, force: bool = False):
        self._count += 1
        if self._should_emit(force=force):
            print_log(msg)

    def warn(self, msg: str, force: bool = True):
        if not self.args.all_ranks_log and self.rank != 0:
            return
        print_log(msg)


def main():
    args = parse_args()

    # dist init
    if args.launcher != 'none':
        init_dist(args.launcher)
        rank, world_size = get_dist_info()
    else:
        rank, world_size = 0, 1

    logger = ThrottledLogger(rank, args)

    # set device per rank
    if torch.cuda.is_available():
        ngpu = torch.cuda.device_count()
        if ngpu > 0:
            torch.cuda.set_device(rank % ngpu)
        device = torch.device('cuda', rank % max(1, ngpu))
    else:
        device = torch.device('cpu')

    dtype = TORCH_DTYPE_MAP[args.torch_dtype]

    # config
    cfg_path = args.config if osp.isfile(args.config) else cfgs_name_path.get(args.config, None)
    if not cfg_path:
        raise FileNotFoundError(f'Cannot find config {args.config}')
    cfg = load_config(cfg_path)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    register_function(cfg._cfg_dict)

    # A full LoC-Path checkpoint already includes the trained visual modules.
    # Do not require the historical MAE/Stage-1 paths during deployment.
    if args.checkpoint:
        cfg.model.resampler_ckpt = None
        cfg.model.pretrained_pth = None

    # build model
    logger.info("Building LlavaVisAdapterModel from config...", force=True)
    model = BUILDER.build(cfg.model)

    if args.checkpoint:
        sd = guess_load_checkpoint(args.checkpoint)
        required = set(model.state_dict())
        absent = required.difference(sd)
        if absent:
            raise RuntimeError(
                f'Checkpoint is missing {len(absent)} LoC-Path weights: '
                f'{sorted(absent)[:5]}. Use a complete checkpoint or the '
                'locpath_model.pth export from this release, and a matching config.')
        missing, unexpected = model.load_state_dict(sd, strict=False)
        
        logger.info(f'Checkpoint loaded. Missing: {len(missing)} | Unexpected: {len(unexpected)}', force=True)
        logger.info(f'Unexpected: {unexpected}', force=True)
    
    base_router_top_m = getattr(model, 'router_top_m', None)
    if args.router_top_m is not None:
        if getattr(model, 'router', None) is None:
            logger.warn('router_top_m override requested but model.router is None; ignoring.', force=True)
        else:
            model.router_top_m = int(args.router_top_m)
            base_router_top_m = model.router_top_m
            logger.info(f'router_top_m overridden to {model.router_top_m}', force=True)

    model = model.eval().to(device, dtype = dtype)
    llm = model.llm.eval()

    # tokenizer / template / stops
    llm_name_or_path = getattr(llm.config, "_name_or_path", "")
    llm_arch = getattr(llm.config, "architectures", []) or []
    raw_name = str(llm_name_or_path).lower() + " " + " ".join(map(str, llm_arch)).lower()

    auto_is_vicuna_like = any(k in raw_name for k in ["vicuna", "llama", "llama-2", "llama-3"])
    if args.prompt_template == 'vicuna':
        use_vicuna = True
    elif args.prompt_template == 'qwen_chat':
        use_vicuna = False
    else:
        use_vicuna = auto_is_vicuna_like

    tok_model_name = llm_name_or_path if llm_name_or_path else ("lmsys/vicuna-7b-v1.5" if use_vicuna else "Qwen/Qwen2.5-7B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained(tok_model_name, trust_remote_code=True, encode_special_tokens=True)

    if use_vicuna:
        prompt_template = PROMPT_TEMPLATE.vicuna
        stop_words = ["</s>"]
    else:
        prompt_template = PROMPT_TEMPLATE.qwen_chat
        stop_words = prompt_template.get('STOP_WORDS', [])

    stop_criteria = StoppingCriteriaList([StopWordStoppingCriteria(tokenizer, w) for w in stop_words])
    gen_config_open = build_generation_config(args, tokenizer)

    # also build greedy config for close-ended questions
    _decoding_backup = args.decoding
    try:
        args.decoding = 'greedy'
        gen_config_close = build_generation_config(args, tokenizer)
        logger.info('Building greedy config for close questions', force=(rank == 0))
    finally:
        args.decoding = _decoding_backup

    # collect samples on rank 0
    all_samples_to_process = []
    if rank == 0:
        logger.info("Collecting *_test.json samples...", force=True)
        json_files = sorted(glob.glob(osp.join(args.data_dir, '*_test.json')))
        if not json_files:
            raise FileNotFoundError(f'No *_test.json found in {args.data_dir}')
        for jfpath in json_files:
            with open(jfpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            samples = data if isinstance(data, list) else [data]
            for sample in samples:
                sample['__source_file__'] = osp.basename(jfpath)
                all_samples_to_process.append(sample)
        if args.debug_max_samples is not None:
            all_samples_to_process = all_samples_to_process[:int(args.debug_max_samples)]
        logger.info(f"Collected {len(all_samples_to_process)} total samples.", force=True)

    # broadcast to all ranks
    if world_size > 1:
        dist.barrier()
        objects = [all_samples_to_process]
        dist.broadcast_object_list(objects, src=0)
        all_samples_to_process = objects[0]

    # shard samples
    samples_for_this_rank = all_samples_to_process[rank::world_size]
    logger.info(f"Rank {rank} has {len(samples_for_this_rank)} samples.", force=True)

    rows, times = [], []
    profile_rows = []
    trace_saved = False
    flops_recorded = False
    model_device = next(model.parameters()).device
    rng = _rng(3407)

    for i, sample in enumerate(samples_for_this_rank):
        source_file = sample.get('__source_file__', 'unknown_file')

        # choose decoding config based on filename
        is_close_question = ('choice' in source_file) or ('TF_question' in source_file)
        if is_close_question:
            selected_gen_config = gen_config_close
        else:
            selected_gen_config = gen_config_open

        processing_msg = f"Processing {i}/{len(samples_for_this_rank)} on rank {rank} (src: {source_file})"

        sid = str(sample.get('id', ''))
        image_field = sample.get('image') or sample.get('images') or ''
        conv = sample.get('conversations', [])
        human_msgs = [c for c in conv if c.get('from') == 'human']
        raw_prompt = human_msgs[0]['value'] if human_msgs else '<image>\nDescribe the slide.'

        def _get_ref_answer(conversations):
            if not conversations:
                return ""
            priority_roles = ('gpt', 'assistant', 'answer', 'ref', 'label')
            for role in priority_roles:
                vals = [c.get('value', '') for c in conversations if str(c.get('from', '')).lower() == role]
                if vals:
                    return vals[-1]
            return ""

        ref_answer = _get_ref_answer(conv)

        try:
            feats, coords, feat_path, feature_backend = load_features_and_coords(
                sample=sample, args=args, rng=rng, logger=logger)
        except Exception as e:
            logger.warn(f'[Rank {rank}] ! skip {sid}: {e}', force=True)
            continue

        # to device; keep feats as float32, model will cast internally
        if dtype == 'auto':
            feats = feats.to(device=device)
        else:
            feats = feats.to(device=device, dtype=dtype)
        coords = coords.to(device=device, dtype=torch.float32)

        # tokenize prompt with explicit IMAGE_TOKEN_INDEX inserted
        input_ids = build_input_ids_from_prompt(tokenizer, raw_prompt, prompt_template).to(device).unsqueeze(0)

        # build batch and let model prepare multimodal inputs internally
        batch = {
            'input_ids': input_ids,
            'pixel_values': feats.unsqueeze(0),     # [1, L, C]
            'coords': coords.unsqueeze(0),         # [1, L, 2]
            'attention_mask': None,
            'position_ids': None,
            'past_key_values': None,
        }

        prof_cm = nullcontext()
        profiler_instance = None
        total_flops = None
        profile_enabled = bool(args.profile_infer)
        measure_gpu_mem = bool(profile_enabled and torch.cuda.is_available()
                               and model_device.type == 'cuda')
        resampler_mem_alloc = 0
        resampler_mem_reserved = 0
        alloc_before_prepare = reserved_before_prepare = None
        if measure_gpu_mem:
            try:
                torch.cuda.synchronize(model_device)
            except Exception:
                pass
            try:
                alloc_before_prepare = torch.cuda.memory_allocated(model_device)
                reserved_before_prepare = torch.cuda.memory_reserved(model_device)
            except Exception:
                alloc_before_prepare = reserved_before_prepare = None

        # prepare inputs and generate
        inputs = model._prepare_multimodal_inputs(batch)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if measure_gpu_mem:
            try:
                torch.cuda.synchronize(model_device)
                alloc_after_prepare = torch.cuda.memory_allocated(model_device)
                reserved_after_prepare = torch.cuda.memory_reserved(model_device)
                if alloc_before_prepare is not None:
                    delta_alloc = alloc_after_prepare - alloc_before_prepare
                    resampler_mem_alloc = max(0, int(delta_alloc))
                if reserved_before_prepare is not None:
                    delta_reserved = reserved_after_prepare - reserved_before_prepare
                    resampler_mem_reserved = max(0, int(delta_reserved))
            except Exception:
                resampler_mem_alloc = resampler_mem_reserved = 0
            try:
                torch.cuda.reset_peak_memory_stats(model_device)
            except Exception:
                pass
        elif profile_enabled and torch.cuda.is_available() and model_device.type == 'cuda':
            try:
                torch.cuda.reset_peak_memory_stats(model_device)
            except Exception:
                pass
        need_profiler = (
            profile_enabled and (
                (args.profile_trace and not trace_saved and rank == 0) or
                (args.profile_flops and not flops_recorded)
            )
        )
        if need_profiler:
            try:
                import torch.profiler as profiler
                activities = [profiler.ProfilerActivity.CPU]
                if torch.cuda.is_available():
                    activities.append(profiler.ProfilerActivity.CUDA)
                prof_cm = profiler.profile(
                    activities=activities,
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=False,
                    with_flops=bool(args.profile_flops),
                )
                profiler_instance = prof_cm
            except Exception as exc:
                logger.warn(f'Profiler setup failed ({exc}), continuing without trace.', force=True)
                profiler_instance = None
                prof_cm = nullcontext()

        t0 = time.time()
        # If open-question and multi-sample requested, override config for sampling
        use_multi = (selected_gen_config is gen_config_open) and (args.open_multi_num_return_sequences and args.open_multi_num_return_sequences > 1)
        cfg_to_use = make_open_multi_config(selected_gen_config, args) if use_multi else selected_gen_config

        with prof_cm as prof:
            # Dynamically adjust router_top_m if requested
            router_top_m_used = None
            if args.router_top_m_auto and getattr(model, 'router', None) is not None:
                if is_close_question:
                    preferred = args.router_top_m_close
                else:
                    preferred = args.router_top_m_open
                if preferred is None:
                    preferred = args.router_top_m if args.router_top_m is not None else base_router_top_m
                if preferred is not None:
                    preferred = int(preferred)
                    if getattr(model, 'router_top_m', None) != preferred:
                        model.router_top_m = preferred
                    router_top_m_used = preferred
                else:
                    router_top_m_used = getattr(model, 'router_top_m', None)
            else:
                router_top_m_used = getattr(model, 'router_top_m', None)
            if rank == 0:
                log_msg = processing_msg
                if args.router_top_m_auto and router_top_m_used is not None:
                    typ = "close" if is_close_question else "open"
                    log_msg += f" | router_top_m={router_top_m_used} ({typ})"
                logger.info(log_msg, force=True)

            out_ids = llm.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                inputs_embeds=inputs['inputs_embeds'],
                generation_config=cfg_to_use,
                bos_token_id=tokenizer.bos_token_id,
                stopping_criteria=stop_criteria,
            )
            if profiler_instance is not None:
                try:
                    profiler_instance.step()
                except Exception:
                    pass
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.time() - t0
        peak_alloc = peak_reserved = None
        peak_alloc_total = peak_reserved_total = None
        if profile_enabled and torch.cuda.is_available() and model_device.type == 'cuda':
            try:
                peak_alloc_total = torch.cuda.max_memory_allocated(model_device)
                peak_reserved_total = torch.cuda.max_memory_reserved(model_device)
                peak_alloc = peak_alloc_total
                peak_reserved = peak_reserved_total
                if resampler_mem_alloc:
                    peak_alloc = max(0, int(peak_alloc) - resampler_mem_alloc)
                if resampler_mem_reserved:
                    peak_reserved = max(0, int(peak_reserved) - resampler_mem_reserved)
            except Exception:
                peak_alloc = peak_reserved = None
                peak_alloc_total = peak_reserved_total = None
        if profiler_instance is not None:
            if args.profile_flops:
                try:
                    total_flops = 0
                    for evt in profiler_instance.key_averages():
                        if hasattr(evt, 'flops') and evt.flops is not None:
                            total_flops += evt.flops
                except Exception:
                    total_flops = None
                flops_recorded = True
            if args.profile_trace and not trace_saved:
                trace_dir = args.profile_out if args.profile_out else (args.out_jsonl or args.out_csv or '.')
                if trace_dir.endswith('.json') or trace_dir.endswith('.jsonl') or trace_dir.endswith('.csv'):
                    trace_dir = os.path.dirname(trace_dir) or '.'
                os.makedirs(trace_dir, exist_ok=True)
                trace_path = os.path.join(trace_dir, f'profile_trace_rank{rank}_sample{i}.json')
                try:
                    profiler_instance.export_chrome_trace(trace_path)
                    logger.info(f'Profiler trace saved to {trace_path}', force=True)
                    trace_saved = True
                except Exception as exc:
                    logger.warn(f'Failed to save profiler trace ({exc})', force=True)
        profiler_instance = None

        # Decode possibly multiple sequences, then simple re-rank if needed
        num_ret = int(getattr(cfg_to_use, 'num_return_sequences', 1) or 1)
        candidates = []
        if out_ids.dim() == 2 and out_ids.size(0) >= 1:
            # HF returns (num_return_sequences, T) when batch_size==1
            seq_count = out_ids.size(0)
            for j in range(seq_count):
                txt = tokenizer.decode(out_ids[j])
                txt = _clean_text_tail(txt, use_vicuna)
                candidates.append(txt)
        else:
            txt = tokenizer.decode(out_ids[0])
            txt = _clean_text_tail(txt, use_vicuna)
            candidates = [txt]

        if len(candidates) > 1:
            best_idx = rerank_candidates(tokenizer, candidates)
            out_text = candidates[best_idx]
        else:
            best_idx = 0
            out_text = candidates[0]

        if args.per_sample_log:
            if len(candidates) > 1:
                logger.info(f'[Rank {rank}, {sid}] {dt:.3f}s → chose {best_idx+1}/{len(candidates)}: {out_text[:120].strip()}{"..." if len(out_text) > 120 else ""}')
            else:
                logger.info(f'[Rank {rank}, {sid}] {dt:.3f}s → {out_text[:120].strip()}{"..." if len(out_text) > 120 else ""}')

        times.append(dt)
        if profile_enabled:
            input_ids_tensor = inputs.get('input_ids')
            in_tokens = int(input_ids_tensor.size(-1)) if input_ids_tensor is not None else None
            out_tokens = int(out_ids.shape[-1]) if out_ids is not None else None
            decode_mode = 'sampling' if getattr(cfg_to_use, 'do_sample', False) else 'greedy'
            profile_rows.append({
                'file': source_file,
                'id': sid,
                'rank': rank,
                'dt_sec': dt,
                'input_tokens': in_tokens,
                'output_tokens': out_tokens,
                'gpu_peak_alloc_bytes': peak_alloc,
                'gpu_peak_reserved_bytes': peak_reserved,
                'gpu_peak_alloc_bytes_total': peak_alloc_total,
                'gpu_peak_reserved_bytes_total': peak_reserved_total,
                'router_top_m': router_top_m_used,
                'decoding': decode_mode,
                'num_return_sequences': getattr(cfg_to_use, 'num_return_sequences', 1),
                'total_flops': total_flops,
            })
        rec = dict(
            file=source_file, id=sid,
            image=image_field if isinstance(image_field, str) else image_field[0],
            feature_path=feat_path, prompt=raw_prompt, output=out_text,
            ref_answer=ref_answer, gen_time_sec=dt, feature_backend=feature_backend
        )
        if args.save_all_candidates and len(candidates) > 1:
            rec['outputs_all'] = candidates
            rec['chosen_index'] = int(best_idx)
        rows.append(rec)

    # gather & save
    if world_size > 1:
        dist.barrier()

    all_rows = gather_results(rows, world_size)
    all_times = gather_results(times, world_size)
    all_profiles = gather_results(profile_rows, world_size)

    if rank == 0:
        if args.out_jsonl:
            os.makedirs(osp.dirname(args.out_jsonl) or '.', exist_ok=True)
            with open(args.out_jsonl, 'w', encoding='utf-8') as jf:
                for rec in all_rows:
                    jf.write(json.dumps(rec, ensure_ascii=False) + '\n')

        os.makedirs(osp.dirname(args.out_csv) or '.', exist_ok=True)
        pd.DataFrame(all_rows).to_csv(args.out_csv, index=False)

        if all_times:
            logger.info(f'Average generation time over {len(all_times)} samples: {np.mean(all_times):.4f}s', force=True)
        logger.info(f'Wrote {len(all_rows)} results to: {args.out_csv}', force=True)
        if args.out_jsonl:
            logger.info(f'Wrote: {args.out_jsonl}', force=True)
        if args.profile_out and all_profiles:
            os.makedirs(os.path.dirname(args.profile_out) or '.', exist_ok=True)
            with open(args.profile_out, 'w', encoding='utf-8') as pf:
                json.dump(all_profiles, pf, ensure_ascii=False, indent=2)
            logger.info(f'Profiling stats written to: {args.profile_out}', force=True)

    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
