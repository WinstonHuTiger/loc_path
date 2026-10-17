import json
import os
from typing import Dict, List, Optional, Sequence

import h5py
import numpy as np
import torch
from datasets import Dataset as HFDataset, DatasetDict
from torch.utils.data import Dataset

from xtuner.dataset.feature_mae import (FeatureSamplerConfig,
                                        _linspace_sample_indices,
                                        _sample_indices)
from xtuner.dataset.huggingface import process_hf_dataset
from mmengine import print_log


def _load_json_or_jsonl(path: str) -> List[Dict]:
    if path.endswith('.json'):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def _flatten_path(value) -> Sequence[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        paths: List[str] = []
        for item in value:
            paths.extend(_flatten_path(item))
        return paths
    return []


def _load_unwanted_prefixes(csv_path: Optional[str]) -> Sequence[str]:
    prefixes: List[str] = []
    if not csv_path:
        return prefixes
    if not os.path.exists(csv_path):
        print_log(f'unwanted_prefix_csv {csv_path} not found; skipping.', logger='current')
        return prefixes
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            for line in f:
                token = line.strip()
                if token:
                    prefixes.append(token)
    except Exception as exc:
        print_log(f'Failed to read unwanted_prefix_csv {csv_path}: {exc}', logger='current')
    return prefixes


class SlideChatFeatureDataset(Dataset):
    """Dataset that aligns LLaVA-style text processing with fast feature loading.

    Args:
        use_memmap: When True, load ``.npy`` feature tensors via memory mapping so
            only the required rows are paged into RAM.
        cache_npy_arrays: When True, keep opened ``.npy`` arrays in a per-worker
            cache to avoid re-opening the same files across successive epochs
            (trading host RAM for lower I/O latency).
    """

    _SUFFIX_TO_SUBDIR = {
        '.csv': 'csv_files',
        '.pt': 'pt_files',
        '.h5': 'h5_files',
        '.hdf5': 'h5_files',
        '.npy': 'npy_files',
        '.npz': 'npy_files',
    }

    def __init__(
        self,
        data_path: str,
        tokenizer,
        dataset_map_fn=None,
        template_map_fn=None,
        max_length: int = 2048,
        per_image_length: Optional[int] = None,
        with_image_token: bool = True,
        map_num_proc: int = 8,
        max_dataset_length: Optional[int] = None,
        image_feature_prefix: Optional[str] = None,
        feature_root: Optional[str] = None,
        image_feature_suffix: str = '.h5',
        identifier: str = '',
        sample_strategy: str = 'linspace',
        sampler_cfg: Optional[dict] = None,
        feature_sample_seed: int = 3407,
        feature_key: str = 'features',
        coord_key: str = 'coords',
        length_cache_path: Optional[str] = None,
        require_coords: bool = False,
        unwanted_prefix_csv: Optional[str] = None,
        use_memmap: bool = True,
        cache_npy_arrays: bool = True,
        swap_xy: bool = False,
    ):
        super().__init__()
        raw_items = _load_json_or_jsonl(data_path)
        if not raw_items:
            raise ValueError(f'No samples found in {data_path}')

        unwanted = {
            p.lower()
            for p in _load_unwanted_prefixes(unwanted_prefix_csv)
        } if unwanted_prefix_csv else set()

        filtered = []
        removed = 0
        for item in raw_items:
            img_fields = item.get('image') or item.get('images')
            paths = _flatten_path(img_fields)
            feature_path = item.get('feature_path')
            keep = True
            candidates = paths
            if feature_path:
                candidates = candidates + [feature_path]
            lowered = [p.lower() for p in candidates]
            if unwanted and any(any(pref in path for pref in unwanted) for path in lowered):
                keep = False
            if keep:
                filtered.append(item)
            else:
                removed += 1

        if removed:
            print_log(f'Filtered out {removed} samples via unwanted_prefix_csv.', logger='current')

        hf_dataset = HFDataset.from_list(filtered)
        dataset_dict = DatasetDict({'train': hf_dataset})
        processed = process_hf_dataset(
            dataset=dataset_dict,
            tokenizer=tokenizer,
            max_length=max_length,
            dataset_map_fn=dataset_map_fn,
            template_map_fn=template_map_fn,
            max_dataset_length=max_dataset_length,
            split='train',
            remove_unused_columns=False,
            with_image_token=with_image_token,
            map_num_proc=map_num_proc,
            pack_to_max_length=False,
            )
        if isinstance(processed, DatasetDict):
            processed = processed['train']
        self.text_dataset = processed

        self.feature_root = feature_root or image_feature_prefix or ''
        self.image_feature_suffix = image_feature_suffix
        self.identifier = identifier
        self.feature_key = feature_key
        self.coord_key = coord_key
        self.require_coords = require_coords
        self.unwanted_prefixes = unwanted
        self.use_memmap = use_memmap
        self.cache_npy_arrays = cache_npy_arrays
        self._feature_cache: Optional[Dict[str, np.ndarray]] = (
            {} if self.cache_npy_arrays else None)
        self._coord_cache: Optional[Dict[str, np.ndarray]] = (
            {} if self.cache_npy_arrays else None)
        self.swap_xy = swap_xy

        if sampler_cfg is not None:
            self.sampler_cfg = FeatureSamplerConfig(**sampler_cfg)
        else:
            strategy_map = {
                'linspace': 'linspace',
                'random': 'random',
                'random_full': 'random',
                'head': 'head'
            }
            mapped = strategy_map.get(sample_strategy, 'linspace')
            self.sampler_cfg = FeatureSamplerConfig(
                max_tokens=per_image_length,
                strategy=mapped,
                sample_seed=feature_sample_seed)
        self._random_full = sample_strategy == 'random_full'
        self.length_cache_path = length_cache_path
        self._length_cache: Optional[Dict[str, int]] = None
        self.raw_items = filtered
        if self.length_cache_path:
            self._length_cache = {}
            if os.path.exists(self.length_cache_path):
                with open(self.length_cache_path, 'r', encoding='utf-8') as f:
                    stored = json.load(f)
                    self._length_cache.update(
                        {k: int(v) for k, v in stored.items()})

    def __len__(self) -> int:
        return len(self.text_dataset)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _cache_key(self, path: str) -> str:
        return os.path.relpath(path, self.feature_root) if self.feature_root else path

    def _persist_length_cache(self):
        if not self.length_cache_path or self._length_cache is None:
            return
        os.makedirs(os.path.dirname(self.length_cache_path), exist_ok=True)
        tmp_path = self.length_cache_path + '.tmp'
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(self._length_cache, f)
        os.replace(tmp_path, self.length_cache_path)

    def _parse_stub(self, image_path: str):
        norm = os.path.normpath(image_path)
        parts = norm.split(os.sep)
        if len(parts) < 2:
            fname = os.path.splitext(parts[-1])[0]
            tumor = fname.split('-')[0].lower() if '-' in fname else 'unknown'
            return tumor, fname
        tumor = parts[-2].lower()
        case = os.path.splitext(parts[-1])[0]
        return tumor, case

    def _feature_subdir(self) -> str:
        return self._SUFFIX_TO_SUBDIR.get(self.image_feature_suffix, '')

    def _resolve_feature_path(self, sample: Dict) -> str:
        if 'feature_path' in sample and sample['feature_path']:
            rel = sample['feature_path']
            if os.path.isabs(rel):
                return rel
            return os.path.join(self.feature_root, rel)

        image_field = sample.get('image') or sample.get('images')
        paths = _flatten_path(image_field)
        if not paths:
            raise KeyError('Sample does not contain feature_path or image field.')
        image_path = paths[0]
        if os.path.isabs(image_path) and os.path.exists(image_path):
            return image_path

        tumor, case = self._parse_stub(image_path)
        subdir = self._feature_subdir()
        parts = [self.feature_root]
        if tumor:
            parts.append(f'{tumor}{self.identifier}')
        if subdir:
            parts.append(subdir)
        filename = case + self.image_feature_suffix
        if self.image_feature_suffix == '.npy':
            parts.append(f'{case}_features.npy')
            candidate = os.path.join(*parts)
            if not os.path.exists(candidate):
                parts[-1] = filename
            return candidate if os.path.exists(candidate) else os.path.join(*parts)
        parts.append(filename)
        return os.path.join(*parts)

    def _resolve_h5_dataset(self, handle: h5py.File, key: str):
        if key in handle:
            return handle[key]
        first = next(iter(handle.keys()))
        return handle[first]

    def _ensure_npy_array(self,
                          path: str,
                          cache: bool = True,
                          is_coord: bool = False) -> np.ndarray:
        cache_dict: Optional[Dict[str, np.ndarray]]
        cache_dict = self._coord_cache if is_coord else self._feature_cache
        if cache_dict is not None and cache:
            cached = cache_dict.get(path)
            if cached is not None:
                return cached

        load_kwargs = {}
        if self.use_memmap:
            load_kwargs['mmap_mode'] = 'r'

        try:
            array = np.load(path, **load_kwargs)
        except ValueError as exc:
            if self.use_memmap and 'mmap_mode' in str(exc):
                array = np.load(path)
            else:
                raise

        if cache_dict is not None and cache:
            cache_dict[path] = array
        return array

    def _get_total_tokens(self,
                          path: str,
                          preloaded: Optional[np.ndarray] = None) -> int:
        if self._length_cache is not None:
            key = self._cache_key(path)
            cached = self._length_cache.get(key)
            if cached is not None:
                return cached
        if path.endswith('.npy'):
            array = preloaded if preloaded is not None else self._ensure_npy_array(
                path, cache=self.cache_npy_arrays)
            total = int(array.shape[0])
        elif path.endswith('.npz'):
            with np.load(path) as data:
                tokens = data[self.feature_key]
                total = int(tokens.shape[0])
        elif path.endswith(('.h5', '.hdf5')):
            with h5py.File(path, 'r') as f:
                dataset = self._resolve_h5_dataset(f, self.feature_key)
                total = int(dataset.shape[0])
        else:
            raise ValueError(f'Unsupported feature format: {path}')

        if self._length_cache is not None:
            key = self._cache_key(path)
            self._length_cache[key] = total
            self._persist_length_cache()
        return total

    def _choose_indices(self, total_tokens: int, seed: int) -> Optional[np.ndarray]:
        limit = self.sampler_cfg.max_tokens
        if limit is None or total_tokens <= limit:
            return None
        if self._random_full:
            rng = np.random.RandomState(seed)
            replace = total_tokens < limit
            idx = rng.choice(total_tokens, size=limit, replace=replace)
            return np.sort(idx.astype(np.int64))
        if self.sampler_cfg.strategy == 'head':
            return np.arange(min(limit, total_tokens), dtype=np.int64)
        if self.sampler_cfg.strategy == 'linspace':
            rng = np.random.RandomState(seed)
            return _linspace_sample_indices(total_tokens, limit, rng)
        return _sample_indices(total_tokens, limit, seed)

    def _load_feature_tokens(self, path: str, rng_seed: int):
        preloaded_npy: Optional[np.ndarray] = None
        if path.endswith('.npy'):
            preloaded_npy = self._ensure_npy_array(
                path, cache=self.cache_npy_arrays)

        total_tokens = self._get_total_tokens(path, preloaded=preloaded_npy)
        indices = self._choose_indices(total_tokens, rng_seed)

        if path.endswith('.npy'):
            array = preloaded_npy if preloaded_npy is not None else self._ensure_npy_array(
                path, cache=self.cache_npy_arrays)
            if array.ndim != 2:
                raise ValueError(f'Expect feature tensor to be 2D, got {array.shape}')
            tokens = array if indices is None else array[indices]
            coord_path = path
            if coord_path.endswith('_features.npy'):
                coord_path = coord_path.replace('_features.npy', '_coords.npy')
            else:
                coord_path = coord_path.replace('.npy', '_coords.npy')
            coords = None
            if os.path.exists(coord_path):
                coord_array = self._ensure_npy_array(
                    coord_path, cache=self.cache_npy_arrays, is_coord=True)
                if coord_array.ndim != 2:
                    raise ValueError(
                        f'Expect coord tensor to be 2D, got {coord_array.shape}')
                coords = coord_array if indices is None else coord_array[indices]
            elif self.require_coords:
                raise FileNotFoundError(
                    f'Coordinate data not found alongside features: {coord_path}')
        elif path.endswith('.npz'):
            npz = np.load(path)
            tokens = npz[self.feature_key]
            coords = npz.get(self.coord_key)
            if indices is not None:
                tokens = tokens[indices]
                if coords is not None:
                    coords = coords[indices]
            npz.close()
        else:
            with h5py.File(path, 'r') as f:
                dataset = self._resolve_h5_dataset(f, self.feature_key)
                if dataset.ndim != 2:
                    raise ValueError(f'Expect feature tensor to be 2D, got {dataset.shape}')
                if indices is not None:
                    tokens = dataset[indices]
                else:
                    tokens = dataset[()]
                coords = None
                if self.coord_key in f:
                    coord_ds = f[self.coord_key]
                    coords = coord_ds[indices] if indices is not None else coord_ds[()]
        tokens = np.asarray(tokens, dtype=np.float32)
        coords_tensor = None
        if coords is not None:
            coords = np.asarray(coords, dtype=np.float32)
            if self.swap_xy and coords.shape[-1] >= 2:
                coords = coords[..., [1, 0]]
            coords_tensor = torch.from_numpy(coords.copy())
        if self.require_coords and coords_tensor is None:
            raise FileNotFoundError(
                f'Coordinate data not found alongside features: {path}')
        return torch.from_numpy(tokens.copy()), coords_tensor

    # ------------------------------------------------------------------ #
    # Dataset API
    # ------------------------------------------------------------------ #
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self.text_dataset[idx]
        raw_record = self.raw_items[idx]
        feature_path = self._resolve_feature_path(raw_record)
        lowered_path = feature_path.lower()
        if self.unwanted_prefixes and any(pref in lowered_path for pref in self.unwanted_prefixes):
            raise KeyError(f'Feature path {feature_path} filtered by unwanted prefixes.')

        tokens, coords = self._load_feature_tokens(
            feature_path, self.sampler_cfg.sample_seed + idx)

        item = {
            'input_ids': record['input_ids'],
            'labels': record['labels'],
            'vision_tokens': tokens,
            'path': feature_path,
        }
        if coords is not None:
            item['vision_coords'] = coords
        if 'image' in raw_record:
            item['image'] = raw_record['image']
        if 'images' in raw_record:
            item['images'] = raw_record['images']
        if 'feature_path' in raw_record:
            item['feature_path'] = raw_record['feature_path']
        return item
