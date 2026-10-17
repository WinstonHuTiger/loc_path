import csv
import json
import os
import random
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Dict

import numpy as np
import torch
from torch.utils.data import Dataset


def _gather_feature_files(root: str, suffix: str) -> List[str]:
    files = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.endswith(suffix):
                files.append(os.path.join(dirpath, name))
    files.sort()
    return files


def _gather_pt_feature_files(root: str, filename: str) -> List[str]:
    files = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name == filename:
                files.append(os.path.join(dirpath, name))
    files.sort()
    return files


def _sample_indices(total: int, limit: int, seed: int) -> np.ndarray:
    if total <= limit:
        return np.arange(total)
    rng = np.random.RandomState(seed)
    return np.sort(rng.choice(total, size=limit, replace=False))


def _linspace_sample_indices(total: int, limit: int,
                             rng: np.random.RandomState) -> np.ndarray:
    if limit <= 0:
        return np.array([], dtype=np.int64)
    if total <= limit:
        return np.arange(total)
    edges = np.linspace(0, total, num=limit + 1, dtype=int)
    picks: List[int] = []
    used = set()
    for start, end in zip(edges[:-1], edges[1:]):
        if end <= start:
            continue
        idx = int(rng.randint(start, end))
        picks.append(idx)
        used.add(idx)
    if len(picks) < limit:
        pool = np.setdiff1d(np.arange(total), np.array(list(used), dtype=int),
                            assume_unique=True)
        if pool.size > 0:
            extra = rng.choice(pool, size=min(limit - len(picks), pool.size),
                               replace=False)
            picks.extend(int(x) for x in np.atleast_1d(extra))
    picks = np.array(sorted(picks), dtype=np.int64)
    if picks.size == 0:
        return _sample_indices(total, limit, rng.randint(0, 2**31 - 1))
    return picks


def _flatten_paths(value) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _flatten_paths(item)


def _load_include_tokens(json_path: str,
                         include_key: str) -> List[str]:
    if not json_path:
        return []
    with open(json_path, 'r', encoding='utf-8') as f:
        if json_path.endswith('.jsonl'):
            items = [json.loads(line) for line in f if line.strip()]
        else:
            data = json.load(f)
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict):
                items = data.get('data') if isinstance(data.get('data'),
                                                       list) else [data]
            else:
                raise ValueError(
                    f'Unsupported JSON structure in {json_path}: {type(data)}')

    tokens: List[str] = []
    for item in items:
        paths = item
        if include_key and isinstance(item, dict):
            paths = item.get(include_key, [])
        for path in _flatten_paths(paths):
            base = os.path.basename(path)
            stem, _ = os.path.splitext(base)
            tokens.append(base)
            tokens.append(stem)
    return tokens


def _load_unwanted_prefixes(csv_path: Optional[str]) -> List[str]:
    if not csv_path or not os.path.exists(csv_path):
        return []
    prefixes: List[str] = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            prefixes.append(row[0].strip())
    return prefixes


@dataclass
class FeatureSamplerConfig:
    max_tokens: Optional[int] = None
    sample_seed: int = 3407
    strategy: str = 'random'


class FeatureNPYDataset(Dataset):
    """Loads feature tokens stored as .npy arrays.

    Args:
        Features are typically stored as ``(N_tokens, D_in)``. For slide-level
        embeddings (e.g., pooled features) stored as ``(D_in,)``, the array is
        treated as a single token (``N_tokens=1``).

        include_json: Optional path to a JSON/JSONL file whose entries list
            images to keep. The specified ``include_key`` (default: ``image``)
            is read from each entry; matching is case-insensitive and checks
            whether any listed basename/stem appears in the feature file path.
        include_key: Key used to extract image paths from each include-json
            entry. Ignored when the entry itself is already a string/list.
        unwanted_prefix_csv: Optional CSV file (single column) containing
            substrings; any feature path containing one of these prefixes is
            filtered out.
        use_memmap: When True, feature/coord arrays are memory-mapped so only
            the required tokens are paged into RAM, reducing disk thrash for
            large slides.
        length_cache_path: Optional JSON file that stores per-feature token
            counts. Populating this cache lets the dataset compute sampling
            indices without reopening files in each worker.
    """

    def __init__(
        self,
        root: str,
        feature_suffix: str = '_features.npy',
        coord_suffix: str = '_coords.npy',
        max_samples: Optional[int] = None,
        shuffle_index: bool = False,
        index_seed: int = 0,
        dtype: str = 'float32',
        sampler_cfg: Optional[dict] = None,
        require_coords: bool = False,
        include_json: Optional[str] = None,
        include_key: str = 'image',
        unwanted_prefix_csv: Optional[str] = None,
        use_memmap: bool = True,
        length_cache_path: Optional[str] = None,
    ):
        super().__init__()
        self.root = root
        self.feature_suffix = feature_suffix
        self.coord_suffix = coord_suffix
        self.dtype = np.dtype(dtype)
        self.require_coords = require_coords
        self.use_memmap = use_memmap
        self.length_cache_path = length_cache_path

        files = _gather_feature_files(root, feature_suffix)
        if not files:
            raise FileNotFoundError(
                f'No files with suffix {feature_suffix} found under {root}')
        include_tokens = {
            token.lower()
            for token in _load_include_tokens(include_json,
                                              include_key) if token
        } if include_json else set()
        if include_tokens:
            files = [
                path for path in files
                if any(token in path.lower() for token in include_tokens)
            ]
            if not files:
                raise FileNotFoundError(
                    f'No feature files matched include list from {include_json}'
                )

        unwanted_prefixes = {
            p.lower()
            for p in _load_unwanted_prefixes(unwanted_prefix_csv) if p
        }
        if unwanted_prefixes:
            files = [
                path for path in files
                if not any(pref in path.lower() for pref in unwanted_prefixes)
            ]
            if not files:
                raise FileNotFoundError(
                    'All feature files filtered out by unwanted_prefix_csv')

        if shuffle_index:
            rng = random.Random(index_seed)
            rng.shuffle(files)

        if max_samples is not None:
            files = files[:max_samples]

        self.files = files
        self.sampler_cfg = FeatureSamplerConfig(
            **sampler_cfg) if sampler_cfg is not None else FeatureSamplerConfig(
            )
        if self.sampler_cfg.strategy not in ('random', 'head', 'linspace'):
            raise ValueError(
                f"Unsupported sampler strategy {self.sampler_cfg.strategy}")

        self._length_cache: Optional[Dict[str, int]] = None
        if self.length_cache_path:
            self._length_cache = {}
            if os.path.exists(self.length_cache_path):
                with open(self.length_cache_path, 'r', encoding='utf-8') as f:
                    stored = json.load(f)
                    self._length_cache.update(
                        {k: int(v)
                         for k, v in stored.items()})
            updated = False
            load_kwargs = {'mmap_mode': 'r'} if self.use_memmap else {}
            for path in self.files:
                key = self._cache_key(path)
                if key in self._length_cache:
                    continue
                arr = np.load(path, **load_kwargs)
                if arr.ndim == 1:
                    token_count = 1
                elif arr.ndim == 2:
                    token_count = int(arr.shape[0])
                else:
                    raise ValueError(
                        f'Expect feature tensor to be 1D/2D, got {arr.shape} '
                        f'from {path}')
                self._length_cache[key] = token_count
                updated = True
            if updated:
                self._persist_length_cache()

    def _cache_key(self, path: str) -> str:
        return os.path.relpath(path, self.root)

    def _persist_length_cache(self):
        if not self.length_cache_path or self._length_cache is None:
            return
        os.makedirs(os.path.dirname(self.length_cache_path), exist_ok=True)
        tmp_path = self.length_cache_path + '.tmp'
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(self._length_cache, f)
        os.replace(tmp_path, self.length_cache_path)

    def __len__(self) -> int:
        return len(self.files)

    def _choose_indices(self, total_tokens: int,
                        sample_seed: int) -> Optional[np.ndarray]:
        limit = self.sampler_cfg.max_tokens
        if limit is None or total_tokens <= limit:
            return None
        if self.sampler_cfg.strategy == 'head':
            indices = np.arange(min(limit, total_tokens))
        elif self.sampler_cfg.strategy == 'linspace':
            rng = np.random.RandomState(sample_seed)
            indices = _linspace_sample_indices(total_tokens, limit, rng)
        else:
            indices = _sample_indices(total_tokens, limit, sample_seed)
        return indices

    def __getitem__(self, index: int):
        feature_path = self.files[index]
        load_kwargs = {'mmap_mode': 'r'} if self.use_memmap else {}
        total_tokens = None
        tokens_array = None
        cache_key = self._cache_key(feature_path)
        if self._length_cache is not None:
            total_tokens = self._length_cache.get(cache_key)
        if total_tokens is None:
            tokens_array = np.load(feature_path, **load_kwargs)
            if tokens_array.ndim == 1:
                tokens_array = tokens_array[None, :]
            if tokens_array.ndim != 2:
                raise ValueError(
                    f'Expect feature tensor to be 2D [N,D], got {tokens_array.shape} '
                    f'from {feature_path}')
            total_tokens = int(tokens_array.shape[0])
            if self._length_cache is not None and cache_key not in self._length_cache:
                self._length_cache[cache_key] = total_tokens
                self._persist_length_cache()
        indices = self._choose_indices(total_tokens,
                                       self.sampler_cfg.sample_seed + index)
        if tokens_array is None:
            tokens_array = np.load(feature_path, **load_kwargs)
        if tokens_array.ndim == 1:
            tokens_array = tokens_array[None, :]
        if tokens_array.ndim != 2:
            raise ValueError(
                f'Expect feature tensor to be 2D [N,D], got {tokens_array.shape} '
                f'from {feature_path}')
        if indices is not None:
            tokens = tokens_array[indices]
            sel_idx = indices
        else:
            tokens = tokens_array
            sel_idx = None
        tokens = np.asarray(tokens, dtype=self.dtype)
        if not tokens.flags.writeable:
            tokens = np.array(tokens, dtype=self.dtype, copy=True)
        coords = None
        if self.coord_suffix:
            coord_path = feature_path.replace(self.feature_suffix,
                                              self.coord_suffix)
            if os.path.exists(coord_path):
                coords = np.load(coord_path, **load_kwargs)
                if sel_idx is not None:
                    coords = coords[sel_idx]
                if coords.ndim == 1:
                    coords = coords[None, :]
                coords = np.asarray(coords, dtype=self.dtype)
                if not coords.flags.writeable:
                    coords = np.array(coords, dtype=self.dtype, copy=True)
        if self.require_coords and coords is None:
            raise FileNotFoundError(
                f'coords file missing for {feature_path} with suffix '
                f'{self.coord_suffix}')

        item = {
            'tokens': torch.from_numpy(tokens).float(),
            'length': tokens.shape[0],
            'path': feature_path,
        }
        if coords is not None:
            item['coords'] = torch.from_numpy(coords).float()
        return item


def _resolve_torch_dtype(dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        attr = dtype.lower()
        if not hasattr(torch, attr):
            raise AttributeError(f'Unsupported torch dtype string: {dtype}')
        resolved = getattr(torch, attr)
        if not isinstance(resolved, torch.dtype):
            raise TypeError(f'Attribute {dtype} is not a torch.dtype')
        return resolved
    raise TypeError(f'Unsupported dtype specification: {dtype}')


class FeaturePTDataset(Dataset):
    """Loads feature tokens stored as torch .pt files emitted by prov-gigapath.

    Each file is expected to contain a dict with keys:
      - embeddings: Tensor [N_tokens, D_in]
      - coords (optional): Tensor [N_tokens, C]
    """

    def __init__(
        self,
        root: str,
        file_name: str = 'tile_embeddings.pt',
        max_samples: Optional[int] = None,
        shuffle_index: bool = False,
        index_seed: int = 0,
        dtype: str = 'float32',
        sampler_cfg: Optional[dict] = None,
        require_coords: bool = False,
        include_json: Optional[str] = None,
        include_key: str = 'image',
        unwanted_prefix_csv: Optional[str] = None,
        length_cache_path: Optional[str] = None,
    ):
        super().__init__()
        self.root = root
        self.file_name = file_name
        self.dtype = _resolve_torch_dtype(dtype)
        self.require_coords = require_coords
        self.length_cache_path = length_cache_path

        files = _gather_pt_feature_files(root, file_name)
        if not files:
            raise FileNotFoundError(
                f'No files named {file_name} found under {root}')

        include_tokens = {
            token.lower()
            for token in _load_include_tokens(include_json, include_key)
            if token
        } if include_json else set()
        if include_tokens:
            files = [
                path for path in files
                if any(token in path.lower() for token in include_tokens)
            ]
            if not files:
                raise FileNotFoundError(
                    f'No PT feature files matched include list from {include_json}'
                )

        unwanted_prefixes = {
            p.lower()
            for p in _load_unwanted_prefixes(unwanted_prefix_csv) if p
        }
        if unwanted_prefixes:
            files = [
                path for path in files
                if not any(pref in path.lower() for pref in unwanted_prefixes)
            ]
            if not files:
                raise FileNotFoundError(
                    'All PT feature files filtered out by unwanted_prefix_csv')

        if shuffle_index:
            rng = random.Random(index_seed)
            rng.shuffle(files)
        if max_samples is not None:
            files = files[:max_samples]

        self.files = files
        self.sampler_cfg = FeatureSamplerConfig(
            **sampler_cfg) if sampler_cfg is not None else FeatureSamplerConfig(
            )
        if self.sampler_cfg.strategy not in ('random', 'head', 'linspace'):
            raise ValueError(
                f"Unsupported sampler strategy {self.sampler_cfg.strategy}")

        self._length_cache: Optional[Dict[str, int]] = None
        if self.length_cache_path:
            self._length_cache = {}
            if os.path.exists(self.length_cache_path):
                with open(self.length_cache_path, 'r', encoding='utf-8') as f:
                    stored = json.load(f)
                    self._length_cache.update(
                        {k: int(v)
                         for k, v in stored.items()})
            updated = False
            for path in self.files:
                key = self._cache_key(path)
                if key in self._length_cache:
                    continue
                embeddings, _ = self._load_tensors(path)
                self._length_cache[key] = int(embeddings.shape[0])
                updated = True
            if updated:
                self._persist_length_cache()

    def _cache_key(self, path: str) -> str:
        return os.path.relpath(path, self.root)

    def _persist_length_cache(self):
        if not self.length_cache_path or self._length_cache is None:
            return
        os.makedirs(os.path.dirname(self.length_cache_path), exist_ok=True)
        tmp_path = self.length_cache_path + '.tmp'
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(self._length_cache, f)
        os.replace(tmp_path, self.length_cache_path)

    def _load_tensors(self, path: str) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        data = torch.load(path, map_location='cpu', weights_only= True)
        if 'embeddings' not in data:
            raise KeyError(f"'embeddings' key missing in {path}")
        embeddings = data['embeddings']
        if not torch.is_tensor(embeddings):
            embeddings = torch.as_tensor(embeddings)
        if embeddings.ndim != 2:
            raise ValueError(
                f'Expect embeddings to be 2D [N,D], got {embeddings.shape} from {path}'
            )
        coords = data.get('coords')
        if coords is not None and not torch.is_tensor(coords):
            coords = torch.as_tensor(coords)
        return embeddings, coords

    def __len__(self) -> int:
        return len(self.files)

    def _choose_indices(self, total_tokens: int,
                        sample_seed: int) -> Optional[np.ndarray]:
        limit = self.sampler_cfg.max_tokens
        if limit is None or total_tokens <= limit:
            return None
        if self.sampler_cfg.strategy == 'head':
            indices = np.arange(min(limit, total_tokens))
        elif self.sampler_cfg.strategy == 'linspace':
            rng = np.random.RandomState(sample_seed)
            indices = _linspace_sample_indices(total_tokens, limit, rng)
        else:
            indices = _sample_indices(total_tokens, limit, sample_seed)
        return indices

    def __getitem__(self, index: int):
        feature_path = self.files[index]
        cache_key = self._cache_key(feature_path)
        total_tokens = None
        if self._length_cache is not None:
            total_tokens = self._length_cache.get(cache_key)

        embeddings, coords = self._load_tensors(feature_path)
        if total_tokens is None:
            total_tokens = int(embeddings.shape[0])
            if self._length_cache is not None and cache_key not in self._length_cache:
                self._length_cache[cache_key] = total_tokens
                self._persist_length_cache()

        indices = self._choose_indices(total_tokens,
                                       self.sampler_cfg.sample_seed + index)
        if indices is not None:
            idx_tensor = torch.as_tensor(indices, dtype=torch.long)
            embeddings = embeddings.index_select(0, idx_tensor)
            if coords is not None:
                coords = coords.index_select(0, idx_tensor)

        tokens = embeddings.to(dtype=self.dtype)
        if not tokens.is_contiguous():
            tokens = tokens.contiguous()
        coords_out = None
        if coords is not None:
            coords_out = coords.to(dtype=self.dtype if coords.dtype.is_floating_point else torch.float32)
            if not coords_out.is_contiguous():
                coords_out = coords_out.contiguous()

        if self.require_coords and coords_out is None:
            raise FileNotFoundError(
                f'coords missing for {feature_path} but require_coords=True')

        item = {
            'tokens': tokens.float(),
            'length': tokens.shape[0],
            'path': feature_path,
        }
        if coords_out is not None:
            item['coords'] = coords_out.float()
        return item


def collate_feature_tokens(batch, max_tokens: Optional[int] = None):
    lengths = [item['tokens'].shape[0] for item in batch]
    dim = batch[0]['tokens'].shape[1]
    max_len = max(lengths)
    if max_tokens is not None:
        max_len = min(max_len, max_tokens)
    batch_size = len(batch)
    tokens = torch.empty(batch_size,
                         max_len,
                         dim,
                         dtype=batch[0]['tokens'].dtype)
    valid_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    coords = None
    if 'coords' in batch[0]:
        coord_dim = batch[0]['coords'].shape[-1]
        coords = torch.empty(batch_size,
                             max_len,
                             coord_dim,
                             dtype=batch[0]['coords'].dtype)
    paths = []
    for i, item in enumerate(batch):
        n = min(item['tokens'].shape[0], max_len)
        tokens[i, :n] = item['tokens'][:n]
        if n < max_len:
            tokens[i, n:] = 0
        valid_mask[i, :n] = True
        if coords is not None:
            coords[i, :n] = item['coords'][:n]
            if n < max_len:
                coords[i, n:] = 0
        paths.append(item['path'])
    out = {'tokens': tokens, 'valid_mask': valid_mask, 'paths': paths}
    if coords is not None:
        out['coords'] = coords
    return out
