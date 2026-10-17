# Copyright (c) OpenMMLab. All rights reserved.
import os

try:
    from mmengine.utils import digit_version  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    def digit_version(version_str: str) -> tuple:
        """Minimal fallback so core utilities work without mmengine installed."""

        parts = []
        for token in version_str.split('.'):
            num = ''
            for ch in token:
                if ch.isdigit():
                    num += ch
                else:
                    break
            parts.append(int(num) if num else 0)
        while len(parts) < 3:
            parts.append(0)
        return tuple(parts[:3])

from .entry_point import cli
from .version import __version__, version_info

HF_CEPH_HUB = os.getenv('HF_CEPH_HUB', '')
HF_USE_CEPH = os.getenv('HF_USE_CEPH', 0) or HF_CEPH_HUB != ''
DS_CEPH_DIR = os.getenv('DS_CEPH_DIR', None)
if HF_USE_CEPH:
    from .utils.fileio import (patch_hf_auto_from_pretrained,
                               patch_hf_save_pretrained)
    patch_hf_auto_from_pretrained(HF_CEPH_HUB)
    patch_hf_save_pretrained()

if DS_CEPH_DIR:
    from .utils.fileio import patch_deepspeed_engine
    patch_deepspeed_engine()

__all__ = [
    '__version__', 'version_info', 'digit_version', 'cli', 'HF_USE_CEPH',
    'DS_CEPH_DIR'
]
