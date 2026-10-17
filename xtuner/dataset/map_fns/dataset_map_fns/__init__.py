# Copyright (c) OpenMMLab. All rights reserved.
from .llava_map_fn import llava_image_only_map_fn, llava_map_fn

DATASET_FORMAT_MAPPING = dict(
    llava_image_only=llava_image_only_map_fn,
    llava=llava_map_fn,
)

__all__ = ['llava_image_only_map_fn', 'llava_map_fn', 'DATASET_FORMAT_MAPPING']
