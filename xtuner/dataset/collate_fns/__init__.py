# Copyright (c) OpenMMLab. All rights reserved.
from .default_collate_fn import default_collate_fn
from .slidechat_feature_collate import slidechat_feature_collate_fn

__all__ = ['default_collate_fn', 'slidechat_feature_collate_fn']
