# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from .feature_mae import FeatureNPYDataset, FeaturePTDataset, collate_feature_tokens
from .huggingface import process_hf_dataset
from .slidechat_features import SlideChatFeatureDataset

# ignore FutureWarning in hf datasets
warnings.simplefilter(action='ignore', category=FutureWarning)

__all__ = [
    'process_hf_dataset', 'FeatureNPYDataset', 'FeaturePTDataset',
    'collate_feature_tokens', 'SlideChatFeatureDataset'
]
