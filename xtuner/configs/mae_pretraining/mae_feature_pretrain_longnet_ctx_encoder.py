import os

# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.dataset import DefaultSampler
from mmengine.visualization import Visualizer, WandbVisBackend
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from torch.optim import AdamW

from xtuner.dataset import FeatureNPYDataset, FeaturePTDataset, collate_feature_tokens
from xtuner.engine.runner import TrainLoop
from xtuner.model import MAEFeaturePretrain
from xtuner.engine.hooks import TopKAnnealHook


# ----------------------------------------------------------------------
# Global settings
# ----------------------------------------------------------------------
seed = 2025
max_tokens = 61440
num_workers = 16
batch_size = 2
accumulative_counts = 4
max_iters = 100000
warmup_ratio = 0.03
mask_ratio = 0.7
save_interval = 8192  # iters
save_total_limit = 10

log_iterations = 40

lr=1e-4

betas = (0.9, 0.999)
# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Feature source (switch between legacy NPY features and new PT features)
# ----------------------------------------------------------------------

dataset_root = os.getenv(
    'LOCPATH_FEATURE_ROOT',
    os.getenv('SLIDECHAT_FEATURE_ROOT', '/path/to/feature_root'))
stage1_report_json = os.getenv(
    'LOCPATH_STAGE1_REPORT_JSON',
    os.getenv('SLIDECHAT_STAGE1_REPORT_JSON', '/path/to/stage1_report.json'))
unwanted_prefix_csv = (
    os.getenv('LOCPATH_UNWANTED_PREFIX_CSV')
    or os.getenv('SLIDECHAT_UNWANTED_PREFIX_CSV')
    or None
)
dataset_type = FeatureNPYDataset
dataset_extra_kwargs = dict(
    feature_suffix='_features.npy',
    coord_suffix='_coords.npy',
    use_memmap=True,
)
embedding_dim = 512


model = dict(
    type=MAEFeaturePretrain,
    d_in=embedding_dim,
    d_model=512,
    n_latents=256,
    num_heads=8,
    resampler_layers=2,
    resampler_topk=(128, 64),

    use_latent_self_attn=True,
    resampler_latent_drop = 0.1,

    resampler_context_mode='longnet',
    resampler_ctx_layers=2,
    
    resampler_ctx_longnet_segments=[1024, 2048, 4096],
    resampler_ctx_longnet_dilations=[[1, 2, 4], [4, 8, 16]],
    resampler_ctx_longnet_use_xpos=True,
    resampler_ctx_longnet_mlp_ratio=2.0,
    
    dec_depth=2,
    dec_mlp_ratio=4.0,
    mask_ratio=mask_ratio,
    
    resampler_dense_latents = 8, 
    
    
    
    reg_cfg=dict(
        attn_entropy=dict(enable=False, weight=1e-3, log_only=True),
        attn_decor=dict(enable=False, weight=5e-3, log_only= True),
        
        feature_decor=dict(enable=True, weight=1e-3),
        
        peak_var=dict(enable=False, weight=0.0, log_only=True),

        cover=dict(enable=True, weight=5e-4),
        
        coverage_stats_enable=True,
        
        cover_thresholds=(1e-4, 1e-3, 1e-2),
        
        cover_tau=1e-4,
        cover_tau_alpha=0.2,
        cover_tau_cap=1e-3,
        cover_tau_floor=1e-6,
    ),
    
    use_pos_embed=True,
    pos_embed_type='2d_fourier',
    fourier_num_bands=64,
    fourier_max_freq=10000.0,
)

# ----------------------------------------------------------------------
# Dataset & Dataloader
# ----------------------------------------------------------------------
train_dataset = dict(
    type=dataset_type,
    root=dataset_root,
    max_samples=None,  # set to an int for quicker experiments
    shuffle_index=True,
    index_seed=seed,
    dtype='float32',
    sampler_cfg=dict(
        max_tokens=max_tokens,
        sample_seed=seed,
        strategy='linspace',
    ),
    require_coords=True,
    include_json=stage1_report_json,
    include_key='image',
    unwanted_prefix_csv=unwanted_prefix_csv,
    **dataset_extra_kwargs,
)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    pin_memory=True,
    dataset=train_dataset,
    sampler=dict(type=DefaultSampler, shuffle=True),
    collate_fn=dict(type=collate_feature_tokens, max_tokens=max_tokens),
)

# ----------------------------------------------------------------------
# Optimizer & Scheduler
# ----------------------------------------------------------------------
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(type=AdamW, lr=lr,betas = betas,
                   weight_decay=0.05),
    accumulative_counts=accumulative_counts,
    clip_grad=dict(max_norm=1.0, error_if_nonfinite=True),
    # BF16 gradients do not need scaling; PyTorch 2.4 cannot unscale BF16.
    loss_scale=dict(enabled=False),
    dtype='bfloat16'
    )

_warmup_iters = int(max_iters * warmup_ratio)

param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-4,
        by_epoch=False,
        begin=0,
        end=max(1, _warmup_iters),
    ),
    dict(
        type=CosineAnnealingLR,
        eta_min=5e-7,
        by_epoch=False,
        begin=max(1, _warmup_iters),
        end=max_iters,
    ),
]

# ----------------------------------------------------------------------
# Runtime
# ----------------------------------------------------------------------
train_cfg = dict(type=TrainLoop, max_iters=max_iters)

default_hooks = dict(
    timer=dict(type=IterTimerHook),
    logger=dict(type=LoggerHook, interval=log_iterations, log_metric_by_epoch=False),
    param_scheduler=dict(type=ParamSchedulerHook),
    checkpoint=dict(
        type=CheckpointHook,
        by_epoch=False,
        interval=save_interval,
        max_keep_ckpts=save_total_limit,
    ),
    sampler_seed=dict(type=DistSamplerSeedHook),
)


custom_hooks = [
    dict(type=TopKAnnealHook,
         layers=[0, 1],           # 对 resampler 的第0/1层同时退火
         start=[128, 64],         # 起始 K
         end=[64, 32],            # 目标 K
         schedule='cosine',       # 或 'linear'
         total_iters=max_iters,
         warmup_iters=max(0, _warmup_iters),
         log_interval = log_iterations
    ),
]

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

# To log to Weights & Biases, install the `wandb` extra and uncomment:
# visualizer = dict(
#    type=Visualizer,
#    vis_backends=[
#        dict(type=WandbVisBackend, 
#             init_kwargs=dict(project='mae_resampler_pretraining', 
#                              name = 'mae_resampler_pretrain'))])
# 
# 
visualizer = None

log_level = 'INFO'
log_processor = dict(
    by_epoch=False,
   window_size=10,
)
randomness = dict(seed=seed, deterministic=False)
load_from = None
resume = False
