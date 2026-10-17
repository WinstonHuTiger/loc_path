import os

import torch
from mmengine.dataset import DefaultSampler
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.visualization import Visualizer, WandbVisBackend
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from xtuner.dataset import SlideChatFeatureDataset
from xtuner.dataset.collate_fns import slidechat_feature_collate_fn
from xtuner.dataset.map_fns import llava_map_fn, template_map_fn_factory
from xtuner.engine.hooks import (DatasetInfoHook,EvaluateVisAdapterHook, 
                                 HFCheckpointHook, ThroughputHook, 
                                 RouterAnnealHook)
from xtuner.engine.runner import TrainLoop
from xtuner.model.llava_vis_adapter import LlavaVisAdapterModel
from xtuner.utils import PROMPT_TEMPLATE

from peft import LoraConfig

# -------------------------------
# Settings
# -------------------------------
llm_name_or_path = os.getenv('LOCPATH_LLM_PATH', 'Qwen/Qwen2.5-7B-Instruct')
prompt_template = PROMPT_TEMPLATE.qwen_chat

BATCH_SIZE = 2
NUM_WORKERS = 16
PREFETCH_FACTOR = 2
PER_IMAGE_LENGTH = 256
MAX_LENGTH = 2048
MAX_VISUAL_TOKENS = 61440
MAP_NUM_PROC = 8
TRAIN_STAGE = '2'
warmup_ratio = 0.05
MAX_ITERS = 20480
WARMUP_ITERS = int(MAX_ITERS * warmup_ratio)
ACCUMULATIVE_COUNTS = 16
LR = 2e-5
BETAS = (0.9, 0.999)
WEIGHT_DECAY = 0.01
MAX_NEW_TOKENS = 512
EVAL_INTERVAL = 256

router_top_m = 128


RESAMPLER_CKPT = os.getenv(
    'LOCPATH_RESAMPLER_CKPT',
    os.getenv('SLIDECHAT_RESAMPLER_CKPT', '/path/to/resampler_ckpt.pth'))
STAGE2_PRETRAINED_PTH = os.getenv(
    'LOCPATH_STAGE2_PRETRAINED_PTH',
    os.getenv('SLIDECHAT_STAGE2_PRETRAINED_PTH', '/path/to/stage1_checkpoint.pth'))
STAGE2_DATA_PATH = os.getenv(
    'LOCPATH_STAGE2_DATA_PATH',
    os.getenv('SLIDECHAT_STAGE2_DATA_PATH', '/path/to/stage2_dataset.json'))
FEATURE_ROOT = os.getenv(
    'LOCPATH_FEATURE_ROOT',
    os.getenv('SLIDECHAT_FEATURE_ROOT', '/path/to/feature_root'))
UNWANTED_PREFIX_CSV = (
    os.getenv('LOCPATH_UNWANTED_PREFIX_CSV')
    or os.getenv('SLIDECHAT_UNWANTED_PREFIX_CSV')
    or None
)


SYSTEM = ''

evaluation_images = []
evaluation_inputs = []

save_steps = 2048 + 1024
save_total_limit = 8
bnb = dict(
    type=BitsAndBytesConfig,
    load_in_4bit=True,
    load_in_8bit=False,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

llm_lora = dict(
    type=LoraConfig,
    r=128,
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

tokenizer_cfg = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=llm_name_or_path,
    trust_remote_code=True,
    padding_side='right',
)

model = dict(
    type=LlavaVisAdapterModel,
    freeze_llm=True,
    train_stage=TRAIN_STAGE,
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=llm_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb,
    ),
   llm_lora=llm_lora,
  visual_dim=512,
  resampler_cfg=dict(
       d_model=512,
       n_latents=256,
       num_heads=8,
       n_layers=2,
       topk=(128, 64),
       use_latent_self_attn=True,
       latent_drop=0.1,
       dense_latents=0,
       context_mode='longnet',
       ctx_layers=2,
       ctx_longnet_segments=[1024, 2048, 4096],
       ctx_longnet_dilations=[[1, 2, 4], [4, 8, 16]],
       ctx_longnet_use_xpos=True,
       ctx_longnet_mlp_ratio=2.0,
    ),
    resampler_ckpt=RESAMPLER_CKPT,
    projector_depth=2,
    projector_dropout=0.05,
    
    
    adapter_layer_indices=(1, 3, 5, 7),
    adapter_dropout=0.1,
    vision_to_llm_scale = 0.5, 
    use_token_merger=True,
    use_activation_checkpointing = False,
    
    freeze_resampler=True,
    tune_resampler_last_latents_inproj = True, 
    feat_decor_weight = 1e-4,
    
     
    router_top_m=router_top_m,
    router_aux_weight=0.02,
    router_temperature_s=1.5,   # 学生更平一点
    router_temperature_t=1.0,   # 老师默认
    router_diag=True,
    # use text for router
    router_use_text = True,
    router_fuse='add',
    router_dropout=0.1,
    
    router_jitter = router_top_m * 0.15,
    
    # not enable capture mass
    router_enable_capture_mass=False,
    teacher_pool = "topk",
    teacher_topk_frac = 0.05,
    
    router_rank_lambda=0.5,     # 0.3~0.7 之间扫一下
    router_rank_pairs=8,
    router_rank_margin=0.0,
    
    pretrained_pth=STAGE2_PRETRAINED_PTH,
)

# -------------------------------
# Dataset
# -------------------------------
train_dataset = dict(
    type=SlideChatFeatureDataset,
    data_path=STAGE2_DATA_PATH,
    tokenizer=tokenizer_cfg,
    max_length=MAX_LENGTH,
    per_image_length=PER_IMAGE_LENGTH,
    with_image_token=True,
    map_num_proc=MAP_NUM_PROC,
    feature_root=FEATURE_ROOT,
    image_feature_suffix='.npy',
    identifier='_224x224_b20_t15',
    dataset_map_fn=llava_map_fn,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    sample_strategy='linspace',
    sampler_cfg=dict(max_tokens=MAX_VISUAL_TOKENS, strategy='linspace'),
    feature_key='features',
    coord_key='coords',
    require_coords=True,
    unwanted_prefix_csv=UNWANTED_PREFIX_CSV,
)

train_dataloader = dict(
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=PREFETCH_FACTOR,
    dataset=train_dataset,
    sampler=dict(type=DefaultSampler, shuffle=True),
    collate_fn=dict(type=slidechat_feature_collate_fn, max_tokens=MAX_VISUAL_TOKENS),
)

# -------------------------------
# Optimizer & Scheduler
# -------------------------------
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(type=AdamW, lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        custom_keys={
            # Router 学得更稳一点：半速 & 轻微正则
            'router':        dict(lr_mult=0.5, decay_mult=1.0, weight_decay=1e-4),
            # TokenMerger 不做权重衰减（避免合并权重被推向零）
            'token_merger':  dict(decay_mult=0.0),
            # Projector 可稍微更快些（加速对齐视觉/LLM空间）
            'projector':     dict(lr_mult=2.0),
            
            'resampler':       dict(lr_mult=0.25, weight_decay=1e-4),
            'feature_in_proj':      dict(lr_mult=0.5,  weight_decay=1e-4),
            # LoRA：建议 0 weight decay + 更大学习率
            'lora_':          dict(lr_mult=4.0, decay_mult=0.0),
        }
    ),
    clip_grad=dict(max_norm=1.0, error_if_nonfinite=True),
    accumulative_counts=ACCUMULATIVE_COUNTS,
    # BF16 gradients do not need scaling; PyTorch 2.4 cannot unscale BF16.
    loss_scale=dict(enabled=False),
    dtype='bfloat16',
)

param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=False,
        begin=0,
        end=max(1, WARMUP_ITERS),
    ),
    dict(
        type=CosineAnnealingLR,
        eta_min=2e-8,
        by_epoch=False,
        begin=max(1, WARMUP_ITERS),
        end=MAX_ITERS,
    ),
]

# -------------------------------
# Training settings
# -------------------------------
train_cfg = dict(
    type=TrainLoop,
    max_iters=MAX_ITERS,
)

custom_hooks = [
    dict(type=DatasetInfoHook, tokenizer = tokenizer_cfg),
    dict(type=ThroughputHook),
    dict(
        type = EvaluateVisAdapterHook,
        tokenizer = tokenizer_cfg,
        every_n_iters = EVAL_INTERVAL,
        prompt_template = prompt_template,
        evaluation_inputs = evaluation_inputs,
        evaluation_features = evaluation_images,
        max_new_tokens = MAX_NEW_TOKENS,
        system = SYSTEM,
    ),
     dict(
    type=RouterAnnealHook,
        milestones=(7000, 14000),
        # NEW: keep more tokens in the last stage for Open QA
        top_m=(int(router_top_m * 1.0), int(router_top_m * 0.80), int(router_top_m * 0.75)),
        temp_s=(1.6, 1.30, 1.15),
        temp_t=(0.85, 0.95, 1.0),
        aux_w=(0.02, 0.015, 0.01),
        mode='linear',
        log_interval=200),
        
]

default_hooks = dict(
    timer=dict(type=IterTimerHook),
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=20),
    param_scheduler=dict(type=ParamSchedulerHook),
    checkpoint=dict(type=CheckpointHook, 
                    by_epoch=False,
                    interval=save_steps, 
                    max_keep_ckpts=save_total_limit),
    sampler_seed=dict(type=DistSamplerSeedHook),
)

env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

# To log to Weights & Biases, install the `wandb` extra and uncomment:
# visualizer = dict(
#    type=Visualizer,
#    vis_backends=[
#        dict(type=WandbVisBackend, 
#             init_kwargs=dict(project='LoC-Path',
#                              name = 'stage2_loc_path'))])
visualizer = None

log_level = 'INFO'


# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False
 
# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)

# set log processor
log_processor = dict(by_epoch=False, 
                     window_size=10,
                     mean_pattern=r".*(loss|time|data_time|grad_norm|tflops).*",
                     )
