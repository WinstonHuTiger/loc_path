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
                                 HFCheckpointHook, ThroughputHook)
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
MAX_LENGTH = 2024  + PER_IMAGE_LENGTH
MAX_VISUAL_TOKENS = 61440
MAP_NUM_PROC = 8
TRAIN_STAGE = '1'
MAX_EPOCHS = 3
warmup_ratio = 0.05
ACCUMULATIVE_COUNTS = 64
LR = 1e-4
BETAS = (0.9, 0.999)
WEIGHT_DECAY = 0.01
MAX_NEW_TOKENS = 512
EVAL_INTERVAL = 256

RESAMPLER_CKPT = os.getenv(
    'LOCPATH_RESAMPLER_CKPT',
    os.getenv('SLIDECHAT_RESAMPLER_CKPT', '/path/to/resampler_ckpt.pth'))
STAGE1_DATA_PATH = os.getenv(
    'LOCPATH_STAGE1_DATA_PATH',
    os.getenv('SLIDECHAT_STAGE1_DATA_PATH', '/path/to/stage1_dataset.json'))
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

save_steps = 1024
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
   llm_lora=None,
  visual_dim=512,
  resampler_cfg=dict(
       d_model=512,
       n_latents=128,
       num_heads=8,
       n_layers=2,
       topk=(128, 64),
       use_latent_self_attn=True,
       latent_drop=0.1,
       dense_latents=8,
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
    
    router_top_m=128,
    router_aux_weight=0.02,
    router_temperature_s=1.5,   # 学生更平一点
    router_temperature_t=1.0,   # 老师默认
    router_diag=True,
    # use text for router
    router_use_text = True,
    router_fuse='add',
    router_dropout=0.1,
    
    # not enable capture mass
    router_enable_capture_mass=False,
    teacher_pool = "topk",
    teacher_topk_frac = 0.1,
)

# -------------------------------
# Dataset
# -------------------------------
train_dataset = dict(
    type=SlideChatFeatureDataset,
    data_path=STAGE1_DATA_PATH,
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
        by_epoch=True,
        begin=0,
        end=MAX_EPOCHS * warmup_ratio,
        convert_to_iter_based=True  # 按 iter 计算
    ),
    dict(
        type=CosineAnnealingLR,
        eta_min=1e-5,
        by_epoch=True,
        begin=MAX_EPOCHS * warmup_ratio,
        end=MAX_EPOCHS,
        convert_to_iter_based=True  # 按 iter 计算
    ),
]

# -------------------------------
# Training settings
# -------------------------------
train_cfg = dict(
    type=TrainLoop,
    max_epochs=MAX_EPOCHS,
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
    # dict(type=RouterUpdateHook, 
    #      interval=ACCUMULATIVE_COUNTS, 
    #      accum = ACCUMULATIVE_COUNTS)
]

default_hooks = dict(
    timer=dict(type=IterTimerHook),
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
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
#                              name = 'stage1_loc_path'))])
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
