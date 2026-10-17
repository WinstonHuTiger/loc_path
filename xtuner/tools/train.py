# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
import logging
import os
import time
import torch
import os.path as osp
from functools import partial
from types import FunctionType

import torch.distributed as dist

from mmengine.config import Config, DictAction
from mmengine.config.lazy import LazyObject
from mmengine.logging import print_log
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from mmengine.utils import digit_version
from mmengine.hooks import Hook  # ← added
from mmengine.logging import print_log

from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import TrainingArguments

from xtuner.configs import cfgs_name_path
from xtuner.configs.loader import load_config
from xtuner.dataset.collate_fns import default_collate_fn
from xtuner.model.modules import dispatch_modules
from xtuner.model.modules.dispatch import SUPPORT_FLASH2
from xtuner.model.utils import LoadWoInit, find_all_linear_names, traverse_dict
from xtuner.registry import BUILDER, MAP_FUNC
from xtuner.tools.utils import (auto_dtype_of_deepspeed_config,
                                get_seed_from_checkpoint)

torch.autograd.set_detect_anomaly(True)
# torch.cuda.set_sync_debug_mode(1)  # 可选，便于定位哪一步产生Inf/NaN

def _unwrap(m):
    for a in ("module","model","_model","module_wrapped"):
        if hasattr(m, a) and hasattr(getattr(m,a), "named_parameters"):
            return getattr(m,a)
    return m

# ---------------- 新增：训练时打印梯度&探针的 Hook ----------------
class GradProbeHook(Hook):
    """每隔 n iter 打印 projector 梯度与 A/B/C 三个探针的 grad L2；并校验占位符被替换。"""
    priority = "LOW"
    def __init__(self, every_n_iters=20, only_rank0=True):
        self.n = every_n_iters
        self.only0 = only_rank0

    def _is0(self):
        try:
            return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank()==0
        except Exception:
            return True

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if self.n<=0 or (batch_idx+1)%self.n!=0: return
        if self.only0 and not self._is0(): return

        m = _unwrap(runner.model)
        if not hasattr(m, "projector"):
            print_log("[GradProbe] no projector found", logger=runner.logger)
            return

        # 1) projector 参数梯度统计
        n_tot = n_with = n_none = 0
        total_l2 = 0.0; max_abs = 0.0
        for name, p in m.projector.named_parameters():
            n_tot += 1
            g = p.grad
            if g is None:
                n_none += 1
                continue
            n_with += 1
            gf = g.detach().float()
            total_l2 += float(torch.linalg.vector_norm(gf).item())
            if gf.numel()>0:
                max_abs = max(max_abs, float(gf.abs().max().item()))

        # 2) 三个探针的 grad
        def gnorm(x):
            if x is None or x.grad is None: return None
            return float(torch.linalg.vector_norm(x.grad.detach().float()).item())

        gA = gnorm(getattr(m, "_tap_A_proj_out", None))        # projector 输出
        gB = gnorm(getattr(m, "_tap_B_to_prepare", None))      # 送 prepare 前最终视觉张量
        gC = gnorm(getattr(m, "_tap_C_inputs_embeds", None))   # 传入 LLM 的 inputs_embeds（需要在 compute_loss 前设置）

        # 3) 插入检查结果（来自 forward 的缓存）
        repl = getattr(m, "_tap_replacement_ok", None)

        msg = (f"=== [GradProbe] iter={batch_idx+1}\n"
               f"projector params: with_grad={n_with}/{n_tot} none={n_none} "
               f"sumL2={total_l2:.3e} max|g|={max_abs:.3e}\n"
               f"A=proj_out_grad_L2: {gA}\n"
               f"B=to_prepare_grad_L2: {gB}\n"
               f"C=inputs_embeds_grad_L2: {gC}\n"
               f"replace_ok={repl}")
        print_log(msg, logger=runner.logger)


# ------------- helper to print trainable parameters -------------
def _resolve_torch_model(obj):
    """Try to unwrap containers (DDP/Deepspeed) to an nn.Module."""
    if obj is None:
        return None
    m = obj
    # unwrap common wrappers
    for attr in ("module", "model", "_model", "module_wrapped"):
        if hasattr(m, attr) and hasattr(getattr(m, attr), "named_parameters"):
            m = getattr(m, attr)
            break
    return m if hasattr(m, "named_parameters") else None

def print_trainable_parameters(model_like, *, show_names=True, max_names=200, logger='current'):
    try:
        model = _resolve_torch_model(model_like) or model_like
        if not hasattr(model, "named_parameters"):
            print_log("[Trainable] Model is not built yet (no named_parameters).", logger=logger)
            return
        total = 0
        trainable = 0
        trainable_lines = []
        for n, p in model.named_parameters():
            num = p.numel()
            total += num
            if p.requires_grad:
                trainable += num
                if show_names:
                    trainable_lines.append(f"{n}  shape={tuple(p.shape)}  dtype={p.dtype}")
        pct = (trainable / total * 100) if total else 0.0
        print_log(f"[Trainable] {trainable:,} / {total:,} params ({pct:.2f}%)", logger=logger)
        if show_names:
            if len(trainable_lines) > max_names:
                head = "\n".join(trainable_lines[:max_names])
                print_log(head, logger=logger)
                print_log(f"... ({len(trainable_lines) - max_names} more trainable tensors not shown)", logger=logger)
            else:
                print_log("\n".join(trainable_lines) if trainable_lines else "(no trainable tensors)", logger=logger)
    except Exception as e:
        print_log(f"[Trainable] Failed to enumerate parameters: {e}", logger=logger)
# ---------------------------------------------------------------

# -------- Hook to print after MMEngine actually builds the model --------
class PrintTrainableParamsHook(Hook):
    """MMEngine hook: print trainable params once the runner/model are ready."""
    priority = "LOW"

    def before_train(self, runner):
        
        # 1. 从 runner 获取 optimizer
        opt = runner.optim_wrapper.optimizer

        names_in_opt = set()
        # 2. 遍历 optimizer 的所有参数组
        for pg in opt.param_groups:
            # 3. 在每个组中遍历所有参数
            for p in pg['params']:
                # MMEngine 会为参数附加 _param_name 属性，便于识别
                if hasattr(p, '_param_name'):
                    names_in_opt.add(p._param_name)

        # 打印出所有在 optimizer 中的参数名
        print("Parameters in optimizer:", names_in_opt)

        # 您的代码甚至更进一步，检查了 projector 中有哪些可训练参数没有被加入 optimizer
        miss = [n for n,_ in _unwrap(runner.model).projector.named_parameters() if n not in names_in_opt]
        print_log(f"[check] projector params NOT in optimizer: {miss}", logger=runner.logger)
        
        print("*******************************************************Trainable parameters (MMEngine)")
        
        print_trainable_parameters(runner.model)
# -----------------------------------------------------------------------


class MemoryProfileHook(Hook):
    """Measure wall time, GPU memory, and optional FLOPs during training iterations."""

    priority = "LOW"

    def __init__(self, every_n_iters=50, out_path=None, only_rank0=True, profile_flops=False):
        self.every_n_iters = max(1, int(every_n_iters))
        self.out_path = out_path
        self.only_rank0 = only_rank0
        self.profile_flops = bool(profile_flops)
        self._iter_start = None
        self._records = []
        self._profiler = None
        self._flops_measured = False

    def _is_master(self):
        if not dist.is_available() or not dist.is_initialized():
            return True
        return dist.get_rank() == 0

    def before_train_iter(self, runner, batch_idx, data_batch=None):
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
        if self.profile_flops and not self._flops_measured and self._profiler is None:
            try:
                import torch.profiler as profiler  # lazy import
                activities = [profiler.ProfilerActivity.CPU]
                if torch.cuda.is_available():
                    activities.append(profiler.ProfilerActivity.CUDA)
                self._profiler = profiler.profile(
                    activities=activities,
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=False,
                    with_flops=True,
                )
                self._profiler.start()
            except Exception as exc:
                print_log(f"[MemProfile] Failed to start profiler: {exc}", logger=runner.logger, level=logging.WARNING)
                self._profiler = None
                self._flops_measured = True
        self._iter_start = time.time()

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if self.only_rank0 and not self._is_master():
            return

        dt = None
        if self._iter_start is not None:
            dt = time.time() - self._iter_start
        peak_alloc = peak_reserved = None
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                device = torch.cuda.current_device()
                peak_alloc = torch.cuda.max_memory_allocated(device)
                peak_reserved = torch.cuda.max_memory_reserved(device)
            except Exception:
                peak_alloc = peak_reserved = None

        loss_val = None
        if isinstance(outputs, dict):
            loss_tensor = outputs.get('loss', None)
            if loss_tensor is not None:
                try:
                    loss_val = float(loss_tensor.detach().cpu())
                except Exception:
                    loss_val = None

        global_iter = runner.iter + 1
        record = dict(
            iter=global_iter,
            epoch=runner.epoch,
            dt_sec=dt,
            gpu_peak_alloc_bytes=peak_alloc,
            gpu_peak_reserved_bytes=peak_reserved,
            loss=loss_val,
            flops=None,
        )
        if self._profiler is not None:
            try:
                self._profiler.stop()
                total_flops = 0
                for evt in self._profiler.key_averages():
                    if hasattr(evt, 'flops') and evt.flops is not None:
                        total_flops += evt.flops
                record['flops'] = total_flops
            except Exception as exc:
                print_log(f"[MemProfile] Failed to collect FLOPs: {exc}", logger=runner.logger, level=logging.WARNING)
            finally:
                self._profiler = None
                self._flops_measured = True

        self._records.append(record)

        if global_iter % self.every_n_iters == 0:
            time_str = f"{dt:.4f}s" if dt is not None else "N/A"
            msg = (f"[MemProfile] iter={global_iter} epoch={runner.epoch} "
                   f"time={time_str} peak_alloc={peak_alloc}B "
                   f"peak_reserved={peak_reserved}B loss={loss_val} flops={record['flops']}")
            print_log(msg, logger=runner.logger)

    def after_train(self, runner):
        if self.out_path is None or not self._records:
            return
        if self.only_rank0 and not self._is_master():
            return
        try:
            os.makedirs(os.path.dirname(self.out_path) or '.', exist_ok=True)
            with open(self.out_path, 'w', encoding='utf-8') as f:
                json.dump(self._records, f, ensure_ascii=False, indent=2)
            print_log(f"[MemProfile] Stats saved to {self.out_path}", logger=runner.logger)
        except Exception as exc:
            print_log(f"[MemProfile] Failed to save stats: {exc}", logger=runner.logger, level=logging.WARNING)


def parse_args():
    parser = argparse.ArgumentParser(description='Train LLM')
    parser.add_argument('config', help='config file name or path.')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--deepspeed',
        type=str,
        default=None,
        help='the path to the .json file for deepspeed')
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='specify checkpoint path to be resumed from.')
    parser.add_argument(
        '--seed', type=int, default=None, help='Random seed for the training')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--profile-train',
        action='store_true',
        help='Measure per-iteration wall time and GPU memory during training.')
    parser.add_argument(
        '--profile-train-every',
        type=int,
        default=50,
        help='Logging interval (iterations) for training memory profiling.')
    parser.add_argument(
        '--profile-train-out',
        type=str,
        default=None,
        help='Optional JSON file path to save aggregated training profiling stats.')
    parser.add_argument(
        '--profile-train-flops',
        action='store_true',
        help='Also estimate FLOPs for the first profiled training iteration.')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()

    return args


def register_function(cfg_dict):
    if isinstance(cfg_dict, dict):
        for key, value in dict.items(cfg_dict):
            if isinstance(value, FunctionType):
                value_str = str(value)
                if value_str not in MAP_FUNC:
                    MAP_FUNC.register_module(module=value, name=value_str)
                cfg_dict[key] = value_str
            else:
                register_function(value)
    elif isinstance(cfg_dict, (list, tuple)):
        for value in cfg_dict:
            register_function(value)


def check_cfg(cfg, args):
    if getattr(cfg, 'use_varlen_attn',
               False) and cfg.train_dataloader.batch_size > 1:
        raise NotImplementedError(
            f'If utilizing varlen attention, the batch size should be'
            f' set to 1, but got {cfg.train_dataloader.batch_size}')

    if getattr(cfg, 'use_varlen_attn', False):
        sequence_parallel = getattr(cfg, 'sequence_parallel', 1)
        max_length = getattr(cfg.train_dataloader.dataset, 'max_length', None)
        if max_length is not None:
            assert max_length % sequence_parallel == 0, \
                ('When using varlen attention, `max_length` should be evenly '
                 'divided by sequence parallel world size, but got '
                 f'max_length = {max_length} and sequence_parallel = '
                 f'{sequence_parallel}')

    if getattr(cfg, 'sequence_parallel_size', 1) > 1:
        assert SUPPORT_FLASH2, ('`flash_attn` is required if you want to use '
                                'sequence parallel.')
        attn_implementation = getattr(cfg.model.llm, 'attn_implementation',
                                      None)
        assert (attn_implementation is None or
                attn_implementation == 'flash_attention_2'), \
            ('If you want to use sequence parallel, please set '
                'attn_implementation to `flash_attention_2` or do not '
                f'set this attribute. Got `{attn_implementation}` .')

    if getattr(cfg, 'use_varlen_attn', False):
        assert SUPPORT_FLASH2, ('`flash_attn` is required if you set '
                                '`use_varlen_attn` to True.')
        attn_implementation = getattr(cfg.model.llm, 'attn_implementation',
                                      None)
        assert (attn_implementation is None or
                attn_implementation == 'flash_attention_2'), \
            ('If you want to set `use_varlen_attn` to True, please set'
                ' attn_implementation to `flash_attention_2` or do not '
                f'set this attribute. Got `{attn_implementation}` .')

    if args.deepspeed is None:
        assert getattr(cfg, 'sequence_parallel_size', 1) == 1, \
            ('Sequence parallel training without DeepSpeed lacks validation.'
             'Please use DeepSpeed to optimize the training phase by '
             '`--deepspeed deepspeed_zero1 (deepspeed_zero2 or '
             'deepspeed_zero3)`.')


def main():
    print('*******************************************************Start Initialization')

    args = parse_args()

    # parse config
    if not osp.isfile(args.config):
        try:
            args.config = cfgs_name_path[args.config]
        except KeyError:
            raise FileNotFoundError(f'Cannot find {args.config}')

    # load config
    cfg = load_config(args.config)
    # Force registry import so your constructor is registered
    from xtuner.engine.optimizer.muon_wrapper import MuonOptimWrapperConstructor

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    args.work_dir = args.work_dir or cfg.get('work_dir') or osp.join(
        './work_dirs', osp.splitext(osp.basename(args.config))[0])
    if args.resume == 'auto':
        marker = osp.join(args.work_dir, 'last_checkpoint')
        if not osp.isfile(marker):
            raise FileNotFoundError(f'No checkpoint marker found: {marker}')
        with open(marker, encoding='utf-8') as handle:
            args.resume = handle.readline().strip()
        if not args.resume:
            raise ValueError(f'Empty checkpoint marker: {marker}')

    # register FunctionType object in cfg to `MAP_FUNC` Registry and
    # change these FunctionType object to str
    register_function(cfg._cfg_dict)

    check_cfg(cfg, args)


    if cfg.get('framework', 'mmengine').lower() == 'huggingface':
        # set default training_args
        if cfg.get('training_args', None) is None:
            cfg.training_args = dict(type=TrainingArguments)
        if args.seed is not None:
            cfg.training_args.seed = args.seed
        # set work_dir
        if args.work_dir is not None:
            cfg.training_args.output_dir = args.work_dir
        elif cfg.training_args.get('output_dir', None) is None:
            cfg.training_args.output_dir = osp.join(
                './work_dirs',
                osp.splitext(osp.basename(args.config))[0])
        # enable deepspeed
        if args.deepspeed:
            if not osp.isfile(args.deepspeed):
                try:
                    args.deepspeed = cfgs_name_path[args.deepspeed]
                except KeyError:
                    raise FileNotFoundError(f'Cannot find {args.deepspeed}')
            cfg.training_args.deepspeed = args.deepspeed
        if cfg.training_args.get('deepspeed'):
            device_map = None
        else:
            # Data Parallel
            device_map = {
                '': int(os.environ.get('LOCAL_RANK', args.local_rank))
            }
        # build training_args
        training_args = BUILDER.build(cfg.training_args)
        # build model
        print('*******************************************************Build Model')

        with LoadWoInit():
            cfg.model.device_map = device_map
            traverse_dict(cfg.model)
            model = BUILDER.build(cfg.model)
        model.config.use_cache = False
        dispatch_modules(model)
        if cfg.get('lora', None):
            lora = BUILDER.build(cfg.lora)
            model = prepare_model_for_kbit_training(model)
            if lora.target_modules is None:
                modules = find_all_linear_names(model)
                lora.target_modules = modules
            model = get_peft_model(model, lora)

        # Print trainable params (HF branch)
        print("*******************************************************Trainable parameters (HF)")
        print_trainable_parameters(model)

        # build dataset
        print('*******************************************************Build Dataset')
        train_dataset = BUILDER.build(cfg.train_dataset)
        data_collator = partial(default_collate_fn, return_hf_format=True)
        # build trainer
        trainer = cfg.trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator)
        # training
        print('*******************************************************start training')
        trainer.train(resume_from_checkpoint=args.resume)
        trainer.save_state()
        trainer.save_model(output_dir=training_args.output_dir)
    else:
        if args.seed is not None and args.resume is None:
            # Use args.seed
            cfg.merge_from_dict(dict(randomness=dict(seed=args.seed)))
            print_log(
                f'Set the random seed to {args.seed}.',
                logger='current',
                level=logging.INFO)
        elif args.resume is not None: # False
            # Use resumed seed
            from mmengine.fileio import PetrelBackend, get_file_backend

            from xtuner.utils.fileio import patch_fileio
            backend = get_file_backend(args.resume)
            if isinstance(backend, PetrelBackend):
                with patch_fileio():
                    resumed_seed = get_seed_from_checkpoint(args.resume)
            else:
                resumed_seed = get_seed_from_checkpoint(args.resume)
            cfg.merge_from_dict(dict(randomness=dict(seed=resumed_seed)))
            if args.seed is not None and args.seed != resumed_seed:
                print_log(
                    (f'The value of random seed in resume checkpoint '
                     f'"{args.resume}" is different from the value in '
                     f'arguments. The resumed seed is {resumed_seed}, while '
                     f'the input argument seed is {args.seed}. Using the '
                     f'resumed seed {resumed_seed}.'),
                    logger='current',
                    level=logging.WARNING)
            else:
                print_log(
                    f'Set the random seed to {resumed_seed}.',
                    logger='current',
                    level=logging.INFO)

        if 'LOCAL_RANK' not in os.environ:
            os.environ['LOCAL_RANK'] = str(args.local_rank)
        cfg.launcher = args.launcher
        # work_dir is determined in this priority:
        # CLI > segment in file > filename
        if args.work_dir is not None:
            cfg.work_dir = args.work_dir
        elif cfg.get('work_dir', None) is None:
            cfg.work_dir = osp.join('./work_dirs',
                                    osp.splitext(osp.basename(args.config))[0])

        if args.deepspeed:
            try:
                import deepspeed
            except ImportError:
                raise ImportError(
                    'deepspeed is not installed properly, please check.')
            if digit_version(deepspeed.__version__) < digit_version('0.12.3'):
                raise RuntimeError('Please upgrade your DeepSpeed version '
                                   'by using the command pip install '
                                   '`deepspeed>=0.12.3`')
            optim_wrapper = cfg.optim_wrapper.type
            if optim_wrapper == 'DeepSpeedOptimWrapper':
                print_log(
                    'Deepspeed training is already enabled in your config.',
                    logger='current',
                    level=logging.WARNING)
            else:
                if not osp.isfile(args.deepspeed):
                    try:
                        args.deepspeed = cfgs_name_path[args.deepspeed]
                    except KeyError:
                        raise FileNotFoundError(
                            f'Cannot find {args.deepspeed}')
                with open(args.deepspeed) as f:
                    ds_cfg = json.load(f)

                ds_grad_accum = ds_cfg.get('gradient_accumulation_steps',
                                           'auto')
                mm_grad_accum = cfg.optim_wrapper.get('accumulative_counts', 1)
                if ds_grad_accum != 'auto' and ds_grad_accum != mm_grad_accum:
                    print_log(('Mismatch on gradient_accumulation_steps: '
                               f'MMEngine {mm_grad_accum}, '
                               f'Deepspeed {ds_grad_accum}. '
                               f'Set to {mm_grad_accum}'),
                              logger='current',
                              level=logging.WARNING)
                grad_accum = mm_grad_accum

                ds_train_bs = ds_cfg.get('train_micro_batch_size_per_gpu',
                                         'auto')
                mm_train_bs = cfg.train_dataloader.batch_size
                if ds_train_bs != 'auto' and ds_train_bs != mm_train_bs:
                    print_log(
                        ('Mismatch on train_micro_batch_size_per_gpu: '
                         f'MMEngine {mm_train_bs}, Deepspeed {ds_train_bs}. '
                         f'Set to {mm_train_bs}'),
                        logger='current',
                        level=logging.WARNING)
                train_bs = cfg.train_dataloader.batch_size
                print('train_bs', train_bs)

                ds_grad_clip = ds_cfg.get('gradient_clipping', 'auto')
                clip_grad = cfg.optim_wrapper.get('clip_grad', None)
                if clip_grad and clip_grad.get('max_norm', None) is not None:
                    mm_max_norm = cfg.optim_wrapper.clip_grad.max_norm
                else:
                    mm_max_norm = 1.0
                if ds_grad_clip != 'auto' and ds_grad_clip != mm_max_norm:
                    print_log(
                        ('Mismatch on gradient_clipping: '
                         f'MMEngine {mm_max_norm}, Deepspeed {ds_grad_clip}. '
                         f'Set to {mm_max_norm}'),
                        logger='current',
                        level=logging.WARNING)
                grad_clip = mm_max_norm
                ds_cfg = auto_dtype_of_deepspeed_config(ds_cfg)
                exclude_frozen_parameters = True if digit_version(
                    deepspeed.__version__) >= digit_version('0.10.1') else None
                strategy = dict(
                    type=LazyObject('xtuner.engine', 'DeepSpeedStrategy'),
                    config=ds_cfg,
                    gradient_accumulation_steps=grad_accum,
                    train_micro_batch_size_per_gpu=train_bs,
                    gradient_clipping=grad_clip,
                    exclude_frozen_parameters=exclude_frozen_parameters,
                    sequence_parallel_size=getattr(cfg,
                                                   'sequence_parallel_size',
                                                   1))
                cfg.__setitem__('strategy', strategy)
                # 取原始 optim_wrapper，便于拿出可选项
                orig_ow = cfg.optim_wrapper if hasattr(cfg, 'optim_wrapper') else {}

                optim_wrapper = dict(
                    type='DeepSpeedOptimWrapper',
                    optimizer=orig_ow.get('optimizer'),
                    # 关键：把 paramwise_cfg 传过来，custom_keys 才会生效
                    paramwise_cfg=orig_ow.get('paramwise_cfg', None),
                )

                # 如果你在 cfg 里定义了自定义的 OptimWrapperConstructor，也一并传递
                if hasattr(cfg, 'optim_wrapper_constructor'):
                    cfg.optim_wrapper_constructor = cfg.optim_wrapper_constructor
                cfg.__setitem__('optim_wrapper', optim_wrapper)
                
                cfg.runner_type = 'FlexibleRunner'

        # resume is determined in this priority: resume from > auto_resume
        if args.resume is not None:
            cfg.resume = True
            cfg.load_from = args.resume

        # build the runner from config
        if 'runner_type' not in cfg: # False
            # build the default runner
            runner = Runner.from_cfg(cfg)
        else:
            # build customized runner from the registry
            # if 'runner_type' is set in the cfg
            runner = RUNNERS.build(cfg)

        # Register hook to print once the model is actually built
        # runner.register_hook(PrintTrainableParamsHook(), priority='LOW')
        # 注册 projector 梯度检查
        # runner.register_hook(GradProbeHook(every_n_iters=10, only_rank0=True), priority='LOW')
        if args.profile_train:
            runner.register_hook(
                MemoryProfileHook(
                    every_n_iters=args.profile_train_every,
                    out_path=args.profile_train_out,
                    only_rank0=True,
                    profile_flops=args.profile_train_flops),
                priority='LOW')

        # start training
        runner.train()


if __name__ == '__main__':
    main()
