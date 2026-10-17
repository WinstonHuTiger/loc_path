# BEFORE (bad): you were doing this
# return build_optim_wrapper(model, wrapper_cfg)

# AFTER (good): just return the optimizer object
from typing import List
import torch
from copy import deepcopy
from mmengine.registry import OPTIM_WRAPPER_CONSTRUCTORS

@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class MuonOptimWrapperConstructor:
    def __init__(self,
                 optim_wrapper_cfg: dict,
                 paramwise_cfg=None,
                 muon_hidden_lr: float = 2e-2,
                 muon_weight_decay: float = 1e-2,
                 adamw_lr: float = 3e-4,
                 adamw_betas=(0.9, 0.95),
                 adamw_weight_decay: float = 1e-2,
                 keep_lora_on_adamw: bool = False,
                 **kwargs):
        # keep this; builder will wrap around whatever we return
        self.optim_wrapper_cfg = deepcopy(optim_wrapper_cfg)
        self.paramwise_cfg = paramwise_cfg
        self.muon_hidden_lr = muon_hidden_lr
        self.muon_weight_decay = muon_weight_decay
        self.adamw_lr = adamw_lr
        self.adamw_betas = adamw_betas
        self.adamw_weight_decay = adamw_weight_decay
        self.keep_lora_on_adamw = keep_lora_on_adamw

    def _trainable(self, m):
        return [p for p in m.parameters() if p.requires_grad]

    def __call__(self, model):
        module = model.module if hasattr(model, 'module') else model

        # collect trainable params from LongNet, projector, LLM (LoRA)
        params = []
        for name in ('LongNet_encoder', 'projector', 'llm'):
            if hasattr(module, name):
                params += self._trainable(getattr(module, name))

        def is_lora_param(p):
            # wire your own detector if needed
            return getattr(p, '_is_lora_param', False)

        hidden = []
        non_hidden = []
        for p in params:
            if self.keep_lora_on_adamw and is_lora_param(p):
                non_hidden.append(p)
            elif getattr(p, 'ndim', 0) >= 2:
                hidden.append(p)
            else:
                non_hidden.append(p)

        from muon import MuonWithAuxAdam
        param_groups = [
            dict(params=hidden, use_muon=True,
                 lr=self.muon_hidden_lr, weight_decay=self.muon_weight_decay),
            dict(params=non_hidden, use_muon=False,
                 lr=self.adamw_lr, betas=self.adamw_betas,
                 weight_decay=self.adamw_weight_decay),
        ]
        optimizer = MuonWithAuxAdam(param_groups)

        # IMPORTANT: return the optimizer, not the wrapper
        return optimizer