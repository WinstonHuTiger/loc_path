import os
import warnings
from typing import List, Optional, Sequence

import torch
from mmengine.dist import master_only
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.utils import mkdir_or_exist
from mmengine.utils.misc import get_object_from_string
from transformers import GenerationConfig, StoppingCriteriaList

from xtuner.dataset.utils import load_image, load_slide_features
from xtuner.registry import BUILDER
from xtuner.utils import (DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX,
                          StopWordStoppingCriteria)


class EvaluateVisAdapterHook(Hook):
    """Evaluation hook tailored for ``LlavaVisAdapterModel``.

    It supports either plain prompts (no images) or prompts paired with
    pre-extracted slide features + coords (.h5/.npy) the same way the
    training dataset does. Visual features are routed through the model's
    multimodal pipeline before generation.
    """

    priority = 'LOW'

    def __init__(self,
                 tokenizer,
                 evaluation_inputs,
                 evaluation_features=None,
                 system: str = '',
                 prompt_template=None,
                 every_n_iters=None,
                 max_new_tokens: int = 512,
                 stop_word: str = None,
                 stop_words: Sequence[str] = (),
                 generation_kwargs: dict = None,
                 save_outputs: bool = False,
                 conch_snap_coords: bool = True,
                 conch_stride_report: bool = False):
        if isinstance(evaluation_inputs, str):
            evaluation_inputs = [evaluation_inputs]
        self.evaluation_inputs = evaluation_inputs
        self.conch_snap_coords = bool(conch_snap_coords)
        self.conch_stride_report = bool(conch_stride_report)

        if evaluation_features is None:
            self.eval_feats = [None] * len(evaluation_inputs)
            self.eval_coords = [None] * len(evaluation_inputs)
            self.eval_stride = [None] * len(evaluation_inputs)
        else:
            if isinstance(evaluation_features, str):
                evaluation_features = [evaluation_features]
            assert len(evaluation_features) in [1, len(evaluation_inputs)]
            if len(evaluation_features) == 1:
                evaluation_features = evaluation_features * len(evaluation_inputs)
            feats_list: List[Optional[torch.Tensor]] = []
            coords_list: List[Optional[torch.Tensor]] = []
            stride_list: List[Optional[tuple]] = []
            for path in evaluation_features:
                stride_info = None
                if isinstance(path, str):
                    lower = path.lower()
                    if lower.endswith(('.h5', '.hdf5', '.npy', '.npz')):
                        is_conch_h5 = lower.endswith(('.h5', '.hdf5'))
                        feats_np, coords_np, stride_info = load_slide_features(
                            path,
                            feature_key='features',
                            coord_key='coords',
                            expect_coords=True,
                            conch_snap_coords=self.conch_snap_coords and is_conch_h5,
                            zero_anchor=True)
                        feats, coords = feats_np, coords_np
                    else:
                        loaded = load_image(path)
                        if isinstance(loaded, tuple):
                            feats, coords = loaded
                        else:
                            feats, coords = loaded, None
                else:
                    if isinstance(path, tuple) and len(path) >= 2:
                        feats, coords = path[0], path[1]
                    else:
                        feats, coords = path, None
                feats_list.append(feats)
                coords_list.append(coords)
                stride_list.append(stride_info)
            self.eval_feats = feats_list
            self.eval_coords = coords_list
            self.eval_stride = stride_list

        self.tokenizer = BUILDER.build(tokenizer)
        self.every_n_iters = every_n_iters
        self.max_new_tokens = max_new_tokens
        self.save_outputs = save_outputs

        if prompt_template is None:
            instruction = '{input}'
        else:
            if isinstance(prompt_template, str):
                prompt_template = get_object_from_string(prompt_template)
            instruction = prompt_template.get('INSTRUCTION', '{input}')
            stop_words = list(stop_words) + prompt_template.get('STOP_WORDS', [])
            if system:
                system = prompt_template.get('SYSTEM', '{system}\n').format(system=system)
        if stop_word is not None:
            warnings.warn(('`stop_word` is deprecated, please use `stop_words` instead.'),
                          DeprecationWarning)
            stop_words = list(stop_words) + [stop_word]

        self.system = system
        self.instruction = instruction

        default_generation_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id)
        if generation_kwargs:
            default_generation_kwargs.update(generation_kwargs)
        self.gen_config = GenerationConfig(**default_generation_kwargs)

        self.stop_criteria = StoppingCriteriaList()
        for word in stop_words:
            self.stop_criteria.append(StopWordStoppingCriteria(self.tokenizer, word))

    def _format_prompt(self, text: str) -> str:
        if DEFAULT_IMAGE_TOKEN not in text and '<image>' not in text:
            text = DEFAULT_IMAGE_TOKEN + '\n' + text
        text = text.replace('<image>', DEFAULT_IMAGE_TOKEN)
        formatted = (self.system + self.instruction).format(input=text, round=1)
        return formatted

    def _build_inputs(self, prompt: str, device: torch.device):
        chunks = prompt.split(DEFAULT_IMAGE_TOKEN)
        encoded = []
        for idx, chunk in enumerate(chunks):
            ids = self.tokenizer.encode(
                chunk, add_special_tokens=(idx == 0))
            encoded.append(ids)

        input_ids = []
        for idx, ids in enumerate(encoded):
            input_ids.extend(ids)
            if idx != len(encoded) - 1:
                input_ids.append(IMAGE_TOKEN_INDEX)
        input_ids = torch.tensor(input_ids, device=device).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        return input_ids, attention_mask

    def _prepare_visual(self, feats, coords, device, dtype):
        if feats is None:
            return None, None
        if torch.is_tensor(feats):
            tensor = feats.to(device=device, dtype=dtype)
        else:
            tensor = torch.from_numpy(feats).to(device=device, dtype=dtype)
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        pad_mask = torch.zeros(tensor.size()[:-1], dtype=torch.bool, device=device)

        coord_tensor = None
        if coords is not None:
            if torch.is_tensor(coords):
                coord_tensor = coords.to(device=device, dtype=torch.float32)
            else:
                coord_tensor = torch.from_numpy(coords).to(device=device, dtype=torch.float32)
            if coord_tensor.dim() == 2:
                coord_tensor = coord_tensor.unsqueeze(0)
        return tensor, pad_mask, coord_tensor

    @master_only
    def _save_results(self, runner, outputs):
        save_dir = os.path.join(runner.log_dir, 'vis_data')
        mkdir_or_exist(save_dir)
        path = os.path.join(save_dir, f'eval_outputs_iter_{runner.iter}.txt')
        with open(path, 'w', encoding='utf-8') as f:
            for idx, item in enumerate(outputs):
                if isinstance(item, (tuple, list)) and len(item) >= 2:
                    q, a = item[0], item[1]
                    f.write(f'Example {idx + 1}:\nQ: {q}\nA: {a}\n\n')
                else:
                    f.write(f'Example {idx + 1}:\n{item}\n\n')

    @torch.no_grad()
    def evaluate_once(self, runner):
        model = runner.model
        if is_model_wrapper(model):
            model = model.module

        device = next(model.llm.parameters()).device
        dtype = model.llm.dtype
        outputs = []

        for idx_item, (prompt, feats, coords, stride_info) in enumerate(
                zip(self.evaluation_inputs, self.eval_feats, self.eval_coords, self.eval_stride)):
            formatted = self._format_prompt(prompt)
            input_ids, attention_mask = self._build_inputs(formatted, device)
            pixel_values, pad_mask, coords_tensor = self._prepare_visual(feats, coords, device, dtype)

            if self.conch_stride_report and stride_info is not None:
                runner.logger.info(
                    f'[EvalVisAdapter] Sample {idx_item}: CONCH stride≈({stride_info[0]}, {stride_info[1]})')

            batch = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_values=None,
                labels=None,
                pixel_values=pixel_values,
                vision_pad_mask=pad_mask,
                coords=coords_tensor)

            inputs = model._prepare_multimodal_inputs(batch)
            generated = model.llm.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                inputs_embeds=inputs['inputs_embeds'],
                generation_config=self.gen_config,
                stopping_criteria=self.stop_criteria)
            decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
            # Save both question (prompt) and answer
            outputs.append((prompt, decoded))

        if self.save_outputs:
            self._save_results(runner, outputs)
        for idx, item in enumerate(outputs):
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                q, a = item[0], item[1]
                runner.logger.info(f'[EvalVisAdapter] Q{idx + 1}: {q}')
                runner.logger.info(f'[EvalVisAdapter] A{idx + 1}: {a}')
            else:
                runner.logger.info(f'[EvalVisAdapter] Output {idx + 1}: {item}')

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if self.every_n_iters is None:
            return
        if (runner.iter + 1) % self.every_n_iters == 0:
            self.evaluate_once(runner)

    def before_val(self, runner) -> None:
        if self.every_n_iters is None:
            self.evaluate_once(runner)
