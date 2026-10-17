# # Copyright (c) OpenMMLab. All rights reserved.
# from typing import Dict, Sequence
# import numpy as np
# import torch
# from torch.nn.utils.rnn import pad_sequence

# from xtuner.parallel.sequence import (get_sequence_parallel_world_size,
#                                       pad_for_sequence_parallel)
# from xtuner.utils import DEFAULT_PAD_TOKEN_INDEX, IGNORE_INDEX


# def default_collate_fn(instances: Sequence[Dict],
#                        pad_index: int = DEFAULT_PAD_TOKEN_INDEX,
#                        return_hf_format: bool = False,
#                        use_varlen_attn: bool = False):
#     seq_parallel_world_size = get_sequence_parallel_world_size()

#     input_ids, labels = [], []
#     has_image = any(inst.get('pixel_values') is not None for inst in instances)
#     if use_varlen_attn:
#         position_ids, cumulative_len = [], []
#         assert len(instances) == 1, (
#             f'If utilizing varlen attention, the batch size should be'
#             f' set to 1, but got {len(instances)}')
#         assert not has_image, 'Currently, it is not configured to '
#         'accommodate the use of varlen Attention in multimodal training'

#     if has_image:
#         pixel_values = []

#     for example in instances:
#         input_ids.append(torch.LongTensor(example['input_ids']))
#         labels.append(torch.LongTensor(example['labels']))
#         if use_varlen_attn:
#             cumulative_len.append(torch.IntTensor(example['cumulative_len']))
#             position_ids.append(torch.LongTensor(example['position_ids']))

#         if has_image:
#             # pixel_values.append(example['pixel_values'])

#             if isinstance(example['pixel_values'], list):
#                 pixel_values.extend(example['pixel_values'])
#             else:
#                 pixel_values.append(example['pixel_values'])

#     ori_length = [len(ids) for ids in input_ids]
#     if len(instances) > 1:
#         input_ids = pad_sequence(
#             input_ids, batch_first=True, padding_value=pad_index)
#         labels = pad_sequence(
#             labels, batch_first=True, padding_value=IGNORE_INDEX)
#     else:
#         input_ids = torch.stack(input_ids)
#         labels = torch.stack(labels)

#     if use_varlen_attn:
#         assert input_ids.size(1) % seq_parallel_world_size == 0
#         attention_mask = None
#         position_ids = torch.stack(position_ids, dim=0)
#     else:
#         # Some tokenizers have the same eos token and pad token, so input_ids
#         # cannot be masked directly based on the pad token id.
#         attention_mask = torch.zeros_like(input_ids).bool()
#         for i in ori_length:
#             attention_mask[:i] = True

#         bs, seq_len = input_ids.shape
#         position_ids = torch.arange(seq_len).unsqueeze(0).long().repeat(bs, 1)

#     if seq_parallel_world_size > 1:
#         input_ids = pad_for_sequence_parallel(input_ids, pad_index)
#         labels = pad_for_sequence_parallel(labels, IGNORE_INDEX)
#         position_ids = pad_for_sequence_parallel(position_ids, 0)
#         if attention_mask is not None:
#             attention_mask = pad_for_sequence_parallel(attention_mask, 0)

#     if use_varlen_attn:
#         max_seqlen = (
#             cumulative_len[0][1:] -  # noqa: W504
#             cumulative_len[0][:-1]).max().item()
#         data_dict = {
#             'input_ids': input_ids,
#             'cumulative_len': cumulative_len,
#             'position_ids': position_ids,
#             'labels': labels,
#             'max_seqlen': max_seqlen
#         }
#     else:
#         data_dict = {
#             'input_ids': input_ids,
#             'attention_mask': attention_mask,
#             'position_ids': position_ids,
#             'labels': labels
#         }

#     if has_image:

#         pixel_values = torch.stack(pixel_values)
#         data_dict['pixel_values'] = pixel_values


#     if return_hf_format:
#         return data_dict
#     else:
#         return {'data': data_dict, 'data_samples': None}

# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Sequence
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from xtuner.parallel.sequence import (get_sequence_parallel_world_size,
                                      pad_for_sequence_parallel)
from xtuner.utils import DEFAULT_PAD_TOKEN_INDEX, IGNORE_INDEX


def default_collate_fn(instances: Sequence[Dict],
                       pad_index: int = DEFAULT_PAD_TOKEN_INDEX,
                       return_hf_format: bool = False,
                       use_varlen_attn: bool = False):
    seq_parallel_world_size = get_sequence_parallel_world_size()

    input_ids, labels = [], []
    has_image = any(inst.get('pixel_values') is not None for inst in instances)
    if use_varlen_attn:
        position_ids, cumulative_len = [], []
        assert len(instances) == 1, (
            f'If utilizing varlen attention, the batch size should be'
            f' set to 1, but got {len(instances)}')
        assert not has_image, 'Currently, it is not configured to '
        'accommodate the use of varlen Attention in multimodal training'

    if has_image:
        pixel_values = []
        # NEW: detect if any instance provides coords
        has_coords = any(inst.get('coords') is not None for inst in instances)
        if has_coords:
            coords = []

    for example in instances:
        input_ids.append(torch.LongTensor(example['input_ids']))
        labels.append(torch.LongTensor(example['labels']))
        if use_varlen_attn:
            cumulative_len.append(torch.IntTensor(example['cumulative_len']))
            position_ids.append(torch.LongTensor(example['position_ids']))

        if has_image:
            # Flatten any per-sample list of pixel_value tensors across the batch
            if isinstance(example['pixel_values'], list):
                pixel_values.extend(example['pixel_values'])
            else:
                pixel_values.append(example['pixel_values'])

            # NEW: mirror the same flattening for coords if present
            if 'coords' in example and example['coords'] is not None:
                if isinstance(example['coords'], list):
                    coords.extend(example['coords'])
                else:
                    coords.append(example['coords'])

    ori_length = [len(ids) for ids in input_ids]
    if len(instances) > 1:
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=pad_index)
        labels = pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX)
    else:
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)

    if use_varlen_attn:
        assert input_ids.size(1) % seq_parallel_world_size == 0
        attention_mask = None
        position_ids = torch.stack(position_ids, dim=0)
    else:
        # Some tokenizers have the same eos token and pad token, so input_ids
        # cannot be masked directly based on the pad token id.
        attention_mask = torch.zeros_like(input_ids).bool()
        for i in ori_length:
            attention_mask[:i] = True

        bs, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len).unsqueeze(0).long().repeat(bs, 1)

    if seq_parallel_world_size > 1:
        input_ids = pad_for_sequence_parallel(input_ids, pad_index)
        labels = pad_for_sequence_parallel(labels, IGNORE_INDEX)
        position_ids = pad_for_sequence_parallel(position_ids, 0)
        if attention_mask is not None:
            attention_mask = pad_for_sequence_parallel(attention_mask, 0)

    if use_varlen_attn:
        max_seqlen = (
            cumulative_len[0][1:] -  # noqa: W504
            cumulative_len[0][:-1]).max().item()
        data_dict = {
            'input_ids': input_ids,
            'cumulative_len': cumulative_len,
            'position_ids': position_ids,
            'labels': labels,
            'max_seqlen': max_seqlen
        }
    else:
        data_dict = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'labels': labels
        }

    if has_image:
        num_pixel_entries = len(pixel_values)
        if all(x.shape == pixel_values[0].shape for x in pixel_values):
            pixel_values = torch.stack(pixel_values)
        else:
            # Fallback for variable-length visual token sequences in a batch.
            # Pad to max length along dim-0 so multimodal training can proceed.
            if not all(x.dim() == 2 for x in pixel_values):
                raise RuntimeError(
                    'default_collate_fn expects each pixel_values tensor to be '
                    '2D [num_tokens, hidden_dim] when shapes differ.')
            feat_dims = {int(x.shape[1]) for x in pixel_values}
            if len(feat_dims) != 1:
                raise RuntimeError(
                    f'Inconsistent pixel feature dims in batch: {sorted(feat_dims)}')
            pixel_values = pad_sequence(
                pixel_values, batch_first=True, padding_value=0.0)
        data_dict['pixel_values'] = pixel_values

        # NEW: add stacked/padded coords if any were provided by the dataset
        if 'has_coords' in locals() and has_coords and len(coords) > 0:
            if len(coords) != num_pixel_entries:
                raise RuntimeError(
                    f'coords/pixel_values count mismatch in collate: '
                    f'{len(coords)} vs {num_pixel_entries}')
            if all(c.shape == coords[0].shape for c in coords):
                data_dict['coords'] = torch.stack(coords)
            else:
                data_dict['coords'] = pad_sequence(
                    coords, batch_first=True, padding_value=0)

    if return_hf_format:
        return data_dict
    else:
        return {'data': data_dict, 'data_samples': None}
