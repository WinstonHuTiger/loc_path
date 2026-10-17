from typing import Dict, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence

from xtuner.dataset.feature_mae import collate_feature_tokens
from xtuner.parallel.sequence import (get_sequence_parallel_world_size,
                                      pad_for_sequence_parallel)
from xtuner.utils import DEFAULT_PAD_TOKEN_INDEX, IGNORE_INDEX


def slidechat_feature_collate_fn(instances: Sequence[Dict],
                                 pad_index: int = DEFAULT_PAD_TOKEN_INDEX,
                                 return_hf_format: bool = False,
                                 use_varlen_attn: bool = False,
                                 max_tokens: int = None):
    if use_varlen_attn:
        raise NotImplementedError(
            'slidechat_feature_collate_fn does not support varlen attention.')

    seq_parallel_world_size = get_sequence_parallel_world_size()

    input_ids = [torch.LongTensor(inst['input_ids']) for inst in instances]
    labels = [torch.LongTensor(inst['labels']) for inst in instances]

    ori_lengths = [len(ids) for ids in input_ids]
    if len(instances) > 1:
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=pad_index)
        labels = pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX)
    else:
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)

    attention_mask = torch.zeros_like(input_ids).bool()
    for row, length in enumerate(ori_lengths):
        attention_mask[row, :length] = True

    bs, seq_len = input_ids.shape
    position_ids = torch.arange(seq_len).unsqueeze(0).long().repeat(bs, 1)

    feature_items = []
    for inst in instances:
        entry = {'tokens': inst['vision_tokens']}
        if 'vision_coords' in inst:
            entry['coords'] = inst['vision_coords']
        if 'path' in inst:
            entry['path'] = inst['path']
        feature_items.append(entry)
    feature_batch = collate_feature_tokens(feature_items, max_tokens=max_tokens)

    pixel_values = feature_batch['tokens']
    valid_mask = feature_batch['valid_mask']
    vision_pad_mask = ~valid_mask
    coords = feature_batch.get('coords')

    if seq_parallel_world_size > 1:
        input_ids = pad_for_sequence_parallel(input_ids, pad_index)
        labels = pad_for_sequence_parallel(labels, IGNORE_INDEX)
        position_ids = pad_for_sequence_parallel(position_ids, 0)
        attention_mask = pad_for_sequence_parallel(attention_mask, 0)
        vision_pad_mask = pad_for_sequence_parallel(vision_pad_mask, True)
        if coords is not None:
            coords = pad_for_sequence_parallel(coords, 0.0)
        pixel_values = pad_for_sequence_parallel(pixel_values, 0.0)

    data_dict = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'position_ids': position_ids,
        'labels': labels,
        'pixel_values': pixel_values,
        'vision_pad_mask': vision_pad_mask,
    }
    if coords is not None:
        data_dict['coords'] = coords
    paths = feature_batch.get('paths')
    if paths is not None:
        data_dict['paths'] = paths

    if return_hf_format:
        return data_dict
    return {'data': data_dict, 'data_samples': None}
