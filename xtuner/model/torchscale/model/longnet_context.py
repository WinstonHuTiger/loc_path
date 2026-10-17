import copy
from typing import List, Optional, Sequence, Union

from xtuner.model.torchscale.architecture.config import EncoderConfig

from .LongNet import LongNetEncoder, LongNetEncoderLayer
from fairscale.nn import checkpoint_wrapper, wrap


def _normalize_layerwise(
    values: Optional[Union[Sequence[int], Sequence[Sequence[int]]]],
    *,
    expected_len: Optional[int] = None,
    name: str = "value",
) -> Optional[List[List[int]]]:
    if values is None:
        return None
    if len(values) == 0:
        return []
    first = values[0]
    if isinstance(first, (int, float)):
        normalized = [list(values)]  # type: ignore[list-item]
    else:
        normalized = [list(seq) for seq in values]  # type: ignore[arg-type]
    if expected_len is not None:
        for idx, seq in enumerate(normalized):
            if len(seq) != expected_len:
                raise ValueError(
                    f"{name}[{idx}] length {len(seq)} does not match expected length {expected_len}."
                )
    return normalized


class LongNetContextEncoder(LongNetEncoder):
    def __init__(
        self,
        args: EncoderConfig,
        layer_segments: Optional[Union[Sequence[int], Sequence[Sequence[int]]]] = None,
        layer_dilations: Optional[Union[Sequence[int], Sequence[Sequence[int]]]] = None,
    ):
        expected = len(args.segment_length) if args.segment_length is not None else None
        self._layer_segments = _normalize_layerwise(
            layer_segments, expected_len=expected, name="segment_length"
        )
        self._layer_dilations = _normalize_layerwise(
            layer_dilations, expected_len=expected, name="dilated_ratio"
        )
        super().__init__(args)

    def build_encoder_layer(
        self,
        args,
        depth,
        is_moe_layer: bool = False,
        is_encoder_decoder: bool = False,
    ):
        layer_args = copy.deepcopy(args)
        if self._layer_segments:
            idx = min(depth, len(self._layer_segments) - 1)
            layer_args.segment_length = self._layer_segments[idx]
        if self._layer_dilations:
            idx = min(depth, len(self._layer_dilations) - 1)
            layer_args.dilated_ratio = self._layer_dilations[idx]

        layer = LongNetEncoderLayer(
            layer_args,
            depth,
            is_moe_layer=is_moe_layer,
            is_encoder_decoder=is_encoder_decoder,
        )
        if layer_args.checkpoint_activations:
            layer = checkpoint_wrapper(layer)
        if layer_args.fsdp:
            layer = wrap(layer)
        return layer


def make_longnet_context(
    d_model: int,
    num_heads: int,
    n_layers: int = 2,
    mlp_ratio: float = 2.0,
    dropout: float = 0.0,
    drop_path_rate: float = 0.05,
    seg: Sequence[int] = (1024, 2048, 4096),
    dil: Sequence[int] = (1, 2, 4),
    use_xpos: bool = True,
    layer_segments: Optional[Union[Sequence[int], Sequence[Sequence[int]]]] = None,
    layer_dilations: Optional[Union[Sequence[int], Sequence[Sequence[int]]]] = None,
) -> LongNetContextEncoder:
    if not use_xpos:
        raise ValueError("LongNet context encoder expects use_xpos=True to remain aligned with dilated attention pipeline.")
    seg_list = list(seg)
    dil_list = list(dil)
    if len(seg_list) != len(dil_list):
        raise ValueError(
            f"segment_length ({len(seg_list)}) and dilated_ratio ({len(dil_list)}) must have identical lengths."
        )

    args = EncoderConfig(
        encoder_embed_dim=d_model,
        encoder_attention_heads=num_heads,
        encoder_ffn_embed_dim=int(d_model * mlp_ratio),
        encoder_layers=n_layers,
        encoder_normalize_before=True,
        normalize_output=True,
        activation_fn="gelu",
        dropout=dropout,
        drop_path_rate=drop_path_rate,
        attention_dropout=0.0,
        activation_dropout=0.0,
        no_scale_embedding=True,
        layernorm_embedding=False,
        moe_freq=0,
        moe_expert_count=0,
        subln=True,
        flash_attention=True,
        xpos_rel_pos=use_xpos,
        xpos_scale_base=512,
        segment_length=seg_list,
        dilated_ratio=dil_list,
        seq_parallel=False,
        no_output_layer=True,
        vocab_size=0,
    )
    return LongNetContextEncoder(
        args,
        layer_segments=layer_segments,
        layer_dilations=layer_dilations,
    )


__all__ = ["make_longnet_context", "LongNetContextEncoder"]
