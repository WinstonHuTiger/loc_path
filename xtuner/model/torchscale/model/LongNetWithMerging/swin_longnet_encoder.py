from typing import Optional, List, Dict, Union
import torch
import os
import sys, math

this_file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(this_file_dir, '../../../'))

from .sparse_patch_merging import SparsePatchMerging
from .relative_position_bias_2d import RelativePositionBias2D
from torchscale.architecture.utils import init_bert_params
from torchscale.component.multiway_network import set_split_position
from torchscale.architecture.encoder import Encoder, EncoderLayer
from torchscale.component.dilated_attention import DilatedAttention
from fairscale.nn import checkpoint_wrapper, wrap
from mmengine import print_log


class LongNetEncoderLayer(EncoderLayer):
    def build_self_attention(self, embed_dim, args):
        return DilatedAttention(
            args,
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            encoder_decoder_attention=False,
            subln=args.subln,
        )


class LongNetWithPatchMerging(Encoder):
    def __init__(
        self,
        args,
        embed_tokens=None,
        embed_positions=None,
        output_projection=None,
        is_encoder_decoder=False,
        keep_dim_after_merge: bool = True,
        merge_size: int = 2,
        use_rel_pos_2d: bool = True,
        rel_pos_num_buckets: int = 32,
        rel_pos_max_distance: int = 512,
        # NEW: cache-control knobs (match SparsePatchMerging)
        clear_after_forward: bool = False,   # clear CUDA cache after each forward
        clear_on_oom: bool = True,           # clear & retry once on CUDA OOM
        **kwargs,
    ):  
        self.merge_size = merge_size
        self.keep_dim_after_merge = keep_dim_after_merge
        super().__init__(
            args=args,
            embed_tokens=embed_tokens,
            embed_positions=embed_positions,
            output_projection=output_projection,
            is_encoder_decoder=is_encoder_decoder,
            **kwargs
        )

        self.embed_dim = args.encoder_embed_dim
        
        
        # NEW: store cache knobs
        self.clear_after_forward = clear_after_forward
        self.clear_on_oom = clear_on_oom
        self._retrying = False  # guard to avoid infinite OOM loops

        # build patch merge module
        self.patch_merge = self.build_patch_merge(
            args=args,
            embed_dim=self.embed_dim,
            keep_dim=keep_dim_after_merge,
            merge_size=merge_size,
        )

        self.use_rel_pos_2d = use_rel_pos_2d
        self.rel_pos_num_buckets = rel_pos_num_buckets
        self.rel_pos_max_distance = rel_pos_max_distance

        if self.use_rel_pos_2d:
            n_heads = args.encoder_attention_heads
            self.relpos2d = RelativePositionBias2D(
                n_heads=n_heads,
                num_buckets=rel_pos_num_buckets,
                max_distance=rel_pos_max_distance,
                bidirectional=True,
            )
        else:
            self.relpos2d = None

        if args.bert_init:
            self.apply(init_bert_params)

        if args.deepnorm:
            if is_encoder_decoder:
                init_scale = (
                    math.pow(
                        math.pow(args.encoder_layers, 4) * args.decoder_layers, 0.0625
                    )
                    / 1.15
                )
            else:
                init_scale = math.pow(8.0 * args.encoder_layers, 0.25)
            for name, p in self.named_parameters():
                if 'reduction' in name or 'reduction1' in name or 'reduction2' in name:
                    p.data.mul_(init_scale)

        if args.subln:
            if is_encoder_decoder:
                init_scale = math.sqrt(
                    math.log(3 * args.decoder_layers)
                    * math.log(2 * args.encoder_layers)
                    / 3
                )
            else:
                init_scale = math.sqrt(math.log(args.encoder_layers * 2))
            for name, p in self.named_parameters():
                if 'reduction' in name or 'reduction1' in name or 'reduction2' in name:
                    p.data.mul_(init_scale)

    # NEW: utility to clear CUDA cache
    def _clear_cuda_cache(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass

    def build_patch_merge(self, args, embed_dim, keep_dim, merge_size):
        merger = SparsePatchMerging(
            args=args,
            embed_dim=embed_dim,
            layernorm_eps=args.layernorm_eps,
            merge_size=merge_size,
            keep_dim=keep_dim,
            subln=args.subln,
        )
        if args.checkpoint_activations:
            merger = checkpoint_wrapper(merger)
        if args.fsdp:
            merger = wrap(merger)
        return merger

    def build_encoder_layer(
        self, args, depth, is_moe_layer=False, is_encoder_decoder=False
    ):
        
        layer = LongNetEncoderLayer(
            args,
            depth,
            is_moe_layer=is_moe_layer,
            is_encoder_decoder=is_encoder_decoder,
        )
        if args.checkpoint_activations:
            layer = checkpoint_wrapper(layer)
        if args.fsdp:
            layer = wrap(layer)
        return layer

    # NEW: factor original body into _forward_impl so we can OOM-retry & finally-clean
    def _forward_impl(
        self,
        src_tokens,
        encoder_padding_mask=None,
        attn_mask=None,
        return_all_hiddens=False,
        token_embeddings=None,
        multiway_split_position=None,
        features_only=False,
        incremental_state=None,
        positions=None,
        coords: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        **kwargs,
    ):
        try:
            assert src_tokens is not None or token_embeddings is not None

            if encoder_padding_mask is None:
                if src_tokens is not None:
                    encoder_padding_mask = torch.zeros_like(
                        src_tokens, device=src_tokens.device
                    ).bool()
                else:
                    encoder_padding_mask = torch.zeros(
                        [token_embeddings.size(0), token_embeddings.size(1)],
                        device=token_embeddings.device,
                    ).bool()

            if multiway_split_position is not None:
                assert self.args.multiway
                self.apply(set_split_position(multiway_split_position))

            x, encoder_embedding = self.forward_embedding(
                src_tokens, token_embeddings, positions
            )
            x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

            B, L, C = x.shape
            # print('input x shape', x.shape)
            encoder_states = []
            if return_all_hiddens:
                encoder_states.append(x)

            rel_pos_bias = None
            if self.relative_position is not None:
                rel_pos_bias = self.relative_position(
                    batch_size=x.size(0), qlen=x.size(1), klen=x.size(1)
                )

            if coords is None:
                coords_rc = None
            else:
                # Accept: list[tensor], [L,2] tensor, or [B,L,2] tensor
                coords_t = coords[0] if isinstance(coords, list) else coords
                Bx = x.size(0)  # actual batch size of inputs

                if not torch.is_tensor(coords_t):
                    raise ValueError("coords must be a Tensor or list[Tensor].")

                if coords_t.dim() == 2:
                    # [L, 2]
                    coords_rc = coords_t
                elif coords_t.dim() == 3:
                    # [B, L, 2] -> ensure B matches and either B==1 or all examples share coords
                    if coords_t.size(0) != Bx:
                        raise ValueError(f"coords batch dim mismatch: got {coords_t.size(0)} but inputs have B={Bx}")
                    if Bx == 1:
                        coords_rc = coords_t[0]
                    else:
                        # require same coords across the batch (cheap equality check)
                        if not torch.equal(coords_t, coords_t[0].unsqueeze(0).expand_as(coords_t)):
                            raise NotImplementedError(
                                "Per-example coords (varying across batch) are not supported by the current "
                                "patch-merging/layout path. Use batch size 1 or share coords across the batch."
                            )
                        coords_rc = coords_t[0]
                else:
                    raise ValueError("coords must have shape [L,2] or [B,L,2].")

                if coords_rc.size(-1) != 2:
                    raise ValueError("coords last dimension must be 2.")

            new_coords = coords_rc
            new_mask = encoder_padding_mask
            num_layers = self.num_layers
            if incremental_state is None:
                incremental_state = [None] * num_layers

            l_aux = []
            for idx, layer in enumerate(self.layers):
                rel_pos = None
                if new_coords is not None and self.use_rel_pos_2d:
                    rel_pos = self._coords_to_relpos(B, new_coords)

                cur_mask = new_mask if incremental_state is None else None
                x, l_aux_i = layer(
                    x,
                    encoder_padding_mask=cur_mask,
                     attn_mask=attn_mask,
                     rel_pos=rel_pos,
                     multiway_split_position=multiway_split_position,
                 )
                # free per-layer rel_pos ASAP
                del rel_pos

                if return_all_hiddens:
                    encoder_states.append(x)
                l_aux.append(l_aux_i)

                # patch merge after activation, except final layer
                if idx < num_layers - 1 and new_coords is not None:
                    x, new_coords, new_mask = self.patch_merge(
                        x, self._coords_to_rowcol(new_coords), new_mask
                    )
                    # del _aux  # not needed downstream
            # print_log(f'x shape {x.shape}', 'current')
            if self.layer_norm is not None:
                x = self.layer_norm(x)

            if not features_only and self.output_projection is not None:
                x = self.output_projection(x)

            out = {
                "encoder_out": x,
                "encoder_embedding": encoder_embedding,
                "encoder_padding_mask": encoder_padding_mask,
                "encoder_states": encoder_states,
                "l_aux": l_aux,
                'coords': new_coords
            }
            return out
        finally:
            # === FINAL CLEANUP ===
            # Explicitly drop coord tensors to help the allocator reclaim earlier.
            if 'coords' in locals():
                del coords
            if 'coords_rc' in locals():
                del coords_rc
            if 'rel_pos_bias' in locals():
                del rel_pos_bias
            # (x/encoder_states are returned, so we can't drop them here.)

    def forward(
        self,
        src_tokens,
        encoder_padding_mask=None,
        attn_mask=None,
        return_all_hiddens=False,
        token_embeddings=None,
        multiway_split_position=None,
        features_only=False,
        incremental_state=None,
        positions=None,
        coords: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        **kwargs,
    ):
        try:
            return self._forward_impl(
                src_tokens,
                encoder_padding_mask=encoder_padding_mask,
                attn_mask=attn_mask,
                return_all_hiddens=return_all_hiddens,
                token_embeddings=token_embeddings,
                multiway_split_position=multiway_split_position,
                features_only=features_only,
                incremental_state=incremental_state,
                positions=positions,
                coords=coords,
                **kwargs,
            )
        except RuntimeError as e:
            msg = str(e).lower()
            is_cuda_oom = ("cuda out of memory" in msg) or ("cublas" in msg and "alloc" in msg)
            if (
                self.clear_on_oom and is_cuda_oom and torch.cuda.is_available()
                and not self._retrying
            ):
                self._retrying = True
                self._clear_cuda_cache()
                try:
                    torch.cuda.reset_peak_memory_stats()
                except Exception:
                    pass
                try:
                    return self._forward_impl(
                        src_tokens,
                        encoder_padding_mask=encoder_padding_mask,
                        attn_mask=attn_mask,
                        return_all_hiddens=return_all_hiddens,
                        token_embeddings=token_embeddings,
                        multiway_split_position=multiway_split_position,
                        features_only=features_only,
                        incremental_state=incremental_state,
                        positions=positions,
                        coords=coords,
                        **kwargs,
                    )
                finally:
                    self._retrying = False
            raise
        finally:
            # Optional cache cleanup after a successful (or failed) forward
            if self.clear_after_forward and torch.cuda.is_available():
                self._clear_cuda_cache()

    @staticmethod
    def _coords_to_rowcol(coords_xy: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = coords_xy[:, 0]
            y = coords_xy[:, 1]
            x_for_unique = x
            y_for_unique = y
            if x_for_unique.dtype.is_floating_point:
                x_for_unique = x_for_unique.round().to(torch.int)
                y_for_unique = y_for_unique.round().to(torch.int)
            x_sorted = torch.unique(x_for_unique, sorted=True)
            y_sorted = torch.unique(y_for_unique, sorted = True)
            
            col = torch.searchsorted(x_sorted, x)
            row = torch.searchsorted(y_sorted, y)
            return torch.stack([row, col], dim=-1)

    def _coords_to_relpos(self, B: int, coords_xy: torch.Tensor):
        rc = self._coords_to_rowcol(coords_xy)
        return self._compute_relpos(B, rc)

    def _compute_relpos(self, B: int, coords_rc: torch.Tensor):
        if not self.use_rel_pos_2d:
            return None
        return self.relpos2d(B, coords_rc, coords_rc)

# class LongNetEncoderLayer(EncoderLayer):

#     def build_self_attention(self, embed_dim, args):
#         return DilatedAttention(
#             args,
#             embed_dim,
#             args.encoder_attention_heads,
#             dropout=args.attention_dropout,
#             self_attention=True,
#             encoder_decoder_attention=False,
#             subln=args.subln,
#         )


# class LongNetWithPatchMerging(Encoder):
#     def __init__(
#         self,
#         args,
#         embed_tokens=None,
#         embed_positions=None,
#         output_projection=None,
#         is_encoder_decoder=False,
#         keep_dim_after_merge: bool = True,
#         merge_size: int = 2,
#         use_rel_pos_2d: bool = True,
#         rel_pos_num_buckets: int = 32,
#         rel_pos_max_distance: int = 512,
#         **kwargs,
#     ):
#         # call the encoder's init
#         super().__init__(
#             args = args, 
#             embed_tokens=embed_tokens,
#             embed_positions= embed_positions,
#             output_projection= output_projection,
#             is_encoder_decoder= is_encoder_decoder,
#             **kwargs
#         )

#         self.embed_dim = args.encoder_embed_dim
#         # build patch merge module
#         self.merge_size = merge_size
#         self.keep_dim_after_merge = keep_dim_after_merge
#         self.patch_merge = self.build_patch_merge(args = args,
#                                                   embed_dim= self.embed_dim,
#                                                   keep_dim= keep_dim_after_merge,
#                                                   merge_size= merge_size
#                                                   )
     
        
#         self.use_rel_pos_2d = use_rel_pos_2d
#         self.rel_pos_num_buckets = rel_pos_num_buckets
#         self.rel_pos_max_distance = rel_pos_max_distance
        
#         if self.use_rel_pos_2d:
#             n_heads = args.encoder_attention_heads
#             self.relpos2d = RelativePositionBias2D(
#                 n_heads=n_heads,
#                 num_buckets=rel_pos_num_buckets,
#                 max_distance=rel_pos_max_distance,
#                 bidirectional=True,
#             )
#         else:
#             self.relpos2d = None
        
#         # do the initialization for patch merger
#         if args.bert_init:
#             self.apply(init_bert_params)
            
        
#         if args.deepnorm:
#             if is_encoder_decoder:
#                 init_scale = (
#                     math.pow(
#                         math.pow(args.encoder_layers, 4) * args.decoder_layers, 0.0625
#                     )
#                     / 1.15
#                 )
#             else:
#                 init_scale = math.pow(8.0 * args.encoder_layers, 0.25)
#             for name, p in self.named_parameters():
#                 if (
#                     'reduction' in name
#                 ):
#                     p.data.div_(init_scale)

#         if args.subln:
#             if is_encoder_decoder:
#                 init_scale = math.sqrt(
#                     math.log(3 * args.decoder_layers)
#                     * math.log(2 * args.encoder_layers)
#                     / 3
#                 )
#             else:
#                 init_scale = math.sqrt(math.log(args.encoder_layers * 2))
#             for name, p in self.named_parameters():
#                 if (
#                     'reduction' in name
#                 ):
#                     p.data.mul_(init_scale)
        
        
       
        
#     def build_patch_merge(self, args, embed_dim, keep_dim, merge_size):
#         merger = SparsePatchMerging(
#             args = args, 
#             embed_dim= embed_dim,
#             layernorm_eps = args.layernorm_eps,
#             merge_size= merge_size,
#             keep_dim = keep_dim,
#             subln= args.subln
#             )
#         if args.checkpoint_activations:
#             merger = checkpoint_wrapper(merger)
#         if args.fsdp:
#             merger = wrap(merger)
#         return merger
    
#     # build longnet encoder layer
#     def build_encoder_layer(
#         self, args, depth, is_moe_layer=False, is_encoder_decoder=False
#     ):
#         layer = LongNetEncoderLayer(
#             args,
#             depth,
#             is_moe_layer=is_moe_layer,
#             is_encoder_decoder=is_encoder_decoder,
#         )
#         if args.checkpoint_activations:
#             layer = checkpoint_wrapper(layer)
#         if args.fsdp:
#             layer = wrap(layer)
#         return layer

#     def forward(
#         self,
#         src_tokens,
#         encoder_padding_mask=None,
#         attn_mask=None,
#         return_all_hiddens=False,
#         token_embeddings=None,
#         multiway_split_position=None,
#         features_only=False,
#         incremental_state=None,
#         positions=None,
        
#         # coords from clam framework
        
#         coords: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
#         **kwargs
#     ):
#         assert src_tokens is not None or token_embeddings is not None

#         if encoder_padding_mask is None:
#             if src_tokens is not None:
#                 encoder_padding_mask = torch.zeros_like(
#                     src_tokens, device=src_tokens.device
#                 ).bool()
#             else:
#                 encoder_padding_mask = torch.zeros(
#                     [token_embeddings.size(0), token_embeddings.size(1)],
#                     device=token_embeddings.device,
#                 ).bool()

#         if multiway_split_position is not None:
#             assert self.args.multiway
#             self.apply(set_split_position(multiway_split_position))

#         x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings, positions)
#         x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))
        
#         B, L, C = x.shape
#         encoder_states = []

#         if return_all_hiddens:
#             encoder_states.append(x)

#         rel_pos_bias = None
#         if self.relative_position is not None:
#             rel_pos_bias = self.relative_position(
#                 batch_size=x.size(0), qlen=x.size(1), klen=x.size(1)
#             )
        
#         if coords is None:
#             coords_rc = None
#         else:
#             coords_rc = coords[0] if isinstance(coords, list) else coords
#             if coords_rc.dim() != 2 or coords_rc.size(-1) != 2:
#                 raise ValueError("coords must be [L, 2] tensor")

#         new_coords = coords_rc
#         new_mask = encoder_padding_mask
#         num_layers = self.num_layers
#         if incremental_state is None:
#             incremental_state = [None] * num_layers

#         # incremental_state is not None during inference if we use the bidirectional encoder as a generator as in s2s-ft (https://arxiv.org/abs/2110.13640)
#         l_aux = []
#         for idx, layer in enumerate(self.layers):
#             rel_pos = None
#             if new_coords is not None and self.use_rel_pos_2d:
#                 rel_pos = self._coords_to_relpos(B, new_coords)
            
#             x, l_aux_i = layer(
#                 x,
#                 encoder_padding_mask=encoder_padding_mask if incremental_state is None else None,
#                 attn_mask=attn_mask,
#                 rel_pos=rel_pos_bias if not self.use_rel_pos_2d else rel_pos,
#                 multiway_split_position=multiway_split_position,
#                 incremental_state=incremental_state[idx] if incremental_state is not None else None,
#             )
            
#             if return_all_hiddens:
#                 assert encoder_states is not None
#                 encoder_states.append(x)
#             l_aux.append(l_aux_i)
        
#             # patch merge after activation, do this except the final layer
#             if idx < num_layers - 1 and new_coords is not None:
#                 x, new_coords, new_mask, _aux = self.patch_merge(x, self._coords_to_rowcol(new_coords), new_mask)

#         if self.layer_norm is not None:
#             x = self.layer_norm(x)

#         if not features_only and self.output_projection is not None:
#             x = self.output_projection(x)

        
#         return {
#             "encoder_out": x,
#             "encoder_embedding": encoder_embedding,
#             "encoder_padding_mask": encoder_padding_mask,
#             "encoder_states": encoder_states,
#             "l_aux": l_aux,
#         }
    
#     @staticmethod
#     def _coords_to_rowcol(coords_xy: torch.Tensor) -> torch.Tensor:
#         with torch.no_grad():
#             x = coords_xy[:, 0]
#             y = coords_xy[:, 1]
#             x_sorted = x.unique(sorted=True)
#             y_sorted = y.unique(sorted=True)
#             col = torch.searchsorted(x_sorted, x)
#             row = torch.searchsorted(y_sorted, y)
#             return torch.stack([row, col], dim=-1)

#     def _coords_to_relpos(self, B: int, coords_xy: torch.Tensor):
#         rc = self._coords_to_rowcol(coords_xy)
#         return self._compute_relpos(B, rc)
    


    
# class LongNetWithPatchMerging(nn.Module):
#     def __init__(
#         self,
#         backbone_encoder: nn.Module,
#         keep_dim_after_merge: bool = True,
#         merge_size: int = 2,
#         use_rel_pos_2d: bool = True,
#         rel_pos_num_buckets: int = 32,
#         rel_pos_max_distance: int = 512,
#     ):
#         super().__init__()
#         self.backbone = backbone_encoder
#         self.layers = backbone_encoder.layers
#         self.num_layers = getattr(backbone_encoder, "num_layers", len(self.layers))
#         self.embed_dim = getattr(backbone_encoder, "embed_dim", None) or backbone_encoder.args.encoder_embed_dim
#         self.keep_dim_after_merge = keep_dim_after_merge
#         self.merge_size = merge_size

#         self.patch_merge = SparsePatchMerging(self.embed_dim, keep_dim=keep_dim_after_merge, merge_size=merge_size)
#         self.merge_proj_back = None
#         if not keep_dim_after_merge:
#             self.merge_proj_back = nn.Linear(self.patch_merge.out_dim, self.embed_dim)

#         self.use_rel_pos_2d = use_rel_pos_2d
#         if use_rel_pos_2d:
#             n_heads = backbone_encoder.args.encoder_attention_heads
#             self.relpos2d = RelativePositionBias2D(
#                 n_heads=n_heads,
#                 num_buckets=rel_pos_num_buckets,
#                 max_distance=rel_pos_max_distance,
#                 bidirectional=True,
#             )
#         else:
#             self.relpos2d = None

#         for name in ["embed_tokens", "embed_positions", "layer_norm", "output_projection", "args"]:
#             if hasattr(backbone_encoder, name):
#                 setattr(self, name, getattr(backbone_encoder, name))

#     def forward_embedding(self, *args, **kwargs):
#         return self.backbone.forward_embedding(*args, **kwargs)

#     def _compute_relpos(self, B: int, coords_rc: torch.Tensor):
#         if not self.use_rel_pos_2d:
#             return None
#         return self.relpos2d(B, coords_rc, coords_rc)

#     def forward(
#         self,
#         src_tokens=None,
#         encoder_padding_mask: Optional[torch.Tensor] = None,
#         attn_mask: Optional[torch.Tensor] = None,
#         return_all_hiddens: bool = False,
#         token_embeddings: Optional[torch.Tensor] = None,
#         multiway_split_position=None,
#         features_only: bool = False,
#         incremental_state: Optional[List[Dict]] = None,
#         positions=None,
#         coords: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
#         **kwargs,
#     ):
#         assert src_tokens is not None or token_embeddings is not None

#         if encoder_padding_mask is None:
#             if token_embeddings is not None:
#                 encoder_padding_mask = torch.zeros(
#                     [token_embeddings.size(0), token_embeddings.size(1)],
#                     device=token_embeddings.device,
#                     dtype=torch.bool,
#                 )
#             else:
#                 encoder_padding_mask = torch.zeros(
#                     [src_tokens.size(0), src_tokens.size(1)],
#                     device=src_tokens.device,
#                     dtype=torch.bool,
#                 )

#         x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings, positions)
#         x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

#         B, L, C = x.shape
#         encoder_states = []
#         l_aux = []

#         if return_all_hiddens:
#             encoder_states.append(x)

#         if coords is None:
#             coords_rc = None
#         else:
#             coords_rc = coords[0] if isinstance(coords, list) else coords
#             if coords_rc.dim() != 2 or coords_rc.size(-1) != 2:
#                 raise ValueError("coords must be [L, 2] tensor")

#         new_coords = coords_rc
#         new_mask = encoder_padding_mask
#         num_layers = self.num_layers
#         if incremental_state is None:
#             incremental_state = [None] * num_layers

#         for idx, layer in enumerate(self.layers):
#             rel_pos = None
#             if new_coords is not None and self.use_rel_pos_2d:
#                 rel_pos = self._coords_to_relpos(B, new_coords)

#             x, l_aux_i = layer(
#                 x,
#                 new_mask,
#                 attn_mask=attn_mask,
#                 rel_pos=rel_pos,
#                 multiway_split_position=multiway_split_position,
#                 incremental_state=incremental_state[idx],
#             )
#             if return_all_hiddens:
#                 encoder_states.append(x)
#             l_aux.append(l_aux_i)

#             if idx < num_layers - 1 and new_coords is not None:
#                 x, new_coords, new_mask, _aux = self.patch_merge(x, self._coords_to_rowcol(new_coords), new_mask)
#                 if self.merge_proj_back is not None:
#                     x = self.merge_proj_back(x)

#         if hasattr(self, "layer_norm") and self.layer_norm is not None:
#             x = self.layer_norm(x)
#         if not features_only and hasattr(self, "output_projection") and self.output_projection is not None:
#             x = self.output_projection(x)

#         return {
#             "encoder_out": x,
#             "encoder_embedding": encoder_embedding,
#             "encoder_padding_mask": new_mask,
#             "encoder_states": encoder_states,
#             "l_aux": l_aux,
#             "coords": new_coords,
#         }

#     @staticmethod
#     def _coords_to_rowcol(coords_xy: torch.Tensor) -> torch.Tensor:
#         x = coords_xy[:, 0]
#         y = coords_xy[:, 1]
#         x_sorted = x.unique(sorted=True)
#         y_sorted = y.unique(sorted=True)
#         col = torch.searchsorted(x_sorted, x)
#         row = torch.searchsorted(y_sorted, y)
#         return torch.stack([row, col], dim=-1)

#     def _coords_to_relpos(self, B: int, coords_xy: torch.Tensor):
#         rc = self._coords_to_rowcol(coords_xy)
#         return self._compute_relpos(B, rc)
