
# from typing import Optional, Dict, Tuple
# import torch
# import torch.nn as nn
# try:
#     from apex.normalization import FusedLayerNorm as LayerNorm
# except ModuleNotFoundError:
#     from torch.nn import LayerNorm


# class SparsePatchMerging(nn.Module):
#     def __init__(
#         self,
#         args,
#         embed_dim: int,
#         layernorm_eps: float,
#         keep_dim: bool = True,
#         merge_size: int = 2,
#         out_dim: Optional[int] = None,
        
#         subln = False,
#     ):
#         super().__init__()
#         self.args = args
        
#         assert merge_size == 2, "Only 2x2 merging is supported (Swin-style)."

#         self.embed_dim = embed_dim
#         self.merge_size = merge_size
#         self.keep_dim = keep_dim
#         # self.use_pre_norm = use_pre_norm

#         in_dim = embed_dim * (merge_size ** 2)
#         if out_dim is None:
#             out_dim = embed_dim if keep_dim else (2 * embed_dim)
#         self.out_dim = out_dim

#         self.layer_norm = LayerNorm(in_dim, eps=layernorm_eps) if subln else None
#         self.reduction = nn.Linear(in_dim, out_dim)
        
        
        
#     def reset_parameters(self):
#         # self.fc1.reset_parameters()
#         nn.init.xavier_uniform_(self.reduction.weight)
#         if self.layer_norm is not None:
#             self.layer_norm.reset_parameters()

#     def forward(
#         self,
#         x: torch.Tensor,                 # [B, L, C]
#         coords_rc: torch.Tensor,         # [L, 2] (row, col) integer indices OR raw coords (we will rank)
#         padmask: Optional[torch.Tensor]  # [B, L] bool
#     ):
#         B, L, C = x.shape
#         device = x.device

#         if coords_rc.dtype != torch.long and coords_rc.dtype != torch.int64:
#             coords_rc = coords_rc.to(torch.long)
#         rows = coords_rc[:, 0]
#         cols = coords_rc[:, 1]
#         rows_sorted = rows.unique(sorted=True)
#         cols_sorted = cols.unique(sorted=True)
#         row_idx = torch.searchsorted(rows_sorted, rows)
#         col_idx = torch.searchsorted(cols_sorted, cols)
#         rc_idx = torch.stack([row_idx, col_idx], dim=-1)  # [L, 2]

#         r_group = rc_idx[:, 0] // self.merge_size
#         c_group = rc_idx[:, 1] // self.merge_size
#         group_pairs = torch.stack([r_group, c_group], dim=-1)                      # [L, 2]
#         unique_groups, inverse = torch.unique(group_pairs, dim=0, return_inverse=True)
        
#         G = unique_groups.size(0)
#         M = self.merge_size * self.merge_size  # 4
        
#         # print("SparsePatchMerging: Merging {} tokens into {} groups.".format(L, G))
#         # print('inverse:', inverse)
        
#         r_off = rc_idx[:, 0] % self.merge_size
#         c_off = rc_idx[:, 1] % self.merge_size
#         offset_id = r_off * self.merge_size + c_off                                  # [L] in {0,1,2,3}
#         slot_idx = inverse * M + offset_id                                           # [L] in [0, G*M)
        
#         # print('slot_idx shape:', slot_idx.shape)
#         merged_slots = x.new_zeros(B, G * M, C)
#         for b in range(B):
#             merged_slots[b].index_add_(0, slot_idx, x[b])
        
#         merged_slots = merged_slots.view(B, G, M, C)
#         merged_concat = merged_slots.flatten(2)  # [B, G, 4*C]
#         if self.layer_norm is not None:
#             out = self.reduction(self.layer_norm(merged_concat))  # [B, G, C_out]
#         else:
#             out = self.reduction(merged_concat)
        
#         # recalculate the padding mask
#         if padmask is not None:
#             valid = (~padmask).to(x.dtype)  # [B, L] in {0,1}
#             group_valid = valid.new_zeros(B, G * M)
#             for b in range(B):
#                 group_valid[b].index_add_(0, slot_idx, valid[b])
#             group_valid_any = group_valid.view(B, G, M).sum(dim=-1) > 0  # [B, G] bool
#             padmask_out = ~group_valid_any
#         else:
#             padmask_out = None

#         aux = {
#             "unique_groups": unique_groups,  # [G, 2] rows//2, cols//2
#             "slot_idx": slot_idx,            # [L]
#             "inverse": inverse,              # [L] group index per token
#         }

#         coords_out = unique_groups
#         return out, coords_out, padmask_out, aux


# from typing import Optional
# import torch
# import torch.nn as nn
# try:
#     from apex.normalization import FusedLayerNorm as LayerNorm
# except ModuleNotFoundError:
#     from torch.nn import LayerNorm


# class SparsePatchMerging(nn.Module):
#     def __init__(
#         self,
#         args,
#         embed_dim: int,
#         layernorm_eps: float,
#         keep_dim: bool = True,
#         merge_size: int = 2,
#         out_dim: Optional[int] = None,
#         subln: bool = False,
#         # NEW: cache-control knobs
#         clear_after_forward: bool = False,  # clear CUDA cache after each forward
#         clear_on_oom: bool = True,          # clear & retry once on OOM
#     ):
#         super().__init__()
#         self.args = args
#         self.clear_after_forward = clear_after_forward
#         self.clear_on_oom = clear_on_oom
#         self._retrying = False  # guard to avoid infinite recursion on OOM

#         assert merge_size == 2, "Only 2x2 merging is supported (Swin-style)."

#         self.embed_dim = embed_dim
#         self.merge_size = merge_size
#         self.keep_dim = keep_dim

#         in_dim = embed_dim * (merge_size ** 2)
#         if out_dim is None:
#             out_dim = embed_dim if keep_dim else (2 * embed_dim)
#         self.out_dim = out_dim

#         self.layer_norm = LayerNorm(in_dim, eps=layernorm_eps) if subln else None
#         self.reduction = nn.Linear(in_dim, out_dim)

#     def _clear_cuda_cache(self):
#         if torch.cuda.is_available():
#             # Release cached blocks back to the driver, and collect any stray IPC handles.
#             torch.cuda.empty_cache()
#             try:
#                 torch.cuda.ipc_collect()
#             except Exception:
#                 # ipc_collect may not be available on some builds/devices; ignore.
#                 pass

#     def reset_parameters(self):
#         nn.init.xavier_uniform_(self.reduction.weight)
#         if self.layer_norm is not None:
#             self.layer_norm.reset_parameters()

#     def _forward_impl(
#         self,
#         x: torch.Tensor,                 # [B, L, C]
#         coords_rc: torch.Tensor,         # [L, 2] or [B, L, 2]
#         padmask: Optional[torch.Tensor]  # [B, L] bool
#     ):
#         B, L, C = x.shape
#         device = x.device

#         # --- NEW: normalize coords_rc to [L,2] ---
#         if coords_rc.dim() == 3:
#             if coords_rc.size(0) != B:
#                 raise ValueError(f"coords batch dim mismatch: got {coords_rc.size(0)} but inputs have B={B}")
#             if B == 1:
#                 coords_rc = coords_rc[0]
#             else:
#                 if not torch.equal(coords_rc, coords_rc[0].unsqueeze(0).expand_as(coords_rc)):
#                     raise NotImplementedError(
#                         "Per-example coords (varying across batch) are not supported by the current "
#                         "merging implementation. Use batch size 1 or share coords across the batch."
#                     )
#                 coords_rc = coords_rc[0]
#         elif coords_rc.dim() != 2:
#             raise ValueError("coords_rc must be [L,2] or [B,L,2].")
#         if coords_rc.size(-1) != 2:
#             raise ValueError("coords_rc last dimension must be 2.")
#         # --- end NEW ---

#         if coords_rc.dtype not in (torch.long, torch.int64):
#             coords_rc = coords_rc.to(torch.long)

#         # Pure index math; ensure no autograd tracking (saves tiny overhead).
#         with torch.no_grad():
#             rows = coords_rc[:, 0]
#             cols = coords_rc[:, 1]
#             rows_sorted = rows.unique(sorted=True)
#             cols_sorted = cols.unique(sorted=True)
#             row_idx = torch.searchsorted(rows_sorted, rows)
#             col_idx = torch.searchsorted(cols_sorted, cols)
#             rc_idx = torch.stack([row_idx, col_idx], dim=-1)  # [L, 2]

#             r_group = rc_idx[:, 0] // self.merge_size
#             c_group = rc_idx[:, 1] // self.merge_size
#             group_pairs = torch.stack([r_group, c_group], dim=-1)  # [L, 2]
#             unique_groups, inverse = torch.unique(group_pairs, dim=0, return_inverse=True)

#             G = unique_groups.size(0)
#             M = self.merge_size * self.merge_size  # 4

#             r_off = rc_idx[:, 0] % self.merge_size
#             c_off = rc_idx[:, 1] % self.merge_size
#             offset_id = r_off * self.merge_size + c_off  # [L] in {0,1,2,3}
#             slot_idx = (inverse * M + offset_id).to(device)  # ensure device alignment

#         # Build merged slots
#         merged_slots = x.new_zeros(B, G * M, C)
#         for b in range(B):
#             merged_slots[b].index_add_(0, slot_idx, x[b])

#         merged_slots = merged_slots.view(B, G, M, C)
#         merged_concat = merged_slots.flatten(2)  # [B, G, 4*C]
#         if self.layer_norm is not None:
#             out = self.reduction(self.layer_norm(merged_concat))  # [B, G, C_out]
#         else:
#             out = self.reduction(merged_concat)

#         # recalculate the padding mask
#         padmask_out = None
#         if padmask is not None:
#             valid = (~padmask).to(x.dtype)  # [B, L] in {0,1}
#             group_valid = valid.new_zeros(B, G * M)
#             for b in range(B):
#                 group_valid[b].index_add_(0, slot_idx, valid[b])
#             group_valid_any = group_valid.view(B, G, M).sum(dim=-1) > 0  # [B, G] bool
#             padmask_out = ~group_valid_any

#         # Prepare aux (kept lightweight; none of these require grad)
#         aux = {
#             "unique_groups": unique_groups.to(device),  # [G, 2]
#             "slot_idx": slot_idx,                       # [L]
#             "inverse": inverse.to(device),              # [L]
#         }

#         # Proactively drop large temporaries before returning
#         del merged_slots, merged_concat
#         if padmask is not None:
#             del group_valid
#             del group_valid_any
#             del valid

#         # Optional allocator cleanup after forward
#         if self.clear_after_forward and x.is_cuda:
#             self._clear_cuda_cache()

#         coords_out = unique_groups
#         return out, coords_out, padmask_out, aux

#     def forward(
#         self,
#         x: torch.Tensor,
#         coords_rc: torch.Tensor,
#         padmask: Optional[torch.Tensor]
#     ):
#         try:
#             return self._forward_impl(x, coords_rc, padmask)
#         except RuntimeError as e:
#             msg = str(e).lower()
#             is_cuda_oom = ("cuda out of memory" in msg) or ("cublas" in msg and "alloc" in msg)
#             if self.clear_on_oom and is_cuda_oom and x.is_cuda and not self._retrying:
#                 # Clear cache and retry once
#                 self._retrying = True
#                 self._clear_cuda_cache()
#                 torch.cuda.reset_peak_memory_stats()
#                 try:
#                     return self._forward_impl(x, coords_rc, padmask)
#                 finally:
#                     self._retrying = False
#             raise


# sparse_patch_merging.py
from typing import Optional, Tuple
import torch
import torch.nn as nn

class SparsePatchMerging(nn.Module):
    """
    Stable 2x2 (stride-2) patch merging for Swin/LongNet with:
      - Pre/Post LayerNorm (eps small) for bf16 stability
      - Deterministic TL/TR/BL/BR concat ordering
      - FP32 reductions for scatter/index_add
      - Variance-preserving "sum" option
      - Robust handling of ragged tokens via coords_rc + padmask
      - Optional clear-on-OOM retry

    forward(x: [B,L,C], coords_rc: [B,L,2] or [L,2], padmask: Optional[bool[B,L]])
      -> (x_merged: [B,L_out,C'], coords_merged: [B,L_out,2], padmask_merged: [B,L_out])
    """
    def __init__(
        self,
        args,
        embed_dim: int,
        layernorm_eps: float,
        keep_dim: bool = True,
        merge_size: int = 2,
        out_dim: Optional[int] = None,
        pre_norm: bool = True,
        post_norm: bool = True,
        fp32_reduce: bool = True,
        mode: str = 'concat',          # 'concat' or 'sum'
        clear_after_forward: bool = False,
        clear_on_oom: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        assert merge_size == 2, "Only 2x2 merging supported."
        self.args = args
        self.embed_dim = embed_dim
        self.merge_size = merge_size
        self.keep_dim = keep_dim
        self.clear_after_forward = clear_after_forward
        self.clear_on_oom = clear_on_oom
        self._retrying = False

        if out_dim is None:
            out_dim = embed_dim if keep_dim else (2 * embed_dim)
        self.out_dim = out_dim

        self.pre_norm = pre_norm
        self.post_norm = post_norm
        self.fp32_reduce = fp32_reduce
        self.mode = mode

        eps = layernorm_eps if layernorm_eps is not None else 1e-6
        self.ln_in = nn.LayerNorm(embed_dim, eps=eps) if pre_norm else None
        in_linear = embed_dim * 4 if mode == 'concat' else embed_dim
        # change the reduction to MLP instead of single linear layer
        # self.reduction1 = nn.Sequential([
        #     nn.Linear(in_linear, in_linear//2),
        #     nn.GELU(),
        #     nn.Linear(in_linear // 2, out_dim)
        # ])
        self.reduction1 = nn.Linear(in_linear, in_linear//2)
        self.act  = nn.GELU()
        self.reduction2 = nn.Linear(in_linear //2, out_dim)
        
        self.ln_out = nn.LayerNorm(out_dim, eps=eps) if post_norm else None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        
        nn.init.xavier_uniform_(self.reduction1.weight)
        nn.init.xavier_uniform_(self.reduction2.weight)
        
        if self.reduction1.bias is not None:
            nn.init.zeros_(self.reduction1.bias)
        if self.reduction2.bias is not None:
            nn.init.zeros_(self.reduction2.bias)
    
        if self.ln_in is not None:
            self.ln_in.reset_parameters()
        if self.ln_out is not None:
            self.ln_out.reset_parameters()
        

    def _clear_cuda_cache(self):
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass

    @staticmethod
    def _ensure_batched_coords(coords_rc: torch.Tensor, B: int) -> torch.Tensor:
        if coords_rc.dim() == 2:
            coords_rc = coords_rc.unsqueeze(0).expand(B, -1, -1)
        return coords_rc

    def _forward_impl(
        self,
        x: torch.Tensor,                 # [B, L, C]
        coords_rc: torch.Tensor,         # [B, L, 2] or [L, 2]
        padmask: Optional[torch.Tensor]  # [B, L] bool (True=pad)
    ):
        B, L, C = x.shape
        coords_rc = self._ensure_batched_coords(coords_rc, B)
        assert coords_rc.shape[:2] == (B, L)

        if self.ln_in is not None:
            x = self.ln_in(x)

        x_dtype = x.dtype
        red_dtype = torch.float32 if self.fp32_reduce else x.dtype

        out_x_list = []
        out_coords_list = []

        key_stride = int(2**20)  # enough for typical H,W<1e6

        for b in range(B):
            xb = x[b]                              # [L, C]
            coords_b = coords_rc[b].to(torch.long) # [L, 2]
            valid = torch.ones(L, dtype=torch.bool, device=x.device) if padmask is None else (~padmask[b])

            rb = coords_b[:, 0]
            cb = coords_b[:, 1]
            gr = torch.div(rb, self.merge_size, rounding_mode='floor')
            gc = torch.div(cb, self.merge_size, rounding_mode='floor')
            keys = gr * key_stride + gc

            # TL=0 (r%2=0,c%2=0), TR=1, BL=2, BR=3 — fixed ordering
            rmod = torch.remainder(rb, self.merge_size)
            cmod = torch.remainder(cb, self.merge_size)
            corner = (rmod << 1) | cmod

            sel = valid
            if sel.sum() == 0:
                out_x_list.append(xb.new_zeros(0, self.out_dim))
                out_coords_list.append(coords_b.new_zeros(0, 2))
                continue

            keys = keys[sel]
            corner = corner[sel]
            xb_sel = xb[sel]

            uniq, inv = torch.unique(keys, sorted=True, return_inverse=True)
            G = uniq.numel()

            # recover (gr,gc) from linear keys
            gc_out = torch.remainder(uniq, key_stride)
            gr_out = torch.div(uniq, key_stride, rounding_mode='floor')
            coords_out = torch.stack([gr_out, gc_out], dim=-1)  # [G,2]

            if self.mode == 'concat':
                # do FP32 accumulation, then cast back
                out_buf = torch.zeros(G, 4, C, device=x.device, dtype=red_dtype)
                counts = torch.zeros(G, 4, device=x.device, dtype=red_dtype)

                for k in range(4):
                    mask_k = (corner == k)
                    if mask_k.any():
                        gi = inv[mask_k]                    # [Nk]
                        xk = xb_sel[mask_k].to(red_dtype)   # [Nk, C]
                        out_buf[:, k, :].index_add_(0, gi, xk)
                        counts[:, k].index_add_(0, gi, torch.ones(gi.shape[0], device=x.device, dtype=red_dtype))

                # average duplicates; zeros stay zeros for missing corners
                counts_clamped = counts.clamp_min(1.0).unsqueeze(-1)  # [G,4,1]
                out_buf = out_buf / counts_clamped
                out_feat = out_buf.reshape(G, 4*C)                     # [G, 4C]
                out_feat = self.reduction2(self.act(self.reduction1(out_feat.to(x_dtype)))) 

            elif self.mode == 'sum':
                # variance-preserving sum: scale by 1/sqrt(k)
                out_buf = torch.zeros(G, C, device=x.device, dtype=red_dtype)
                counts = torch.zeros(G, 1, device=x.device, dtype=red_dtype)
                gi = inv
                xk = xb_sel.to(red_dtype)
                out_buf.index_add_(0, gi, xk)
                counts.index_add_(0, gi, torch.ones(gi.shape[0], device=x.device, dtype=red_dtype).unsqueeze(-1))
                scale = counts.clamp_min(1.0).sqrt().reciprocal()
                out_feat = out_buf * scale
                out_feat = self.reduction2(self.act(self.reduction1(out_feat.to(x_dtype)))) 
            else:
                raise ValueError(f"Unknown mode {self.mode}")

            if self.ln_out is not None:
                out_feat = self.ln_out(out_feat)

            out_x_list.append(out_feat)        # [G, out_dim]
            out_coords_list.append(coords_out) # [G, 2]

        # pack (pad) to max group length across batch for dense return
        Gmax = max((t.shape[0] for t in out_x_list), default=0)
        out_x = x.new_zeros(B, Gmax, self.out_dim)
        out_coords = coords_rc.new_zeros(B, Gmax, 2)
        out_mask = torch.ones(B, Gmax, dtype=torch.bool, device=x.device)  # True=pad

        for b in range(B):
            G = out_x_list[b].shape[0]
            out_x[b, :G] = out_x_list[b]
            out_coords[b, :G] = out_coords_list[b]
            out_mask[b, :G] = False

        return out_x, out_coords, out_mask

    def forward(
        self,
        x: torch.Tensor,                 # [B, L, C]
        coords_rc: torch.Tensor,         # [B, L, 2] or [L, 2]
        padmask: Optional[torch.Tensor] = None
    ):
        try:
            out = self._forward_impl(x, coords_rc, padmask)
            if self.clear_after_forward and x.is_cuda:
                self._clear_cuda_cache()
            return out
        except RuntimeError as e:
            msg = str(e).lower()
            is_cuda_oom = ("out of memory" in msg) or ("cuda" in msg and "alloc" in msg) or ("cublas" in msg and "alloc" in msg)
            if self.clear_on_oom and is_cuda_oom and x.is_cuda and not self._retrying:
                self._retrying = True
                self._clear_cuda_cache()
                torch.cuda.reset_peak_memory_stats()
                try:
                    return self._forward_impl(x, coords_rc, padmask)
                finally:
                    self._retrying = False
            raise