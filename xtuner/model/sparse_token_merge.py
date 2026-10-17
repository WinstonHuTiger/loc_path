from typing import Optional, Tuple
import torch
import torch.nn as nn
from mmengine import print_log

class SparsePatchMerging(nn.Module):
    """
    Stable NxN (stride-N) patch merging for Swin/LongNet-style grids with:
      - Pre/Post LayerNorm (eps small) for bf16 stability
      - Deterministic row-major within-cell ordering (TL->TR->...->BR) in concat mode
      - FP32 reductions for scatter/index_add
      - Variance-preserving "sum" option
      - Robust handling of ragged tokens via coords_rc + padmask
      - Optional clear-on-OOM retry
      - Configurable coordinate order via coord_order ('rc' or 'cr')

    forward(x: [B,L,C], coords_rc: [B,L,2] or [L,2], padmask: Optional[bool[B,L]])
         -> (x_merged: [B,L_out,C_out], coords_merged: [B,L_out,2], padmask_merged: [B,L_out])
         where C_out = (merge_size ** 2) * C if mode='concat' else C  (no final projection)
    """
    def __init__(
        self,
        embed_dim: int,
        layernorm_eps: float,
        keep_dim: bool = True,          # kept for API compatibility; unused now
        merge_size: int = 2,
        out_dim: Optional[int] = None,  # kept for API compatibility; ignored
        pre_norm: bool = True,
        post_norm: bool = True,
        fp32_reduce: bool = True,
        mode: str = 'concat',           # 'concat' or 'sum'
        coord_order: str = 'rc',        # 'rc' for (row, col) or 'cr' for (col, row)
        clear_after_forward: bool = False,
        clear_on_oom: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        assert isinstance(merge_size, int) and merge_size >= 2, "merge_size must be an int >= 2."
        self.embed_dim = embed_dim
        self.merge_size = merge_size
        self.m2 = merge_size * merge_size
        self.clear_after_forward = clear_after_forward
        self.clear_on_oom = clear_on_oom
        self._retrying = False

        # --- No final reduction: output dim is purely based on merge mode ---
        in_linear = embed_dim * self.m2 if mode == 'concat' else embed_dim
        self.out_dim = in_linear

        self.pre_norm = pre_norm
        self.post_norm = post_norm
        self.fp32_reduce = fp32_reduce
        self.mode = mode
        assert coord_order in ('rc', 'cr'), "coord_order must be 'rc' or 'cr'"
        self.coord_order = coord_order

        eps = layernorm_eps if layernorm_eps is not None else 1e-6
        self.ln_in = nn.LayerNorm(embed_dim, eps=eps) if pre_norm else None
        self.ln_out = nn.LayerNorm(self.out_dim, eps=eps) if post_norm else None

        self.reset_parameters()

    def reset_parameters(self) -> None:
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
        # print(coords_rc.shape, x.shape)
        assert coords_rc.shape[:2] == (B, L)
        coords_internal = coords_rc if self.coord_order == 'rc' else coords_rc[..., [1, 0]]  # normalize to (r, c)

        if self.ln_in is not None:
            x = self.ln_in(x)

        x_dtype = x.dtype
        red_dtype = torch.float32 if self.fp32_reduce else x.dtype

        out_x_list = []
        out_coords_list = []

        key_stride = int(2**20)  # enough for typical H,W<1e6

        for b in range(B):
            xb = x[b]                              # [L, C]
            coords_b = coords_internal[b].to(torch.long) # [L, 2]
            valid = torch.ones(L, dtype=torch.bool, device=x.device) if padmask is None else (~padmask[b])

            rb = coords_b[:, 0]
            cb = coords_b[:, 1]
            gr = torch.div(rb, self.merge_size, rounding_mode='floor')
            gc = torch.div(cb, self.merge_size, rounding_mode='floor')
            keys = gr * key_stride + gc

            # Row-major index within the merge cell
            rmod = torch.remainder(rb, self.merge_size)
            cmod = torch.remainder(cb, self.merge_size)
            pos_idx = rmod * self.merge_size + cmod

            sel = valid
            if sel.sum() == 0:
                out_x_list.append(xb.new_zeros(0, self.out_dim))
                out_coords_list.append(coords_b.new_zeros(0, 2))
                continue

            keys = keys[sel]
            pos_idx = pos_idx[sel]
            xb_sel = xb[sel]

            uniq, inv = torch.unique(keys, sorted=True, return_inverse=True)
            G = uniq.numel()

            # recover (gr,gc) from linear keys
            gc_out = torch.remainder(uniq, key_stride)
            gr_out = torch.div(uniq, key_stride, rounding_mode='floor')
            coords_out = torch.stack([gr_out, gc_out], dim=-1)  # [G,2]

            if self.mode == 'concat':
                # do FP32 accumulation, then cast back; no projection
                out_buf = torch.zeros(G, self.m2, C, device=x.device, dtype=red_dtype)
                counts = torch.zeros(G, self.m2, device=x.device, dtype=red_dtype)

                for k in range(self.m2):
                    mask_k = (pos_idx == k)
                    if mask_k.any():
                        gi = inv[mask_k]                    # [Nk]
                        xk = xb_sel[mask_k].to(red_dtype)   # [Nk, C]
                        out_buf[:, k, :].index_add_(0, gi, xk)
                        counts[:, k].index_add_(0, gi, torch.ones(gi.shape[0], device=x.device, dtype=red_dtype))

                counts_clamped = counts.clamp_min(1.0).unsqueeze(-1)  # [G,self.m2,1]
                out_buf = out_buf / counts_clamped
                out_feat = out_buf.reshape(G, self.m2 * C).to(x_dtype)        # [G, self.m2*C]

            elif self.mode == 'sum':
                # variance-preserving sum; no projection
                out_buf = torch.zeros(G, C, device=x.device, dtype=red_dtype)
                counts = torch.zeros(G, 1, device=x.device, dtype=red_dtype)
                gi = inv
                xk = xb_sel.to(red_dtype)
                out_buf.index_add_(0, gi, xk)
                counts.index_add_(0, gi, torch.ones(gi.shape[0], device=x.device, dtype=red_dtype).unsqueeze(-1))
                scale = counts.clamp_min(1.0).sqrt().reciprocal()
                out_feat = (out_buf * scale).to(x_dtype)              # [G, C]
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

        if self.coord_order == 'cr':
            out_coords = out_coords[..., [1, 0]]  # restore to caller's (c, r) format

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
    
    def enable_input_require_grads(self):
        print_log("enable input required grads for patch merging", 'current')
        
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        self.model.register_forward_hook(make_inputs_require_grad)
