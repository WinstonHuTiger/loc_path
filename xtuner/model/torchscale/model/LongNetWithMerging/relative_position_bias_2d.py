
from typing import Optional
import torch
import torch.nn as nn

class RelativePositionBias2D(nn.Module):
    def __init__(
        self,
        n_heads: int,
        num_buckets: int = 32,
        max_distance: int = 128,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.bidirectional = bidirectional

        self.rpb_x = nn.Embedding(num_buckets, n_heads)
        self.rpb_y = nn.Embedding(num_buckets, n_heads)
        nn.init.zeros_(self.rpb_x.weight)
        nn.init.zeros_(self.rpb_y.weight)

    @torch.no_grad()
    def _relative_position_bucket(self, relative_position: torch.Tensor) -> torch.Tensor:
        num_buckets = self.num_buckets
        max_distance = self.max_distance
        n = -relative_position
        if self.bidirectional:
            num_buckets //= 2
            sign = (n < 0).to(torch.long)
            n = n.abs()
        else:
            sign = None
            n = torch.clamp(n, min=0)

        max_exact = num_buckets // 2
        is_small = n < max_exact
        val_if_large = max_exact + (
            (torch.log(n.float() / max_exact + 1e-6) / torch.log(torch.tensor(max_distance / max_exact)))
            * (num_buckets - max_exact)
        ).to(torch.long)
        val_if_large = torch.clamp(val_if_large, max=num_buckets - 1)
        buckets = torch.where(is_small, n, val_if_large)
        if self.bidirectional:
            buckets = buckets + (sign * num_buckets)
        return buckets

    def compute_from_coords(self, coords_q: torch.Tensor, coords_k: torch.Tensor) -> torch.Tensor:
        # Accept [L,2] or [B,L,2]; if 3D, use the first (shared) coords
        if coords_q.dim() == 3:
            coords_q = coords_q[0]
        if coords_k.dim() == 3:
            coords_k = coords_k[0]

        assert coords_q.dim() == 2 and coords_q.size(-1) == 2
        assert coords_k.dim() == 2 and coords_k.size(-1) == 2

        dy = coords_k[:, 0].unsqueeze(0) - coords_q[:, 0].unsqueeze(1)
        dx = coords_k[:, 1].unsqueeze(0) - coords_q[:, 1].unsqueeze(1)

        by = self._relative_position_bucket(dy)
        bx = self._relative_position_bucket(dx)

        vy = self.rpb_y(by)
        vx = self.rpb_x(bx)
        v = (vx + vy).permute(2, 0, 1).contiguous()
        return v

    def forward(self, batch_size: int, coords_q: torch.Tensor, coords_k: torch.Tensor) -> torch.Tensor:
        v = self.compute_from_coords(coords_q, coords_k)
        return v.unsqueeze(0).repeat(batch_size, 1, 1, 1).view(-1, v.size(1), v.size(2))
