# xtuner/engine/hooks/router_anneal_hook.py
from __future__ import annotations
from typing import Iterable, Optional, Sequence, Tuple
import bisect
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper

def _unwrap(m):
    return m.module if is_model_wrapper(m) else m

class RouterAnnealHook(Hook):
    """
    Step/linear 调度 router 的温度/aux 权重/Top-M。

    Args:
        milestones: 里程碑（以 iter 计数），严格递增。例如 (6000, 10000, 13000)
        temp_s:     学生温度在各阶段的取值，长度= len(milestones)+1
        temp_t:     老师温度在各阶段的取值
        aux_w:      router_aux_weight 在各阶段的取值
        top_m:      可选，router_top_m 在各阶段的取值（int 列表）
        mode:       'step' | 'linear'；linear 会在相邻里程碑内做线性插值
        log_interval: 打印间隔
    """
    def __init__(self,
                 milestones: Sequence[int],
                 temp_s: Sequence[float],
                 temp_t: Sequence[float],
                 aux_w: Sequence[float],
                 top_m: Optional[Sequence[int]] = None,
                 mode: str = 'step',
                 log_interval: int = 200):
        assert len(temp_s) == len(milestones) + 1
        assert len(temp_t) == len(milestones) + 1
        assert len(aux_w) == len(milestones) + 1
        if top_m is not None:
            assert len(top_m) == len(milestones) + 1
        assert mode in ('step', 'linear')
        self.milestones = list(milestones)
        self.temp_s = list(temp_s)
        self.temp_t = list(temp_t)
        self.aux_w = list(aux_w)
        self.top_m = list(top_m) if top_m is not None else None
        self.mode = mode
        self.log_interval = log_interval

        self._model = None
        self._last_log = -1

    def before_train(self, runner):
        self._model = _unwrap(runner.model)
        schedule_parts = [
            f'{ms}->(ts={ts:.3f}, tt={tt:.3f}, aux={aw:.4f}'
            + (f', top_m={tm}' if self.top_m is not None else '') + ')'
            for ms, ts, tt, aw, tm in zip(
                self.milestones,
                self.temp_s[1:],
                self.temp_t[1:],
                self.aux_w[1:],
                self.top_m[1:] if self.top_m is not None else [None] * len(self.milestones))
        ]
        runner.logger.info(
            '[RouterAnnealHook] init ts=%.3f tt=%.3f aux=%.4f%s '
            '(mode=%s, milestones: %s)',
            self.temp_s[0],
            self.temp_t[0],
            self.aux_w[0],
            f', top_m={self.top_m[0]}' if self.top_m is not None else '',
            self.mode,
            ', '.join(schedule_parts) if schedule_parts else 'none')

    def _piecewise(self, it: int, xs: Sequence[int], ys: Sequence[float]) -> float:
        """返回 step/linear 调度后的值."""
        idx = bisect.bisect_right(xs, it)  # 当前所处的区间右边界索引
        if self.mode == 'step' or idx == 0 or idx >= len(ys):
            return float(ys[min(idx, len(ys) - 1)])
        if idx >= len(xs):
            return float(ys[-1])
        # linear: 在 milestones[idx-1] 到 milestones[idx] 之间插值
        x0, x1 = xs[idx - 1], xs[idx]
        y0, y1 = ys[idx - 1], ys[idx]
        w = (it - x0) / max(1, (x1 - x0))
        return float((1 - w) * y0 + w * y1)

    def after_train_iter(self, runner, batch_idx: int, data_batch=None, outputs=None):
        if self._model is None:
            return
        it = runner.iter + 1

        # 计算新值
        new_ts = self._piecewise(it, self.milestones, self.temp_s)
        new_tt = self._piecewise(it, self.milestones, self.temp_t)
        new_aw = self._piecewise(it, self.milestones, self.aux_w)

        # 应用到模型
        changed = False
        if getattr(self._model, 'router_temperature_s', None) != new_ts:
            self._model.router_temperature_s = new_ts; changed = True
        if getattr(self._model, 'router_temperature_t', None) != new_tt:
            self._model.router_temperature_t = new_tt; changed = True
        if getattr(self._model, 'router_aux_weight', None) != new_aw:
            self._model.router_aux_weight = new_aw; changed = True

        if self.top_m is not None:
            new_tm = int(round(self._piecewise(it, self.milestones, self.top_m)))
            if getattr(self._model, 'router_top_m', None) != new_tm:
                self._model.router_top_m = new_tm; changed = True

        if changed and (it % self.log_interval == 0 or self._last_log < 0):
            runner.logger.info(
                f'[RouterAnnealHook] iter={it} | '
                f'temp_s={self._model.router_temperature_s:.3f} '
                f'temp_t={self._model.router_temperature_t:.3f} '
                f'aux_w={self._model.router_aux_weight:.4f} '
                + (f'top_m={self._model.router_top_m}' if self.top_m is not None else '')
            )
            self._last_log = it
