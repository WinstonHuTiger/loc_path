import math
from typing import Optional, Sequence

from mmengine.hooks import Hook


def _get_model(module):
    # unwrap DataParallel/DistributedDataParallel
    if hasattr(module, 'module'):
        return _get_model(module.module)
    return module



class TopKAnnealHook(Hook):
    """Anneal ``resampler.attn_blocks[i].topk`` during training.

    Supports both epoch-based and iteration-based schedules. Provide the
    ``*_epochs`` parameters for coarse updates (once per epoch) and/or the
    ``*_iters`` parameters for fine-grained updates each iteration. At least
    one total length (``total_epochs`` or ``total_iters``) must be positive.

    Args:
        layers: Indices of ``resampler.attn_blocks`` to update.
        start: Initial Top-K values per layer.
        end: Target Top-K values per layer.
        schedule: ``'cosine'`` or ``'linear'`` interpolation.
        total_epochs: Total epochs covering warmup + hold + anneal phases.
        warmup_epochs: Epochs to keep the start value before annealing.
        hold_epochs: Epochs to hold after warmup but before annealing.
        total_iters: Optional iteration-based horizon (warmup + hold + anneal).
        warmup_iters: Iterations to warm up before annealing.
        hold_iters: Iterations to hold before annealing (after warmup).
        log_prefix: Prefix for log messages showing the applied Top-K values.
    """

    def __init__(self,
                 layers: Sequence[int],
                 start: Sequence[int],
                 end: Sequence[int],
                 schedule: str = 'cosine',
                 total_epochs: Optional[int] = None,
                 warmup_epochs: int = 0,
                 hold_epochs: int = 0,
                 total_iters: Optional[int] = None,
                 warmup_iters: int = 0,
                 hold_iters: int = 0,
                 log_prefix: str = 'TopKAnnealHook',
                 log_interval: Optional[int] = None):
        if not (len(layers) == len(start) == len(end)):
            raise ValueError('layers, start, and end must have the same length.')
        self.layers = list(layers)
        self.start = list(start)
        self.end = list(end)
        self.schedule = schedule
        self.log_prefix = log_prefix
        self.log_interval = None if log_interval in (None, 0) else max(1, int(log_interval))
        self._runner_logger = None
        self._last_log_epoch: Optional[int] = None
        self._last_log_iter: Optional[int] = None

        self.total_epochs = None if total_epochs is None else max(0, int(total_epochs))
        self.warmup_epochs = max(0, int(warmup_epochs))
        self.hold_epochs = max(0, int(hold_epochs))
        self.total_iters = None if total_iters is None else max(0, int(total_iters))
        self.warmup_iters = max(0, int(warmup_iters))
        self.hold_iters = max(0, int(hold_iters))

        self._anneal_epochs = (0 if not self._use_epoch_mode else
                               max(0, self.total_epochs - self.warmup_epochs - self.hold_epochs))
        self._anneal_iters = (0 if not self._use_iter_mode else
                              max(0, self.total_iters - self.warmup_iters - self.hold_iters))

        if not (self._use_epoch_mode or self._use_iter_mode):
            raise ValueError(
                'TopKAnnealHook requires total_epochs > 0 and/or total_iters > 0.')

    def _interp(self, s: int, e: int, t: float) -> float:
        if self.schedule == 'linear':
            return s + (e - s) * t
        # cosine
        return e + 0.5*(s - e)*(1 + math.cos(math.pi * t))

    @property
    def _use_epoch_mode(self) -> bool:
        return self.total_epochs is not None and self.total_epochs > 0

    @property
    def _use_iter_mode(self) -> bool:
        return self.total_iters is not None and self.total_iters > 0

    @staticmethod
    def _fraction(current: int,
                  total: Optional[int],
                  warmup: int,
                  hold: int,
                  anneal_total: int) -> float:
        if total is not None and total > 0:
            max_index = total - 1
            current = min(current, max_index)
        if current < warmup:
            return 0.0
        if current < warmup + hold:
            return 0.0
        if anneal_total <= 0:
            return 0.0
        if anneal_total == 1:
            return 1.0
        progress = current - warmup - hold
        progress = max(0, min(progress, anneal_total - 1))
        return min(1.0, progress / (anneal_total - 1))

    def _set_topk(self, resampler, frac: float, where: str):
        if where.startswith('iter') and self.log_interval is not None:
            current = int(where.split()[-1])
            if self._last_log_iter is not None and current < self._last_log_iter + self.log_interval:
                return
            self._last_log_iter = current
        elif where.startswith('epoch') and self.log_interval is not None:
            current = int(where.split()[-1])
            if self._last_log_epoch is not None and current < self._last_log_epoch + self.log_interval:
                return
            self._last_log_epoch = current
        updated = []
        for idx, layer in enumerate(self.layers):
            s = int(self.start[idx])
            e = int(self.end[idx])
            k = int(round(self._interp(s, e, frac)))
            try:
                resampler.attn_blocks[layer].topk = k
                updated.append((layer, k))
            except Exception:
                pass

        if updated:
            msg_body = ', '.join(f'layer {layer}->topk {val}' for layer, val in updated)
            message = f'[{self.log_prefix}] {where}: {msg_body}'
            if self._runner_logger is not None:
                self._runner_logger.info(message)
            else:
                print(message)

    def before_train_epoch(self, runner):
        if not self._use_epoch_mode:
            return
        model = _get_model(runner.model)
        if not hasattr(model, 'mae'):
            return
        resampler = getattr(model.mae, 'resampler', None)
        if resampler is None:
            return

        self._runner_logger = getattr(runner, 'logger', None)
        epoch = getattr(runner, 'epoch', 0)
        frac = self._fraction(epoch,
                              self.total_epochs,
                              self.warmup_epochs,
                              self.hold_epochs,
                              self._anneal_epochs)
        self._set_topk(resampler, frac, f'epoch {epoch}')

    def before_train_iter(self, runner, batch_idx: int, data_batch=None):
        if not self._use_iter_mode:
            return
        model = _get_model(runner.model)
        if not hasattr(model, 'mae'):
            return
        resampler = getattr(model.mae, 'resampler', None)
        if resampler is None:
            return

        self._runner_logger = getattr(runner, 'logger', None)
        current_iter = getattr(runner, 'iter', batch_idx)
        frac = self._fraction(current_iter,
                              self.total_iters,
                              self.warmup_iters,
                              self.hold_iters,
                              self._anneal_iters)
        self._set_topk(resampler, frac, f'iter {current_iter}')
