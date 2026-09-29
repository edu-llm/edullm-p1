"""Selection masks, frozen weights, EMA history, and attention scoring."""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Optional

import numpy as np
import torch
from torch import Tensor, nn

# Two independent per-instance random streams, namespaced so the "random"
# method's own selection scores never correlate with the tie-break stream
# every score-based method uses. Values are arbitrary; only their distinctness
# matters.
_RANDOM_MASK_TAG = 0x5A1E5
_TIEBREAK_TAG = 0x71EBAA


def _per_instance_uniform(
    index: Tensor, *, data_seed: int, tag: int, length: int, device: torch.device
) -> Tensor:
    """Deterministic float32 U[0,1) draws, one row of ``length`` per instance id.

    Depends only on ``(tag, data_seed, instance index)`` -- not on world size,
    rank, microbatch composition, or where the instance lands in a batch. Two
    ArmSpecs with different ``data_seed``s never draw the same stream, and a
    single arm's stream never repeats across instances.
    """
    rows = index.shape[0]
    out = np.empty((rows, length), dtype=np.float32)
    for row, idx in enumerate(index.detach().to("cpu", dtype=torch.int64).tolist()):
        rng = np.random.Generator(
            np.random.PCG64(np.random.SeedSequence([tag, int(data_seed), int(idx)]))
        )
        out[row] = rng.random(length, dtype=np.float32)
    return torch.from_numpy(out).to(device)


def _local(tensor: Tensor) -> Tensor:
    to_local = getattr(tensor, "to_local", None)
    return to_local() if callable(to_local) else tensor


@torch.no_grad()
def _write(parameter: Tensor, value: Tensor) -> None:
    destination = _local(parameter)
    source = _local(value).detach()
    if destination.shape == source.shape:
        destination.copy_(source.to(destination))
        return
    if hasattr(parameter, "device_mesh") and tuple(source.shape) == tuple(parameter.shape):
        from torch.distributed.tensor import distribute_tensor

        sharded = distribute_tensor(
            source.to(device=destination.device, dtype=destination.dtype),
            parameter.device_mesh,
            parameter.placements,
        )
        destination.copy_(_local(sharded))
        return
    raise ValueError(
        f"reference shape {tuple(source.shape)} cannot populate parameter "
        f"global={tuple(parameter.shape)} local={tuple(destination.shape)}"
    )


def _snapshot(parameter: Tensor) -> Tensor:
    full_tensor = getattr(parameter, "full_tensor", None)
    if callable(full_tensor):
        return full_tensor().detach().clone()
    return _local(parameter).detach().clone()


def _reshard(model: nn.Module) -> None:
    from torch.distributed.fsdp import FSDPModule

    for module in reversed(tuple(model.modules())):
        if isinstance(module, FSDPModule):
            module.reshard()


def _tiebroken_order(values: Tensor, tiebreak: Optional[Tensor], *, descending: bool) -> Tensor:
    """Argsort ``values`` along the last dim, breaking exact ties via ``tiebreak``.

    Shuffles by a random permutation (``tiebreak.argsort()``), then takes a
    *stable* sort of the shuffled values: a stable sort preserves the relative
    order of non-tied elements exactly, while tied elements keep the random
    relative order the permutation gave them. With ``tiebreak=None``, falls
    back to a plain (implementation-defined tie order) argsort.
    """
    if tiebreak is None:
        return values.argsort(dim=-1, descending=descending)
    perm = tiebreak.argsort(dim=-1)
    shuffled = values.gather(-1, perm)
    shuffled_order = shuffled.argsort(dim=-1, descending=descending, stable=True)
    return perm.gather(-1, shuffled_order)


def per_row_topk(
    scores: Tensor, fraction: float, valid: Tensor, *, tiebreak: Optional[Tensor] = None
) -> Tensor:
    fraction = min(max(float(fraction), 1e-8), 1.0)
    shape = scores.shape
    values = scores.reshape(-1, shape[-1]).masked_fill(~valid.reshape(-1, shape[-1]), -torch.inf)
    validity = valid.reshape_as(values)
    count = validity.sum(-1)
    keep_count = torch.minimum(torch.clamp((count.float() * fraction).round().long(), min=1), count)
    order = _tiebroken_order(
        values, None if tiebreak is None else tiebreak.reshape_as(values), descending=True
    )
    ranks = torch.empty_like(order)
    ranks.scatter_(
        1,
        order,
        torch.arange(values.shape[1], device=values.device).expand_as(order),
    )
    return ((ranks < keep_count[:, None]) & validity).reshape(shape)


def per_row_middle(
    scores: Tensor, fraction: float, valid: Tensor, *, tiebreak: Optional[Tensor] = None
) -> Tensor:
    fraction = min(max(float(fraction), 1e-8), 1.0)
    shape = scores.shape
    values = scores.reshape(-1, shape[-1]).masked_fill(~valid.reshape(-1, shape[-1]), torch.inf)
    validity = valid.reshape_as(values)
    count = validity.sum(-1)
    keep_count = torch.minimum(torch.clamp((count.float() * fraction).round().long(), min=1), count)
    lower = (count - keep_count) // 2
    order = _tiebroken_order(
        values, None if tiebreak is None else tiebreak.reshape_as(values), descending=False
    )
    ranks = torch.empty_like(order)
    ranks.scatter_(
        1,
        order,
        torch.arange(values.shape[1], device=values.device).expand_as(order),
    )
    return ((ranks >= lower[:, None]) & (ranks < (lower + keep_count)[:, None]) & validity).reshape(
        shape
    )


def per_instance_tiebreak(
    index: Tensor, *, data_seed: int, length: int, device: torch.device
) -> Tensor:
    """The same per-instance tie-break stream ``selection_weights`` uses internally.

    For callers that build masks directly via ``per_row_topk``/``per_row_middle``
    instead of going through ``selection_weights`` (BLADE's per-row selection,
    which has no ``method`` case in ``selection_weights`` -- see its docstring).
    """
    return _per_instance_uniform(index, data_seed=data_seed, tag=_TIEBREAK_TAG, length=length, device=device)


def selection_weights(
    method: str,
    *,
    valid: Tensor,
    keep_fraction: float,
    seed: int,
    index: Tensor,
    current: Optional[Tensor] = None,
    history: Optional[Tensor] = None,
    reference: Optional[Tensor] = None,
    attention: Optional[Tensor] = None,
) -> Tensor:
    """Return float weights; all reported methods reduce to a deterministic 0/1 mask.

    ``index`` is the batch's global instance id (``batch["index"]``, shape
    ``(rows,)``): every per-instance random stream (the ``random`` method's
    own scores, and every method's tie-break) is keyed on it together with
    ``seed`` (the arm's ``data_seed``), so a mask depends only on which
    instances are in the batch, never on world size, rank, or microbatch size.

    ``blade`` is not handled here: BLADE's own token weights are computed by
    ``BladeCallback`` (proxy-vs-outgoing-reference excess loss over the whole
    batch) and passed in via ``batch["token_weight"]``, which
    ``TokenWeightedTrainModule`` reads before ever calling this function.
    """
    if method == "full":
        return valid.float()
    length = valid.shape[-1]
    tiebreak = _per_instance_uniform(
        index, data_seed=seed, tag=_TIEBREAK_TAG, length=length, device=valid.device
    )
    if method == "random":
        scores = _per_instance_uniform(
            index, data_seed=seed, tag=_RANDOM_MASK_TAG, length=length, device=valid.device
        )
        mask = per_row_topk(scores, keep_fraction, valid, tiebreak=tiebreak)
    elif method == "rho_excess":
        if current is None or reference is None:
            raise ValueError("RHO-1 requires current and reference losses")
        mask = per_row_topk(current - reference, keep_fraction, valid, tiebreak=tiebreak)
    elif method == "rel_ema":
        if current is None or history is None:
            raise ValueError("relative EMA requires current and history losses")
        mask = per_row_topk(current - history, keep_fraction, valid, tiebreak=tiebreak)
    elif method == "middle_ppl":
        if reference is None:
            raise ValueError("middle-PPL requires frozen reference losses")
        mask = per_row_middle(reference, keep_fraction, valid, tiebreak=tiebreak)
    elif method == "attention_topk":
        if attention is None:
            raise ValueError("attention selection requires received-attention scores")
        mask = per_row_topk(attention, keep_fraction, valid, tiebreak=tiebreak)
    else:
        raise ValueError(f"unsupported token-selection method {method!r}")
    return mask.float()


class EMAHistory:
    """Bias-corrected local-shard EMA, optionally initialized from RefHQ."""

    VERSION = 2

    def __init__(self, model: nn.Module, *, seed: Optional[Mapping[str, Tensor]] = None):
        self.shadow = {
            name: _local(parameter).detach().clone().zero_()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.correction = 0.0
        if seed is not None:
            parameters = dict(model.named_parameters())
            for name, destination in self.shadow.items():
                if name not in seed:
                    raise KeyError(f"EMA seed missing parameter {name!r}")
                source = seed[name]
                if source.shape != destination.shape:
                    parameter = parameters[name]
                    saved = _snapshot(parameter)
                    _write(parameter, source)
                    source = _local(parameter).detach().clone()
                    _write(parameter, saved)
                destination.copy_(source.to(destination))
            self.correction = 1.0

    @property
    def has_history(self) -> bool:
        return self.correction > 0

    @torch.no_grad()
    def update(self, model: nn.Module, alpha: float) -> None:
        for name, parameter in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(alpha).add_(_local(parameter).detach(), alpha=1.0 - alpha)
        self.correction = alpha * self.correction + 1.0 - alpha

    @contextlib.contextmanager
    def swap_to(self, model: nn.Module) -> Iterator[nn.Module]:
        if not self.has_history:
            yield model
            return
        saved: dict[str, Tensor] = {}
        try:
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if name in self.shadow:
                        saved[name] = _local(parameter).detach().clone()
                        _local(parameter).copy_(self.shadow[name] / self.correction)
            yield model
        finally:
            with torch.no_grad():
                _reshard(model)
                for name, parameter in model.named_parameters():
                    if name in saved:
                        _local(parameter).copy_(saved[name])

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "correction": self.correction,
            "shadow": {name: value.detach().cpu() for name, value in self.shadow.items()},
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("version") != self.VERSION:
            raise ValueError("unsupported EMA state; pre-v2 states are methodologically invalid")
        shadow = state.get("shadow")
        if not isinstance(shadow, Mapping) or set(shadow) != set(self.shadow):
            raise ValueError("EMA state parameter set differs")
        for name, value in shadow.items():
            self.shadow[name].copy_(value.to(self.shadow[name]))
        self.correction = float(state["correction"])


def ema_alpha(step: int, *, tau: Optional[float], constant: Optional[float]) -> float:
    if tau is not None:
        if tau <= 0:
            raise ValueError("EMA tau must be positive")
        return 1.0 - math.exp(-float(step) / tau)
    if constant is None:
        raise ValueError("relative EMA requires tau or constant alpha")
    return float(constant)


def attention_received_from_qk(
    query: Tensor, key: Tensor, *, scale: Optional[float] = None, chunk: int = 256
) -> Tensor:
    """Mean-head causal column mass: ``mean_h sum_{j>=i} A[j,i]``.

    This raw score carries a position effect pulling in two directions: an
    earlier key is reachable by more queries (under uniform attention the score
    is exactly ``H_L - H_i``, see ``uniform_attention_normalizer``), while a
    trained model's recency-biased attention concentrates each query's mass on
    nearby keys, which favors later ones. Neither shape is assumed here;
    ``AttentionPositionBaseline`` measures the combined position effect of the
    model actually being trained and removes it before the score is used for
    token selection.
    """
    batch, length, heads, dim = query.shape
    if key.shape[2] != heads:
        key = key.repeat_interleave(heads // key.shape[2], dim=2)
    result = torch.zeros(batch, heads, length, device=query.device, dtype=torch.float32)
    q, k = query.float(), key.float()
    for start in range(0, length, chunk):
        stop = min(length, start + chunk)
        logits = torch.einsum("bchd,bthd->bhct", q[:, start:stop], k)
        logits.mul_(float(scale if scale is not None else dim**-0.5))
        invalid = (
            torch.arange(length, device=q.device)[None, :]
            > torch.arange(start, stop, device=q.device)[:, None]
        )
        result.add_(logits.masked_fill(invalid[None, None], -torch.inf).softmax(-1).sum(-2))
    return result.mean(1)


_UNIFORM_ATTENTION_NORMALIZER_CACHE: dict[tuple[int, torch.device, torch.dtype], Tensor] = {}


def uniform_attention_normalizer(
    length: int, device: torch.device, dtype: torch.dtype = torch.float32
) -> Tensor:
    """``e_i = sum_{j=i}^{L-1} 1/(j+1) = H_L - H_i`` for 0-indexed position ``i``.

    This is the causal column mass position ``i`` would receive under uniform
    (non-informative) attention: it is reachable by queries ``j = i, ..., L-1``,
    and query ``j``'s causal row has ``j+1`` valid keys, so its uniform share of
    attention on any one of them is ``1/(j+1)``.

    This is only the *cold-start prior* of ``AttentionPositionBaseline`` (used
    before any real attention statistics exist, i.e. the first step of a fresh
    run, whose freshly initialized model attends nearly uniformly). It is not a
    valid position baseline for a trained model: real attention is
    recency-biased, so a late key receives far more mass than this uniform
    reachability count predicts, and dividing by it (which shrinks ~100x over a
    2048-token row) turns the recency bias into a keep-the-tail heuristic.
    """
    key = (int(length), device, dtype)
    cached = _UNIFORM_ATTENTION_NORMALIZER_CACHE.get(key)
    if cached is not None:
        return cached
    inv = 1.0 / torch.arange(1, length + 1, device=device, dtype=dtype)
    normalizer = torch.flip(torch.cumsum(torch.flip(inv, dims=[0]), dim=0), dims=[0])
    _UNIFORM_ATTENTION_NORMALIZER_CACHE[key] = normalizer
    return normalizer


# Floor on a position's log-mass standard deviation, so a degenerate position
# (every observation identical) cannot produce an infinite z-score.
_MIN_LOG_MASS_STD = 1e-6


def align_log_attention_mass(raw: Tensor) -> Tensor:
    """Log of a raw ``(batch, length)`` column-mass score, aligned to the label it gates.

    ``raw[i]`` is the causal column mass at *input* position ``i``. ``get_labels``
    shifts labels left by one, so the loss/label index ``t`` predicts input
    position ``t+1`` -- the score that should gate that loss term is therefore
    the target token's own score, ``s_{t+1}``, not ``s_t``. This takes the log
    (column mass is a positive, multiplicative, heavy-tailed quantity; a ratio
    to a baseline becomes a difference) and applies that one-position shift,
    padding the now-unused last column with ``-inf`` so ``per_row_topk`` never
    selects it (that column has no label anyway). The clamp only guards an
    underflowed-to-zero softmax column before the log.
    """
    tiny = torch.finfo(torch.float32).tiny
    log_mass = raw.float().clamp(min=tiny).log()
    return torch.nn.functional.pad(log_mass[..., 1:], (0, 1), value=-torch.inf)


def uniform_prior_log_mass(length: int, device: torch.device) -> Tensor:
    """``align_log_attention_mass`` of the uniform-attention score, float64.

    The cold-start position mean ``AttentionPositionBaseline`` uses before it
    has observed any real attention. The unused last column is 0.
    """
    normalizer = uniform_attention_normalizer(length, device, torch.float64)
    return torch.nn.functional.pad(normalizer[1:].log(), (0, 1), value=0.0)


class AttentionPositionBaseline:
    """Empirical per-position baseline for the ``attention_topk`` score.

    The score for label position ``t`` of a row is a per-position z-score of
    that row's target-aligned log attention mass ``x_t``
    (``align_log_attention_mass``)::

        score_t = (x_t - mean_t) / std_t

    where ``mean_t``/``std_t`` are the mean and standard deviation of ``x_t``
    at that same position ``t`` across the rows the model has recently
    processed. Positions in a packed fixed-length row carry no content of
    their own (documents are concatenated at arbitrary offsets), so these
    statistics measure exactly the position effect of the model's *current*
    attention -- query-count reachability, recency bias, and anything else --
    without assuming its shape. What is left is per-token: how much more (or
    less) attention a token draws than tokens at its position typically do
    under this checkpoint. Dividing by ``std_t`` (not just centering) also
    equalizes how spread out the score is at each position, so a position
    whose mass is inherently noisier (the last few positions are sums over
    only a handful of queries) is not systematically over- or under-kept by a
    per-row top-k.

    **Where the statistics come from.** Each training step, every rank adds
    the valid tokens of each of its microbatches to ``pending`` (per-position
    count, sum, and sum of squares of ``x``; ``observe``), the train module
    all-reduces ``pending`` across the data-parallel group once per step
    (``all_reduce_pending``), and ``TokenSelectionState.after_optimizer_step``
    replaces the baseline with it (``commit``). A step is therefore scored
    against the *immediately preceding* step's global batch, unblended with
    any older history: no extra forward pass (the raw scores are already
    computed for selection), a baseline identical on every rank and
    independent of world size and microbatch size, and a one-step lag that is
    immaterial (one optimizer step barely moves attention). Before the first
    commit -- the first step of a fresh run, where the freshly initialized
    model attends almost uniformly -- the baseline is the uniform-attention
    prior (``uniform_prior_log_mass``, unit std), under which the score ranks
    tokens exactly as the old ``raw / uniform_attention_normalizer`` score
    did.

    **Why not smooth over several steps.** An earlier version kept an EMA
    over steps (a "how many steps of history" decay constant). A single step
    already contributes every row of the production global batch (2048 rows
    per position) to each position's estimate, so per-step sampling noise is
    already small relative to the real position effect being measured (an
    offline check against real attention statistics: a single step's std is
    already ~0.008 in aligned-log-mass units, against a measured position
    effect spanning roughly 1.7 units end to end); smoothing over more steps
    reduced that already-small noise further but only by cutting how fast the
    baseline could track a real shift in the model's attention -- a real
    tunable cost for a barely-measurable benefit, and one more unmotivated
    constant to defend. Using only the immediately preceding step avoids both:
    no smoothing constant, and the baseline never lags a real change by more
    than the one unavoidable step.

    **Resume.** The moments are the only state; they are checkpointed through
    ``TokenSelectionState.state_dict`` exactly like ``EMAHistory``, so a
    resumed run scores its next step against the same baseline an
    uninterrupted run would have. ``pending`` is per-step scratch, always
    empty at a step boundary, and never saved.
    """

    VERSION = 2

    def __init__(self) -> None:
        # (3, length) float64 [count, sum x, sum x^2] from the immediately
        # preceding step's global batch (replaced wholesale each commit, not
        # blended with anything older).
        self.moments: Optional[Tensor] = None
        # (3, length) float64 raw moments observed during the current step.
        self.pending: Optional[Tensor] = None

    @property
    def has_history(self) -> bool:
        return self.moments is not None

    @torch.no_grad()
    def observe(self, log_mass: Tensor, valid: Tensor) -> None:
        """Add one microbatch's valid, finite aligned log masses to ``pending``."""
        length = log_mass.shape[-1]
        values = log_mass.reshape(-1, length).double()
        use = valid.reshape(-1, length).to(values.device) & torch.isfinite(values)
        values = torch.where(use, values, torch.zeros_like(values))
        stats = torch.stack(
            [use.to(values.dtype).sum(0), values.sum(0), (values * values).sum(0)]
        )
        if self.pending is None:
            self.pending = stats
        elif self.pending.shape != stats.shape:
            raise ValueError(
                f"attention baseline length changed within a step: "
                f"{tuple(self.pending.shape)} vs {tuple(stats.shape)}"
            )
        else:
            self.pending.add_(stats.to(self.pending))

    @torch.no_grad()
    def all_reduce_pending(self, group: Any = None) -> None:
        """Sum ``pending`` over the data-parallel group (no-op when not distributed).

        Every rank must call this exactly once per training step, after its
        last microbatch, so every rank commits the same global statistics.
        """
        if self.pending is None:
            raise RuntimeError("attention baseline has no observations to reduce this step")
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(self.pending, group=group)

    @torch.no_grad()
    def commit(self) -> None:
        """Replace the baseline with this step's ``pending`` moments (no-op if nothing was observed).

        Unblended: the previous step's moments are simply discarded, not
        averaged in. See the class docstring for why.
        """
        if self.pending is None:
            return
        if self.moments is not None and self.moments.shape != self.pending.shape:
            raise ValueError(
                f"attention baseline length changed: {tuple(self.moments.shape)} vs "
                f"{tuple(self.pending.shape)}"
            )
        self.moments = self.pending
        self.pending = None

    @torch.no_grad()
    def position_statistics(self, length: int, device: torch.device) -> tuple[Tensor, Tensor]:
        """Per-position ``(mean, std)`` of the aligned log mass, float64, shape ``(length,)``."""
        if self.moments is None:
            return (
                uniform_prior_log_mass(length, device),
                torch.ones(length, device=device, dtype=torch.float64),
            )
        if self.moments.shape[-1] != length:
            raise ValueError(
                f"attention baseline covers {self.moments.shape[-1]} positions, "
                f"but the batch has {length}"
            )
        count, total, square = self.moments.to(device=device, dtype=torch.float64)
        tiny = torch.finfo(torch.float64).tiny
        observed = count > 0
        safe_count = count.clamp(min=tiny)
        # A position never observed (only ever the label-less last column in
        # practice) falls back to the statistics pooled over all positions.
        pooled_count = count.sum().clamp(min=tiny)
        mean = torch.where(observed, total / safe_count, total.sum() / pooled_count)
        second = torch.where(observed, square / safe_count, square.sum() / pooled_count)
        std = (second - mean * mean).clamp(min=0.0).sqrt().clamp(min=_MIN_LOG_MASS_STD)
        return mean, std

    @torch.no_grad()
    def score(self, log_mass: Tensor) -> Tensor:
        """Per-position z-score of an aligned log mass; ``-inf`` columns stay ``-inf``."""
        mean, std = self.position_statistics(log_mass.shape[-1], log_mass.device)
        return ((log_mass.double() - mean) / std).float()

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "moments": None if self.moments is None else self.moments.detach().cpu(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("version") != self.VERSION:
            raise ValueError("unsupported attention-baseline state")
        moments = state.get("moments")
        if moments is not None:
            if not isinstance(moments, Tensor) or moments.dim() != 2 or moments.shape[0] != 3:
                raise ValueError("attention-baseline moments must be a (3, length) tensor")
            moments = moments.detach().to(dtype=torch.float64).clone()
        self.moments = moments
        self.pending = None


def aligned_log_attention_mass(capture: "AttentionCapture") -> Tensor:
    """``align_log_attention_mass`` applied to a captured forward pass."""
    return align_log_attention_mass(scores_from_capture(capture))


@dataclass
class AttentionCapture:
    x: Optional[Tensor] = None
    module: Optional[nn.Module] = None
    owner: Optional[nn.Module] = None
    kwargs: Optional[dict[str, Any]] = None


@contextlib.contextmanager
def capture_last_attention(model: nn.Module) -> Iterator[AttentionCapture]:
    while hasattr(model, "module") or hasattr(model, "_orig_mod"):
        model = getattr(model, "module", getattr(model, "_orig_mod", model))
    blocks = list(model.blocks.values()) if hasattr(model.blocks, "values") else list(model.blocks)
    block = blocks[-1]
    attention = block.attention
    capture = AttentionCapture(module=attention, owner=block)

    def hook(_module, args, kwargs):
        capture.x = args[0].detach()
        capture.kwargs = dict(kwargs)

    # OLMo compiles each transformer block before HSDP wrapping. Hooks added later
    # to a child attention module are bypassed by the already-compiled block graph,
    # while a hook on the block boundary still executes through nn.Module._call_impl.
    # ReorderedNormTransformerBlock passes its input directly to attention, so the
    # block input is exactly the last-layer attention input for this recipe.
    handle = block.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        yield capture
    finally:
        handle.remove()


@torch.no_grad()
def scores_from_capture(capture: AttentionCapture) -> Tensor:
    if capture.x is None or capture.module is None or capture.owner is None:
        raise RuntimeError("last-layer attention input was not captured")
    attention, owner, x = capture.module, capture.owner, capture.x
    from torch.distributed.fsdp import FSDPModule

    sharded = isinstance(owner, FSDPModule)
    if sharded:
        owner.unshard()
    try:
        q, k = attention.w_q(x), attention.w_k(x)
        if attention.clip_qkv is not None:
            q = q.clamp(min=-attention.clip_qkv, max=attention.clip_qkv)
            k = k.clamp(min=-attention.clip_qkv, max=attention.clip_qkv)
        head_dim = attention.head_dim
        head_norm = bool(getattr(attention, "use_head_qk_norm", False))
        if not head_norm:
            if attention.q_norm is not None:
                q = attention.q_norm(q)
            if attention.k_norm is not None:
                k = attention.k_norm(k)
        q = q.view(x.shape[0], x.shape[1], -1, head_dim)
        k = k.view(x.shape[0], x.shape[1], -1, head_dim)
        if head_norm:
            if attention.q_norm is not None:
                q = attention.q_norm(q)
            if attention.k_norm is not None:
                k = attention.k_norm(k)
        if getattr(attention, "rope", None) is not None:
            kwargs = capture.kwargs or {}
            q, k = attention._apply_rope(
                q,
                k,
                kwargs.get("start_pos"),
                kwargs.get("pos_sin"),
                kwargs.get("pos_cos"),
                kwargs.get("freqs_cis"),
                kwargs.get("cu_doc_lens"),
            )
    finally:
        if sharded:
            owner.reshard()
    return attention_received_from_qk(q, k, scale=getattr(attention, "softmax_scale", None))
