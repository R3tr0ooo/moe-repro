from __future__ import annotations
import warnings
import torch
SCIENTIFIC_TOLERANCE = 1e-10
_MASS_TOLERANCE = 1e-06
_FLOAT_DTYPES = (torch.float32, torch.float64)
_ID_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)

def _require_cuda(name, value, ndim, dtypes):
    if not isinstance(value, torch.Tensor) or not value.is_cuda or value.layout != torch.strided or (value.ndim != ndim) or (value.dtype not in dtypes):
        raise ValueError(f'{name} must be a strided {ndim}D CUDA tensor with dtype in {dtypes}')

@torch.no_grad()
def one_swap_bank_cuda(native_ids, native_weights, router_logits, extras=4):
    _require_cuda('native_ids', native_ids, 2, _ID_DTYPES)
    _require_cuda('native_weights', native_weights, 2, _FLOAT_DTYPES)
    _require_cuda('router_logits', router_logits, 2, _FLOAT_DTYPES)
    if native_ids.device != native_weights.device or native_ids.device != router_logits.device:
        raise ValueError('native IDs, weights and logits must share a CUDA device')
    if native_weights.shape != native_ids.shape or router_logits.shape[0] != native_ids.shape[0]:
        raise ValueError('native IDs, weights and logits must align')
    b, k = native_ids.shape
    e = router_logits.shape[1]
    if not isinstance(extras, int) or isinstance(extras, bool) or extras < 1 or (b == 0) or (k == 0) or (k + extras > e):
        raise ValueError('nonempty native support and enough alternative experts required')
    ids = native_ids.to(dtype=torch.int64)
    weights = native_weights.to(dtype=torch.float32)
    if not bool(torch.isfinite(weights).all()) or not bool((weights > 0).all()) or (not bool(torch.isfinite(router_logits).all())) or bool(((ids < 0) | (ids >= e)).any()):
        raise ValueError('invalid expert IDs, weights or router values')
    canonical, order = ids.sort(dim=1)
    if bool((canonical[:, 1:] == canonical[:, :-1]).any()):
        raise ValueError('duplicate native expert IDs')
    native_w = weights.gather(1, order)
    selected = torch.zeros_like(router_logits, dtype=torch.bool).scatter_(1, canonical, True)
    unselected_logits = router_logits.masked_fill(selected, -torch.inf)
    additions = unselected_logits.argsort(dim=1, descending=True, stable=True)[:, :extras]
    if bool((router_logits.gather(1, canonical).amin(dim=1) < router_logits.gather(1, additions[:, :1]).squeeze(1)).any()):
        raise ValueError('native IDs are not a top-k set of the supplied logits')
    probabilities = torch.softmax(router_logits, dim=1, dtype=torch.float32)
    if not bool(torch.isfinite(probabilities).all()) or not bool((probabilities > 0).all()) or bool((probabilities.to(torch.float64).sum(dim=1) - 1).abs().gt(_MASS_TOLERANCE).any()):
        raise ValueError('expected positive normalized full native router probabilities')
    alternatives = canonical[:, None, None, :].expand(b, extras, k, k).clone()
    slots = torch.arange(k, device=native_ids.device)
    alternatives[:, :, slots, slots] = additions[:, :, None]
    alternatives = alternatives.sort(dim=-1).values.reshape(b, extras * k, k)
    candidates = torch.cat((canonical[:, None, :], alternatives), dim=1)
    mass = native_w.to(torch.float64).sum(dim=1)
    gathered = probabilities[:, None, :].expand(b, 1 + k * extras, e).gather(2, candidates)
    gathered = gathered.to(torch.float64)
    applied = gathered * (mass[:, None, None] / gathered.sum(dim=2, keepdim=True))
    applied[:, :, -1] = mass[:, None] - applied[:, :, :-1].sum(dim=2)
    applied = applied.to(torch.float32)
    applied[:, 0] = native_w
    if not bool(torch.isfinite(applied).all()) or not bool((applied > 0).all()):
        raise ValueError('recipient conditioning produced an invalid weight')
    if bool((applied.to(torch.float64).sum(dim=2) - mass[:, None]).abs().gt(_MASS_TOLERANCE).any()):
        raise ValueError('FP32 execution weights do not preserve selected mass')
    return {'candidate_ids': candidates, 'candidate_weights': applied, 'extra_experts': additions, 'native_mass': mass}

@torch.no_grad()
def distribution_values(candidate_log_probs, teacher_log_probs):
    _require_cuda('candidate_log_probs', candidate_log_probs, 2, _FLOAT_DTYPES)
    _require_cuda('teacher_log_probs', teacher_log_probs, 1, _FLOAT_DTYPES)
    if candidate_log_probs.device != teacher_log_probs.device:
        raise ValueError('candidate and teacher log-probabilities must share a CUDA device')
    a, v = candidate_log_probs.shape
    if a == 0 or v == 0 or teacher_log_probs.shape != (v,):
        raise ValueError('nonempty aligned candidate [A,V] and teacher [V] required')
    logp = candidate_log_probs.to(torch.float64)
    logq = teacher_log_probs.to(torch.float64)
    if not bool(torch.isfinite(logp).all()):
        raise ValueError('finite candidate log-probabilities required')
    if bool((torch.isnan(logq) | torch.isposinf(logq)).any()):
        raise ValueError('teacher log-probabilities must not contain NaN or +inf')
    normalization_error = torch.cat((torch.logsumexp(logp, dim=1), torch.logsumexp(logq, dim=0).reshape(1))).abs()
    if bool((normalization_error > SCIENTIFIC_TOLERANCE).any()):
        raise ValueError('candidate and teacher log-probabilities must be normalized within 1e-10')
    q = logq.exp()
    gains = logp - logp[:1]
    expected_gain = (gains * q[None, :]).sum(dim=1)
    blind_gain, best_action = expected_gain.max(dim=0)
    informed_gain = (gains.max(dim=0).values * q).sum()
    information_gap = informed_gain - blind_gain
    bounds = torch.stack((blind_gain, informed_gain, information_gap))
    if not bool(torch.isfinite(expected_gain).all()) or not bool(torch.isfinite(bounds).all()):
        raise ValueError('nonfinite distribution values')
    if bool((bounds < -SCIENTIFIC_TOLERANCE).any()):
        raise ValueError('nonnegative distribution-value bound violated beyond 1e-10')
    native_null = torch.equal(logp[0], logq)
    if native_null and bool(blind_gain > SCIENTIFIC_TOLERANCE):
        raise ValueError('native-teacher null bound violated beyond 1e-10')
    roundoff_detected = (bounds < 0).any() | (blind_gain > 0) & native_null
    if bool(roundoff_detected):
        warnings.warn('CUDA FP64 bound/native-null roundoff within 1e-10; raw values retained', RuntimeWarning, stacklevel=2)
    return {'expected_gain': expected_gain, 'blind_gain': blind_gain, 'informed_gain': informed_gain, 'information_gap': information_gap, 'best_action': best_action, 'normalization_error': normalization_error, 'roundoff_detected': roundoff_detected}
