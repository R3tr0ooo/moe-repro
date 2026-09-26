from __future__ import annotations
import torch

def cluster_intervals(values, groups, *, repetitions=5000, seed=20260925):
    if not values.is_cuda or values.dtype != torch.float64 or values.ndim != 2 or (not groups.is_cuda) or (groups.dtype != torch.int64) or (groups.shape != (len(values),)) or (len(values) < 2) or (repetitions < 2) or (not bool(torch.isfinite(values).all())):
        raise ValueError('finite CUDA FP64 observations and aligned integer clusters required')
    _, indices = groups.unique(sorted=True, return_inverse=True)
    n = int(indices.max().item()) + 1
    if n < 2:
        raise ValueError('at least two independent clusters required')
    sums = torch.zeros((n, values.shape[1]), device=values.device, dtype=torch.float64)
    sums.index_add_(0, indices, values)
    sizes = torch.bincount(indices, minlength=n).double()
    generator = torch.Generator(device=values.device).manual_seed(seed)
    draws = []
    for start in range(0, repetitions, 64):
        selected = torch.randint(n, (min(64, repetitions - start), n), generator=generator, device=values.device)
        draws.append(sums[selected].sum(1) / sizes[selected].sum(1)[:, None])
    sample_means = torch.cat(draws)
    quantiles = torch.tensor([0.025, 0.975], device=values.device, dtype=torch.float64)
    return {'mean': values.mean(0), 'interval': torch.quantile(sample_means, quantiles, dim=0), 'clusters': n, 'documents': len(values), 'repetitions': repetitions, 'seed': seed}

def per_document_values(expected, observed, scores):
    if not all((value.is_cuda for value in (expected, observed, scores))) or expected.shape != observed.shape or scores.shape != expected.shape or (expected.ndim != 2) or (not all((bool(torch.isfinite(v).all()) for v in (expected, observed, scores)))) or bool((scores[:, 0] != 0).any()) or bool((expected[:, 0] != 0).any()) or bool((observed[:, 0] != 0).any()):
        raise ValueError('aligned native-anchored CUDA scores/gains required')
    actions = scores.argmax(-1)
    return {'expected_gain': expected.gather(1, actions[:, None]).squeeze(1).double(), 'observed_gain': observed.gather(1, actions[:, None]).squeeze(1).double(), 'actions': actions}

def summarize_columns(columns, groups):
    keys = list(columns)
    receipt = cluster_intervals(torch.stack([columns[key] for key in keys], dim=1), groups)
    result = {key: {'mean': receipt['mean'][i].item(), 'interval95': receipt['interval'][:, i].tolist()} for i, key in enumerate(keys)}
    return {'estimates': result, **{key: receipt[key] for key in ('clusters', 'documents', 'repetitions', 'seed')}, 'scope': 'paired prefix-cluster bootstrap conditional on frozen fitted policies'}
