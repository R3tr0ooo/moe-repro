import argparse
import csv
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'code'))

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--structure-only', action='store_true', help='Check schemas and record counts, without scientific arithmetic')
    args = parser.parse_args()
    human = {name: read(ROOT / f'evidence/human/{name}.json') for name in ('olmoe', 'qwen', 'granite')}
    known = read(ROOT / 'evidence/known_reference/result.json')
    with (ROOT / 'evidence/known_reference/known_optimum_policies.csv').open(newline='') as handle:
        policies = list(csv.DictReader(handle))
    assert len(policies) == len(known['policies']) == 60
    assert len({p['id'] for p in known['policies']}) == 60
    for name, result in human.items():
        if name == 'granite':
            audit = read(ROOT / 'evidence/human/granite_independent_audit.json')
            assert audit['status'] == 'pass' and audit['contexts'] == result['contexts']
            assert result['status'] in {'pass', 'awaiting_independent_audit'}
        else:
            assert result['status'] == 'pass'
        assert len(result['panels']) == result['contexts']
        lists = result['bounds']['class']['per_list'] if name == 'granite' else result['per_list']
        assert len(lists) == len({r['list_id'] for r in lists}) == 8
        assert sum((r['contexts'] for r in lists)) == result['contexts']
        assert all((len(r['pair_witnesses']) == r['independent_pair_count_assumed'] for r in lists))
    if args.structure_only:
        print(json.dumps({'status': 'pass', 'check': 'structure', 'models': 3, 'policies': 60}))
        return
    import torch
    from route_scale_final_metrics import summarize_columns
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('CUDA is required. No CPU FP64 fallback is permitted.')
    torch.cuda.set_device(device)
    torch.set_num_threads(1)

    def tensor(value):
        return torch.tensor(value, dtype=torch.float64, device=device)

    def close(actual, expected):
        torch.testing.assert_close(actual, tensor(expected), atol=1e-10, rtol=0)
    human_report = {}
    for name, result in human.items():
        granite = name == 'granite'
        lists = result['bounds']['class']['per_list'] if granite else result['per_list']
        terms = []
        for row in lists:
            witnesses = tensor(row['pair_witnesses'])
            assert bool(torch.isfinite(witnesses).all()) and bool((witnesses >= 0).all())
            cap, alpha = (tensor(row['cap_nats']), tensor(row['alpha']))
            radius = cap * torch.sqrt(torch.log(1 / alpha) / (2 * witnesses.numel()))
            lower = (witnesses.clamp(max=cap).mean() - radius).clamp(min=0)
            close(lower, row['information_gap_lower_bound'])
            terms.append(lower * tensor(row['contexts']) / tensor(result['contexts']))
        lower = torch.stack(terms).sum()
        expected = result['bounds']['class']['information_gap_lower_bound'] if granite else result['primary']['information_gap_lower_bound_nats']
        close(lower, expected)
        if granite:
            field_map = {'class_hindsight': 'class_hindsight', 'class_support_transfer': 'class_support_transfer', 'class_empirical_gap': 'class_empirical_gap', 'class_policy_class_physical': 'class_policy_class_physical'}
            panels = [p['values'] for p in result['panels']]
            means = result['means']
        else:
            field_map = {key: key for key in result['descriptive_context_means']}
            panels, means = (result['panels'], result['descriptive_context_means'])
        for field, reference in field_map.items():
            close(tensor([p[field] for p in panels]).mean(), means[reference])
        human_report[name] = {'prefixes': result['contexts'], 'information_lower_bound': lower.item()}
    payload = torch.load(ROOT / 'evidence/known_reference/document_values.pt', map_location=device, weights_only=True)
    mask, groups = (payload['novel_mask'], payload['prefix_groups'])
    assert mask.dtype == torch.bool and groups.dtype == torch.int64
    assert int(mask.sum().item()) == known['primary']['documents'] == 1968
    columns = {**payload['columns'], **payload['contrasts']}
    recalculated = summarize_columns({k: v[mask] for k, v in columns.items()}, groups[mask])
    assert set(recalculated['estimates']) == set(known['primary']['estimates'])
    for key, values in recalculated['estimates'].items():
        saved = known['primary']['estimates'][key]
        close(tensor(values['mean']), saved['mean'])
        close(tensor(values['interval95']), saved['interval95'])
    print(json.dumps({'status': 'pass', 'check': 'cuda', 'human_bounds': {key: value['information_lower_bound'] for key, value in human_report.items()}, 'reference_estimates': len(columns), 'policies': 60}))
if __name__ == '__main__':
    main()
