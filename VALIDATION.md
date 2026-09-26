# Validation

Both result-verification commands passed with the supplied numerical records.

| Check | Outcome |
| --- | --- |
| Three human panels and eight lists per model | Pass |
| 60 unique frozen known-reference policy records | Pass |
| OLMoE bound, 917 prefixes | 0.024771685597978384 nat |
| Qwen Base bound, 917 prefixes | 0.018031885239811155 nat |
| Granite bound, 903 prefixes | 0.0017208707276468354 nat |
| Human per-prefix aggregate means against original saved values | Pass, absolute tolerance 1e-10 |
| Known-reference primary sample | 1,968 novel prefixes |
| Known-reference primary estimates and paired contrasts | All 75 reproduced |
| Original prefix-cluster bootstrap | 5,000 draws, means and intervals match within 1e-10 |

Calculations used CUDA FP64. Granite's acquisition result retains its
`awaiting_independent_audit` status; the subsequent passing audit is supplied
as `evidence/human/granite_independent_audit.json`.

These checks cover aggregation from saved statistics, not raw-data processing,
model execution or selector training. Supporting records are outside the
numerical verifier's coverage.

File checks:

```text
python -B audit_release.py
```

The file check passed for text and tensor-container metadata.
