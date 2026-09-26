# Reproducibility

Paper: **Why Better Expert Choices Do Not Always Make Better MoE Routers**.

Code and numerical records for recomputing the paired-answer bounds and
known-reference results. Verification starts from saved statistics; model
execution and selector training are outside this package.

## Usage

From this directory, with Python 3.11 or newer:

```text
python -B verify_results.py --structure-only
python -B verify_results.py --device cuda:0
python -B audit_release.py
```

The first command checks schemas and record counts using the standard library.
The second recomputes statistics with CUDA-enabled PyTorch in FP64. It requires
no pretrained weights or network access and writes no files. The third checks
the release files for local paths and identifiers.

The CUDA check recomputes:

- Three human-response paired-answer bounds, from the original saved per-pair
  witness averages, original caps, list weights and error allocations.
- The human main-table descriptive means from saved per-prefix measurements.
- All primary known-reference family means, paired contrasts and 5,000-draw
  prefix-cluster bootstrap intervals, from saved per-prefix values. The original
  novelty mask excludes the one development-overlapping prefix.

The supplied inputs begin after response pairing and model scoring. See
`SCOPE.md` for the coverage of each check.

## Files

- `evidence/human/`: numerical panels and pair witnesses for OLMoE, Qwen Base and
  Granite. No answer strings, respondent names or raw texts are included.
- `evidence/known_reference/`: all 60 policy records, evaluation summaries,
  protocol lock and compact per-prefix tensors. The CSV contains policy records,
  not learned model parameter files.
- `evidence/recovery/`, `evidence/controls/`: saved supporting result records.
  These include support-budget curves, Provo refits, input-shuffle comparisons,
  whole-text refit uncertainty and identity/weight controls. They are provided
  for inspection, not covered by the CUDA verifier above.
- `evidence/figure_records.json`: numerical records used in the result plots.
- `code/`: CUDA statistical routines and candidate-bank definitions.
- `SCOPE.md`: coverage of the verification routines.
- `environment_observed.json`: software version used for the numerical checks.

Gains follow the definitions in the paper. Human-response bounds and
known-reference gaps refer to different response distributions. Support-answer
selection uses other answers to the current prefix; it is distinct from
input-only selection on a new prefix.
