# Verification scope

## Included and directly recomputable

| Paper component | Included input | Reproduction boundary |
| --- | --- | --- |
| Paired-answer lower bound, Methodology and human main table | Three models' list-level pair witnesses, cap, error allocation, prefix counts and saved per-prefix outcomes | Recomputes bound and descriptive means; raw respondent pairing and model execution are upstream |
| Known-reference study, primary results and Appendix D | Compact per-prefix columns, family contrasts, cluster IDs, novelty mask and original estimates | Recomputes all primary aggregate means and bootstrap intervals; does not refit selectors or replay checkpoints |
| Full frozen policy accounting | CSV with all 60 policy records and structured evaluation records | Checks coverage; CSV is not a set of trained checkpoints |
| Support transfer, cross-input transfer, native gate and complete expert outputs | Selected saved result records | Inspectable supporting evidence; not all supporting estimates are independently recomputed by this package |
| Candidate semantics | Original CUDA candidate-bank implementation | Executable algorithmic source, without pretrained model weights |

## Upstream inputs

Reproducing acquisition and training additionally requires:

1. Human data acquisition and licensing instructions, stable source versions,
   preprocessing, token/class maps, respondent/list partitions and missingness
   rules; exact Provo text folds and model-specific eligibility rules.
2. Complete model identifiers and checkpoint revisions; token alignment and
   shared vocabulary for the known-reference study; original precision settings.
3. Dataset/shard identity, split membership, prefix deduplication, training and
   validation configurations, frozen candidate bank and every selected policy.
4. Acquisition, fitting, validation and final-scoring entry points with their
   experiment configurations and software versions.
5. Access to the source data and weights under the providers' terms.

These upstream inputs and a complete training pipeline are not included here.

## Data coverage

- Pretrained weights, full hidden-state/expert-output caches and raw corpora.
- Raw respondent records or source answer text.

## Interpretation and validation

The numerical records retain the original values. Redacted path fields are
provenance placeholders, not input locations. Verification uses only the files
in this directory. Supporting control records are available for inspection;
the numerical verifier covers the human bounds and known-reference statistics.
