# Models and data

The original experiments use the following models and datasets. The verification
commands operate on saved numerical records and require no pretrained weights.

| Experiment | Model |
| --- | --- |
| Human OLMoE and controlled routed model | allenai/OLMoE-1B-7B-0924 |
| Controlled response distribution | allenai/OLMoE-1B-7B-0125-Instruct |
| Human Qwen comparison | Qwen/Qwen3-30B-A3B-Base, not the instruction-tuned variant |
| Human third architecture | ibm-granite/granite-4.0-h-tiny |

The controlled acquisition source records checkpoint revisions
`6d84c48581ece794365f2b8e9cfb043c68ade9c5` (routed model) and
`b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e` (reference). Exact Qwen and Granite
checkpoint revisions are not recorded in this package.

Human measurements use the paper's UCL response data and Provo cloze counts.
Provo does not supply respondent IDs for the paired-answer bound. No raw human
responses are included.

The controlled experiment uses one FineWeb-Edu shard, one 128-token prefix per
eligible document, with separate engineering, training, validation and test
documents. The exact shard and document split manifest are not included;
the numerical records alone do not reconstruct those input texts.

Original execution uses OLMoE FP32; Qwen BF16 prefix blocks with an FP32 final
MoE block/readout. The controlled two-checkpoint comparison uses FP32 with TF32
disabled. Statistical calculations use GPU FP64.

Use the original providers' model and data terms. This package does not grant
rights to redistribute third-party weights, source corpora or response records.
