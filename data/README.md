## Data preprocessing

Run `run_magpie.sh <language>` to...
1. Translate data with Marian-MT while saving hidden representations and attention patterns.
2. Translate data with Marian-MT while masking a token in the input.
Afterwards, `label_magpie.py <language>` computes the distribution over heuristic labels.

Run `run_opus.sh` to...
1. Translate OPUS sentences containing idioms for En-Nl.
2. Report the label distributions.
