## Data preprocessing

Run `run_magpie.sh <language>` to...
1. Translate data with Marian-MT while saving hidden representations and attention patterns.
2. Translate data with Marian-MT while masking a token in the input.
3. Afterwards, `label_magpie.py <language>` computes the distribution over heuristic labels listed in <b>Section 3</b> of the paper.

Run `run_opus.sh` to...
1. Translate OPUS sentences containing idioms for En-Nl.
2. Report the label distributions listed in <b>Appendix G</b> of the paper.
