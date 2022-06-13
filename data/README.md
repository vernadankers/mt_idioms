## Data preprocessing

`magpie/data` contains data extracted from MAGPIE, with the context removed, and idiomaticity annotations in `tsv` files.
The identifiers correspond to the identifiers in `idiom_keywords.tsv`.
  
For example, consider the following line from `18.tsv`:
```
sentence  annotation  idiom  usage  variant  pos_tags
All hell will break loose in England, I though. 1 1 0 1 1 0 0 0 0	all hell broke loose	figurative	combined-inflection	DET NOUN VERB VERB ADJ ADP PROPN PRON ADV
```

Run `run_magpie.sh <language>` to...
1. Translate data with Marian-MT while saving hidden representations and attention patterns.
2. Translate data with Marian-MT while masking a token in the input.
3. Afterwards, `label_magpie.py <language>` computes the distribution over heuristic labels listed in <b>Section 3</b> of the paper.

Run `run_opus.sh` to...
1. Translate OPUS sentences containing idioms for En-Nl.
2. Report the label distributions listed in <b>Appendix G</b> of the paper.
