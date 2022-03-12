## Data preprocessing

Run `run_magpie.sh` to...
1. Preprocess MAGPIE.
2. Extract keywords.
3. Translate keywords with DeepL (after inserting API key) and Marian-MT.
4. Translate data with Marian-MT while saving hidden representations and attention patterns.
5. Report the label distributions.

Run `run_opus.sh` to... 
1. Translate OPUS sentences containing idioms.
2. Report the label distributions.
