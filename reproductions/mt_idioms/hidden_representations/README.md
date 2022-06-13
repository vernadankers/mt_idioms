## Analysis of hidden representations

These scripts use the `cca_core.py` from https://github.com/google/svcca.

0. Ensure you have all of the hidden states collected in the data folder, specifically the `mask_regular`, `mask_context`, `mask_idiom` and `hidden_states_enc` files.
1. Run `run_cca.sh <language>`, where language can be `nl`, `de`, `sv`, `da`, `fr`, `it`, `es`, that starts 6 scripts:
  - Influence: context -> idiom (`cca_influence.py`, code name in figures `con2idi`)
  - Influence: idiom -> idiom (`cca_influence.py`, code name in figures `idi2idi`)
  - Influence: context -> context (`cca_influence.py`, code name in figures `con2con`)
  - Influence: idiom -> context (`cca_influence.py`, code name in figures `idi2con`)
  - Changes over layers (`over_layers.py`)
  - Appendix E (`appendix.py`).
2. Visualise the results using the `visualise.ipynb` notebook. Figures will be stored in the `figures/<language>` folders. Figures for Appendix D will automatically be generated, and can be found in `figures/appendix_languages_...`
