## Analysis of hidden representations

These scripts use the `cca_core.py` from https://github.com/google/svcca.

0. Ensure you have all of the hidden states collected in the data folder, specifically the `mask_regular`, `mask_context`, `mask_idiom` and `hidden_states_enc` files.
1. Run `run_svcca.sh <language>`, where language can be `nl`, `de`, `sv`, `da`, `fr`, `it`, `es`, that starts 6 scripts:
  - Influence: context -> idiom
  - Influence: idiom -> idiom
  - Influence: context -> context
  - Influence: idiom -> idiom
  - Changes over layers
  - Appendix B.
2. Visualise the results using the `visualise.ipynb` notebook. Figures will be stored in the `figures/<language>` folders.
