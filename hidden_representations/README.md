## Analysis of hidden representations

These scripts use the `cca_core.py` from `https://github.com/google/svcca`.
0. Ensure you have all of the hidden states collected in the data folder, specifically the `influence_regular`, `influence_context` and `influence_idiom` fildes.
1. Run `run_svcca.sh`, that starts 6 scripts:
  - Influence: context -> idiom
  - Influence: idiom -> idiom
  - Influence: context -> context
  - Influence: idiom -> idiom
  - Changes over layers
  - Appendix B.
