## (Amnesic) Probing

0. Ensure that all data for hidden states is downloaded in the data folder by following instructions there.
1. Run `run_probing.sh`.
2. Visualise F1-scores in the `visualise.ipynb` notebook.
3. Run `amnesic_probing.sh`, that calls `compute_inlp.py`to compute projection matrices and then uses `intervene.py` to intervene in translations.
4. Visualise attention patterns again with the `visualise.ipynb` notebook.

Understandably, the name `debias.py` is confusing, but it contains the INLP technology of https://github.com/yanaiela/amnesic_probing originally presented to debias word embeddings.
