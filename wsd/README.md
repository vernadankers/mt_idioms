### WSD

0. Download the wmt19 challenge set data from https://github.com/demelin/detecting_wsd_biases_for_nmt and put that in the `data` folder.
1. Run `run_wsd.sh` to preprocess the WSD data and obtain the translations from MarianMT.
2. Run `compute_attention.py` and `compute_cross_attention.py`, which will produce pickled data files stored in the `data` folder.
3. Reproduce the attention analysis with `visualise.ipynb`.
