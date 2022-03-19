#!/bin/bash

# INLP regular
python compute_inlp.py --folds 0 1 2 3 --stop 1727 --language $1
wait

# INLP for baseline
python compute_inlp.py --folds 0 1 2 3 --stop 1727 --language $1 --baseline
wait

# Regular interventions
python intervene.py --folds 0 1 2 3 --hidden_layers 0 1 2 3 4 --stop 1727 --language $1 \
                    --trace_filename data/trace_${1}.pickle --filename data/attention_${1}.pickle --gather_attention
wait

# Interventions baseline
python intervene.py --folds 0 1 2 3 --hidden_layers 0 1 2 3 4 --stop 1727 --language $1  --baseline \
                    --trace_filename data/trace_baseline_${1}.pickle
wait
