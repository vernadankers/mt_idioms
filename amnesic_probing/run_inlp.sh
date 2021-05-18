#!/bin/bash

# INLP regular
python compute_inlp.py --attention_layers 1 2 3 4 5 6 --folds 0 1 2 3 4 --stop 1726
wait
python compute_inlp.py --attention_layers 1 2 3 4 5 6 --baseline --folds 0 1 2 3 4 --stop 1726
wait

# INLP baseline
python compute_inlp.py --attention_layers 1 2 3 4 5 6 --folds 0 1 2 3 4 --stop 1726 --baseline
wait
python compute_inlp.py --attention_layers 1 2 3 4 5 6 --baseline --folds 0 1 2 3 4 --stop 1726 --baseline
wait

# Regular interventions
python intervene.py --folds 0 1 2 3 --hidden_layers 1 2 3 4 --stop 1726
wait
python intervene.py --folds 0 1 2 3 --attention_layers 1 2 3 4 5 6 --stop 1726 --attention_labels 0 1
wait
python intervene.py --folds 0 1 2 3 --hidden_layers 1 2 3 4 --attention_layers 1 2 3 4 5 6 --stop 1726 --attention_labels 0 1
wait

# Interventions baseline
python intervene.py --folds 0 1 2 3 --hidden_layers 1 2 3 4 --stop 1726 --baseline
wait
python intervene.py --folds 0 1 2 3 --attention_layers 1 2 3 4 5 6 --stop 1726 --attention_labels 0 1 --baseline
wait
python intervene.py --folds 0 1 2 3 --hidden_layers 1 2 3 4 --attention_layers 1 2 3 4 5 6 --stop 1726 --attention_labels 0 1 --baseline
wait
