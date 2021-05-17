#!/bin/bash

python compute_inlp.py --attention_layers 1 2 3 4 5 6 --folds 0 1 2 3 4 --stop 1726
wait
python compute_inlp.py --attention_layers 1 2 3 4 5 6 --baseline --folds 0 1 2 3 4 --stop 1726
wait
