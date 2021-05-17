#!/bin/bash

python compute_attention.py --mode regular --start 0 --stop 1726 --step 25
wait
python compute_attention.py --mode intersection --start 0 --stop 1726 --step 25
wait
python compute_attention.py --mode identical --start 0 --stop 1726 --step 25
wait
python compute_cross_attention.py --mode regular --start 0 --stop 1726 --step 25
wait
python compute_cross_attention.py --mode intersection --start 0 --stop 1726 --step 25
wait
python compute_cross_attention.py --mode identical --start 0 --stop 1726 --step 25
wait