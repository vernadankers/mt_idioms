#!/bin/bash

python probing.py --start 0 --stop 1726 --step 1 --setup hidden --target both --split_by_idiom --stop 80
wait
python probing.py --start 0 --stop 1726 --step 1 --setup attention --target both --split_by_idiom --stop 80
wait
python probing.py --start 0 --stop 1726 --step 1 --setup hidden --target both --split_by_idiom --stop 80 --average_pie
wait
python probing.py --start 0 --stop 1726 --step 1 --setup hidden --target magpie --split_by_idiom --stop 80 
wait
python probing.py --start 0 --stop 1726 --step 1 --setup attention --target magpie --split_by_idiom --stop 80
wait
python probing.py --start 0 --stop 1726 --step 1 --setup hidden --target magpie --split_by_idiom --stop 80 --average_pie
wait