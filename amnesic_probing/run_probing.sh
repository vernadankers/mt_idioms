#!/bin/bash

python probing.py --start 0 --stop 1726 --step 1 --setup hidden --target both --split_by_idiom
wait
python probing.py --start 0 --stop 1726 --step 1 --setup attention --target both --split_by_idiom
wait
python probing.py --start 0 --stop 1726 --step 1 --setup hidden --target both --split_by_idiom --average_pie
wait
python probing.py --start 0 --stop 1726 --step 1 --setup hidden --target magpie --split_by_idiom
wait
python probing.py --start 0 --stop 1726 --step 1 --setup attention --target magpie --split_by_idiom
wait
python probing.py --start 0 --stop 1726 --step 1 --setup hidden --target magpie --split_by_idiom --average_pie
wait