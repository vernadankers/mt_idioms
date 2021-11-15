#!/bin/bash

python probing.py --start 0 --stop 1727 --step 1 --setup hidden --target magpie --split_by_idiom --language $1
wait
python probing.py --start 0 --stop 1727 --step 1 --setup hidden --target both --split_by_idiom --language $1
wait
