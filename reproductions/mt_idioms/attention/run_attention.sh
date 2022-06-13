#!/bin/bash

language=$1

python compute_cross_attention.py --mode regular --start 0 --stop 1727 --language $language --use_precomputed_alignments
wait
python compute_cross_attention.py --mode intersection --start 0 --stop 1727 --language $language --use_precomputed_alignments
wait
python compute_cross_attention.py --mode identical --start 0 --stop 1727 --language $language --use_precomputed_alignments
wait
python compute_cross_attention.py --mode short --start 0 --stop 1727 --language $language --use_precomputed_alignments
wait
python compute_attention.py --mode regular --start 0 --stop 1727 --language $language
wait
python compute_attention.py --mode intersection --start 0 --stop 1727 --language $language
wait
python compute_attention.py --mode identical --start 0 --stop 1727 --language $language
wait
python compute_attention.py --mode short --start 0 --stop 1727 --language $language
wait
