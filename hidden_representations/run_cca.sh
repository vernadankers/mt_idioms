#!/bin/bash

python over_layers.py --language $1
wait
python cca_influence.py --language $1 --stop 1727 --label 0 --masked_folder ../data/magpie/${1}/mask_idiom \
                        --neighbourhood 10 --save_file "data/${1}/cca_idi2con.pickle" --neighbours_only
wait
python cca_influence.py --language $1 --stop 1727 --label 1 --masked_folder ../data/magpie/${1}/mask_idiom \
                        --neighbourhood 10 --save_file "data/${1}/cca_idi2idi.pickle" --neighbours_only
wait
python cca_influence.py --language $1 --stop 1727 --label 1 --masked_folder ../data/magpie/${1}/mask_context \
                        --neighbourhood 10 --save_file "data/${1}/cca_con2idi.pickle" --neighbours_only
wait
python cca_influence.py --language $1 --stop 1727 --label 0 --masked_folder ../data/magpie/${1}/mask_context \
                        --neighbourhood 10 --save_file "data/${1}/cca_con2con.pickle" --neighbours_only
wait
python appendix.py
wait
