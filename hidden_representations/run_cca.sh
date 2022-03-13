#!/bin/bash

# python over_layers.py --language $1
# wait
# python svcca_influence.py --language $1 --stop 1727 --label 0 --masked_folder ../data/magpie/${1}/mask_idiom --neighbourhood 10 --context_context --save_file "data/${1}/svcca_idi2con.pickle" --neighbours_only
# wait
# python svcca_influence.py --language $1 --stop 1727 --label 1 --masked_folder ../data/magpie/${1}/mask_idiom --neighbourhood 10 --context_context --save_file "data/${1}/svcca_idi2idi.pickle"
# wait
# python svcca_influence.py --language $1 --stop 1727 --label 1 --masked_folder ../data/magpie/${1}/mask_context --neighbourhood 10 --context_context --save_file "data/${1}/svcca_con2idi.pickle"
# wait
# python svcca_influence.py --language $1 --stop 1727 --label 0 --masked_folder ../data/magpie/${1}/mask_context --neighbourhood 10 --context_context --save_file "data/${1}/svcca_con2con.pickle" --neighbours_only
# wait
# done
python appendix.py
wait

