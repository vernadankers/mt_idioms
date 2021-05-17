#!/bin/bash

mkdir magpie
mkdir magpie/inputs

## These could be run, but the data is present in the repository.
## Therefore, it's not needed to rerun.
#python preprocess_json.py --pos_tag_method spacy --output_filename idiom_keywords.tsv --frequency_threshold 0.5 --samples_to_file
#wait
#python translate_keywords.py --input_tsv idiom_keywords.tsv --output_tsv idiom_keywords_translated.tsv
#wait

mkdir magpie/hidden_states_enc
mkdir magpie/hidden_states_dec
mkdir magpie/attention
mkdir magpie/query_attention
mkdir magpie/cross_attention
mkdir magpie/prds
mkdir magpie/tokenised_prds

# To collect all data involves a lot of gigabytes, but this will provide you with a nice small subset of the data
# that can be used to illustrate our findings
for i in {0..1726..25}
do
	# Files will be stored in magpie/prds/... magpie/hidden_states_enc/... etc.
	python translate_magpie.py --source magpie/inputs/${i}.tsv --pred ${i}_pred.txt --folder magpie
	wait
done

python label_magpie.py
wait

for i in {0..1726..25}
do
	# Files will be stored in magpie/masked_regular/..., etc.
	python translate_masking.py --source magpie/inputs/${i}.tsv --pred ${i}_pred.txt --folder magpie --mode mask_regular --folder magpie
	wait
	python translate_masking.py --source magpie/inputs/${i}.tsv --pred ${i}_pred.txt --folder magpie --mode mask_context --folder magpie
	wait
	python translate_masking.py --source magpie/inputs/${i}.tsv --pred ${i}_pred.txt --folder magpie --mode mask_idiom --folder magpie
	wait
done