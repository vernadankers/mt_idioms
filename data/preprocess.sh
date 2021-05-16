#!/bin/bash

python preprocess_json.py --pos_tag_method spacy --output_filename idiom_keywords.tsv --frequency_threshold 0.5 --samples_to_file
wait

#python translate_keywords.py --input_tsv idiom_keywords.tsv --output_tsv idiom_keywords_translated.tsv
#wait

mkdir magpie
mkdir magpie/hidden_states_enc
mkdir magpie/hidden_states_dec
mkdir magpie/attention
mkdir magpie/query_attention
mkdir magpie/cross_attention
mkdir magpie/prds
mkdir magpie/tokenised_prds

for i in {0..1726..25}
do
   python translate_magpie.py --source magpie/inputs/${i}.tsv --pred ${i}_pred.txt --folder magpie
   wait
done