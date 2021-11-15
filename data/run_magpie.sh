#!/bin/bash

language=$1

mkdir -p magpie/${language}
mkdir -p magpie/${language}/hidden_states_enc
mkdir -p magpie/${language}/attention
mkdir -p magpie/${language}/query_attention
mkdir -p magpie/${language}/cross_attention
mkdir -p magpie/${language}/prds
mkdir -p magpie/${language}/tokenised_prds
mkdir -p magpie/${language}/mask_context
mkdir -p magpie/${language}/mask_idiom
mkdir -p magpie/${language}/mask_regular

# To collect all data involves a lot of gigabytes, but this will provide you with a nice small subset of the data
# that can be used to illustrate our findings
for i in {0..1726}
do
	echo "---------- IDIOM ${i} ----------"
	# Files will be stored in magpie/LANGUAGE/prds/... magpie/LANGUAGE/hidden_states_enc/... etc.
	python translate_magpie.py --source magpie/inputs/${i}.tsv --pred ${i}_pred.txt --folder magpie/${language} --language $language
	wait
done

python translate_masking.py --source magpie/inputs/${i}.tsv --pred ${i}_pred.txt --folder magpie/${language} --mode mask_regular --language $language
wait
python translate_masking.py --source magpie/inputs/${i}.tsv --pred ${i}_pred.txt --folder magpie/${language} --mode mask_context --language $language
wait
python translate_masking.py --source magpie/inputs/${i}.tsv --pred ${i}_pred.txt --folder magpie/${language} --mode mask_idiom --language $language
wait