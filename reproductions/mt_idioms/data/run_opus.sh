#!/bin/bash

for i in {0..1726}
do
   python translate_opus.py --source opus/${i}.en --pred opus/${i}_pred.txt 
   wait
done

python label_opus.py
wait