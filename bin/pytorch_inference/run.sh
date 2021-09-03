#!/bin/sh

for c in 1 2 3 4 5 6 7 8 9 10 20 30 40 50
do
echo $c 
python3 evaluate.py ~/source/ml-search/projects/universal/torchscript/dbmdz-ner/conll03_traced_ner.pt examples/batch_ner/$c.json --benchmark --numThreads=1
done

