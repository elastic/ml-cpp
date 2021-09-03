#!/bin/sh

for c in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
do
echo $c 
# python3 evaluate.py ~/source/ml-search/projects/universal/torchscript/dbmdz-ner/conll03_traced_ner.pt examples/batch_ner/5.json --benchmark --numThreads=$c
python3 evaluate.py ~/source/ml-search/projects/universal/torchscript/sentiment_analysis/distilbert-base-uncased-finetuned-sst-2-english.pt examples/batch_sa/5.json --benchmark --numThreads=$c
done

