#! /bin/bash

echo "Downloading raw datasets"
curl -o ./data/raw/pie-corpus.json https://raw.githubusercontent.com/hslh/pie-annotation/master/PIE_annotations_all_no_sentences.json
curl -o ./data/raw/magpie-corpus.jsonl https://raw.githubusercontent.com/hslh/magpie-corpus/master/MAGPIE_unfiltered.jsonl