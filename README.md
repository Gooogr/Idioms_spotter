# Idioms_spotter
🤗 RoBerta for identification English idiomatic expressions in text

## Dataset
The model was trained using MAGPIE and PIE corpuses.
* [magpie-corpus](https://github.com/hslh/magpie-corpus) [[Download jsonl](https://raw.githubusercontent.com/hslh/magpie-corpus/master/MAGPIE_unfiltered.jsonl)]
* [pie-annotation](https://github.com/hslh/pie-annotation) [[Download json](https://raw.githubusercontent.com/hslh/pie-annotation/master/PIE_annotations_all_no_sentences.json)]

Run these commands from the project root to get them:
```
curl -o ./data/raw/pie-corpus.json https://raw.githubusercontent.com/hslh/pie-annotation/master/PIE_annotations_all_no_sentences.json
curl -o ./data/raw/magpie-corpus.jsonl https://raw.githubusercontent.com/hslh/magpie-corpus/master/MAGPIE_unfiltered.jsonl
```
Note: the PIE corps needs to be further enriched with data in order to obtain suggestions and context. [Details](https://github.com/hslh/pie-annotation#contents--usage)

Raw and already enriched data can be downloaded [here](https://drive.google.com/file/d/1Hvlqp3VU9DeiZeocJNzG4GaxGduOyFAG/view?usp=sharing).



