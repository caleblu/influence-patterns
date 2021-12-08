# influence-patterns
Code for the Neurips 2021 paper
"Influence Patterns for Explaining Information Flow
in BERT" https://arxiv.org/pdf/2011.00740.pdf

## Prerequisites

the `official` folder is a forked and modified from an earlier version of official bert implementation https://github.com/tensorflow/models/tree/master/official

We found such modification to be necessary since our method involves fundamentally modifying the computational graph of the model,  tracking gradients in each layer of the model and separating the gradients from attention heads and skip connections. 

Data for agreement tasks including `wiki.vocab` and `marvin_linzen_dataset.tsv` are based on repos from  
 -   Subject-verb agreement tasks https://github.com/yoavg/bert-syntax 
 -   Reflexive Anaphora tasks https://github.com/yongjie-lin/bert-opensesame

Model checkpoints are downloaded from https://github.com/google-research/bert. (L = 6, H = 512 is used in the paper for agreement tasks). 

### Dependencies
 - tensorflow 2.1.0
 - python 3.6
 - scipy, numpy, tqdm
 - networkx
 - plotly
 - transformers (for sst-2 only)

## Influence Patterns

### Replication of Figure 3a and 3b
A end to end demonstration of using the code base to replicate figure 3a and 3b can be found in `influence_pattern_demo.ipynb`. 

### Core Components
The core methods of influence patterns are in `influence_extractor.py`, and modifications of the bert code in `official/`, mostly under `official/nlp/modeling/layers` and `official/nlp/modeling/networks/transformer_encoder.py`. An example of extracting influence pattern metadata for 1000 examples of SVA-Across Object clauses (250 examples each for SS, SP, PS, PP):

```
python influence_extractor.py --alpha 50 --num_example 250 --bert-dir /YOUR_PATH_TO_BERT_MODEL/uncased_L-6_H-512_A-8/   --sentence_type obj_rel_across_anim  --baseline_type 'mask'  --mask_indice 7 --threshold 0 --cuda 1
```







