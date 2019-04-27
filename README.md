## 10th place solution for [Gendered Pronoun Resolution on Kaggle](https://www.kaggle.com/c/gendered-pronoun-resolution) ##
### I borrow some code from [pytorch-pretrained-BERT/example](https://github.com/huggingface/pytorch-pretrained-BERT/tree/master/examples) and [[PyTorch] BERT Baseline (Public Score ~ 0.54)](https://www.kaggle.com/ceshine/pytorch-bert-baseline-public-score-0-54). ###

## Instruction ##
### Build docker image using docker/Dockerfile. ###
### Training ###

```
./train.sh offsets 30
./train.sh nli 30
```
It will save 90 models(111GB). If your GPU have only 8GB of RAM, change from

```--train_batch_size 4 --gradient_accumulation_steps 5```

to

```--train_batch_size 1 --gradient_accumulation_steps 20```

in train.sh.
### Prediction ###
```
python offsets_predict.py --test_file "test_stage_2.tsv"
python nli_predict.py --test_file "test_stage_2.tsv"
python make_submission.py
```
