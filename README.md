# Attention

Attention model implementation. [Paper](https://arxiv.org/abs/1409.0473).

## Requisites

* [git](https://git-scm.com/downloads)
* [anaconda](https://www.anaconda.com/products/individual) / [minconda](https://docs.conda.io/en/latest/miniconda.html)
* pytorch-common
  * [Github repo](https://github.com/adrianmarino/pytorch-common/tree/master)
  * [Pypi repo](https://pypi.org/project/pytorch-common/)

## Getting starter

**Step 1**: Clone repo.

```bash
$ git clone https://github.com/adrianmarino/attention.git
$ cd attention
```

**Step 2**: Create environment.

```bash
$ cd dfm
$ conda env create -f environment.yml
```

**Step 3**: Enable project environment.

```bash
$ conda activate attention
```

**Step 3**: Download spacy models.

```bash
$ python -m spacy download en_core_web_sm
$ python -m spacy download de_core_news_sm
```

## Training

```bash
$ python bin/train.py
```

**Helper:**

```bash
$ python bin/train.py --help                                                                                                                                                                         ✔  attention   23:21:19  
Usage: train.py [OPTIONS]

Options:
  --dataset-path TEXT        Dataset path
  --batch-size INTEGER       Batch size
  --origin-language TEXT     Origin language
  --target-language TEXT     Target language
  --origin-min-freq INTEGER  Origin language word min frequency
  --target-min-freq INTEGER  Target language word min frequency
  --device TEXT              Device used to train and optimize model. Values:
                             gpu(Default) or cpu(Fallback).
  --help                     Show this message and exit.
```

## Evaluation

```bash
$ python bin/eval.py \
    --weights=path ./weights/2022-02-16_23-13-14--model--epoch_2--val_loss_3.5578497585497404.pt
```

**Helper:**

```bash
$ python bin/eval.py --help                                                                                                                                                                 1 ✘  3s   attention   23:18:13  
Usage: eval.py [OPTIONS]

Options:
  --dataset-path TEXT        Dataset path
  --batch-size INTEGER       Batch size
  --origin-language TEXT     Origin language
  --target-language TEXT     Target language
  --origin-min-freq INTEGER  Origin language word min frequency
  --target-min-freq INTEGER  Target language word min frequency
  --weights-path TEXT        weights path
  --device TEXT              Device used to train and optimize model. Values:
                             gpu(Default) or cpu(Fallback).
  --help                     Show this message and exit.
```