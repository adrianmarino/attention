# Attention Model

Attention based translation(de-to-en) model (seq to seq) implementation. [Paper](https://arxiv.org/abs/1409.0473).

## Requisites

* [git](https://git-scm.com/downloads)
* [anaconda](https://www.anaconda.com/products/individual) / [miniconda](https://docs.conda.io/en/latest/miniconda.html)
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

**Step 4**: Build dataset.

```bash
$ python bin/dataset_builder

2022-02-19 18:47:49,297 INFO train set saved on ./dataset/train.json!
2022-02-19 18:47:49,300 INFO valid set saved on ./dataset/valid.json!
2022-02-19 18:47:49,304 INFO test set saved on ./dataset/test.json!
```

**Help:**

```bash
$ python bin/dataset_builder --help                                                                                                                                                                     ✔  attention   08:21:18  
Usage: dataset_builder [OPTIONS]

Options:
  --destiny-path TEXT           Dataset destiny path (Default: ./dataset)
  --origin-language-model TEXT  Origin language (Default: de_core_news_sm
  --target-language-model TEXT  Target language (Default: en_core_web_sm)
  --help                        Show this message and exit.
```


## Training

```bash
$ python bin/train

2022-02-19 18:52:38,787 INFO Hyper parameters:
2022-02-19 18:52:38,787 INFO  - dataset_path: ./dataset
2022-02-19 18:52:38,787 INFO  - batch_size: 128
2022-02-19 18:52:38,787 INFO  - origin_language_model: de_core_news_sm
2022-02-19 18:52:38,787 INFO  - target_language_model: en_core_web_sm
2022-02-19 18:52:38,787 INFO  - origin_min_freq: 2
2022-02-19 18:52:38,787 INFO  - target_min_freq: 2
2022-02-19 18:52:38,787 INFO  - device: gpu
2022-02-19 18:52:38,787 INFO  - source_embedding_dim: 256
2022-02-19 18:52:38,787 INFO  - target_embedding_dim: 256
2022-02-19 18:52:38,787 INFO  - rnn_hidden_state_dim: 256
2022-02-19 18:52:38,787 INFO  - dropout: 0.5
2022-02-19 18:52:38,787 INFO  - learning_rate: 0.001
2022-02-19 18:52:38,787 INFO  - epochs: 20
2022-02-19 18:52:38,787 INFO  - early_stop_patience: 5
2022-02-19 18:52:38,787 INFO  - reduce_lr_patience: 4
2022-02-19 18:52:38,787 INFO  - reduce_lr_factor: 0.0015
2022-02-19 18:52:38,787 INFO  - reduce_lr_min: 0.0001
2022-02-19 18:52:42,455 INFO Model:
AttentionSeqToSeqModel(
  (encoder): Encoder(
    (_Encoder__embedding): Embedding(7853, 256)
    (_Encoder__dropout): Dropout(p=0.5, inplace=False)
    (_Encoder__rnn): GRU(256, 256, bidirectional=True)
    (_Encoder__linear): Linear(in_features=512, out_features=256, bias=True)
    (_Encoder__tan_h): Tanh()
  )
  (decoder): Decoder(
    (embedding): Embedding(5893, 256)
    (dropout): Dropout(p=0.5, inplace=False)
    (attention): Attention(
      (_Attention__energy_dense): Linear(in_features=768, out_features=256, bias=True)
      (_Attention__energy_activation): Tanh()
      (_Attention__v): Linear(in_features=256, out_features=1, bias=False)
    )
    (rnn): GRU(768, 256)
    (linear): Linear(in_features=1024, out_features=5893, bias=True)
  )
)
Trainable params: 11465221
2022-02-19 18:52:42,456 INFO Begin model training...
2022-02-19 18:53:00,447 INFO {'time': '0:00:17.99', 'epoch': 1, 'train_loss': 5.2856281549394915, 'val_loss': 5.946224212646484, 'patience': 0, 'lr': 0.001}
2022-02-19 18:53:18,132 INFO Save best model: ./weights/2022-02-19_18-53-18--experiment--epoch_2--val_loss_5.595094442367554.pt
2022-02-19 18:53:18,178 INFO {'time': '0:00:17.68', 'epoch': 2, 'train_loss': 4.625914976985444, 'val_loss': 5.595094442367554, 'patience': 0, 'lr': 0.001}
2022-02-19 18:53:35,472 INFO Save best model: ./weights/2022-02-19_18-53-35--experiment--epoch_3--val_loss_4.812914073467255.pt
2022-02-19 18:53:35,506 INFO {'time': '0:00:17.29', 'epoch': 3, 'train_loss': 4.242437697717271, 'val_loss': 4.812914073467255, 'patience': 0, 'lr': 0.001}
2022-02-19 18:53:52,791 INFO Save best model: ./weights/2022-02-19_18-53-52--experiment--epoch_4--val_loss_4.468636572360992.pt
2022-02-19 18:53:52,825 INFO {'time': '0:00:17.29', 'epoch': 4, 'train_loss': 3.8908402331600107, 'val_loss': 4.468636572360992, 'patience': 0, 'lr': 0.001}
2022-02-19 18:54:10,314 INFO Save best model: ./weights/2022-02-19_18-54-10--experiment--epoch_5--val_loss_4.0225904285907745.pt
2022-02-19 18:54:10,346 INFO {'time': '0:00:17.49', 'epoch': 5, 'train_loss': 3.5310082057499153, 'val_loss': 4.0225904285907745, 'patience': 0, 'lr': 0.001}
2022-02-19 18:54:27,329 INFO Save best model: ./weights/2022-02-19_18-54-27--experiment--epoch_6--val_loss_3.8148567974567413.pt
2022-02-19 18:54:27,361 INFO {'time': '0:00:16.98', 'epoch': 6, 'train_loss': 3.1733871514576646, 'val_loss': 3.8148567974567413, 'patience': 0, 'lr': 0.001}
2022-02-19 18:54:44,405 INFO Save best model: ./weights/2022-02-19_18-54-44--experiment--epoch_7--val_loss_3.7206300497055054.pt
2022-02-19 18:54:44,436 INFO {'time': '0:00:17.04', 'epoch': 7, 'train_loss': 2.9215177880509833, 'val_loss': 3.7206300497055054, 'patience': 0, 'lr': 0.001}
2022-02-19 18:55:01,268 INFO Save best model: ./weights/2022-02-19_18-55-01--experiment--epoch_8--val_loss_3.579160064458847.pt
2022-02-19 18:55:01,301 INFO {'time': '0:00:16.83', 'epoch': 8, 'train_loss': 2.67704423723767, 'val_loss': 3.579160064458847, 'patience': 0, 'lr': 0.001}
2022-02-19 18:55:18,399 INFO Save best model: ./weights/2022-02-19_18-55-18--experiment--epoch_9--val_loss_3.4856561720371246.pt
2022-02-19 18:55:18,430 INFO {'time': '0:00:17.10', 'epoch': 9, 'train_loss': 2.535343906427795, 'val_loss': 3.4856561720371246, 'patience': 0, 'lr': 0.001}
2022-02-19 18:55:35,650 INFO Save best model: ./weights/2022-02-19_18-55-35--experiment--epoch_10--val_loss_3.433660387992859.pt
2022-02-19 18:55:35,683 INFO {'time': '0:00:17.22', 'epoch': 10, 'train_loss': 2.395900103489208, 'val_loss': 3.433660387992859, 'patience': 0, 'lr': 0.001}
2022-02-19 18:55:52,715 INFO Save best model: ./weights/2022-02-19_18-55-52--experiment--epoch_11--val_loss_3.4104368090629578.pt
2022-02-19 18:55:52,748 INFO {'time': '0:00:17.03', 'epoch': 11, 'train_loss': 2.225176060777404, 'val_loss': 3.4104368090629578, 'patience': 0, 'lr': 0.001}
2022-02-19 18:56:09,842 INFO Save best model: ./weights/2022-02-19_18-56-09--experiment--epoch_12--val_loss_3.385885864496231.pt
2022-02-19 18:56:09,874 INFO {'time': '0:00:17.09', 'epoch': 12, 'train_loss': 2.116957746413311, 'val_loss': 3.385885864496231, 'patience': 0, 'lr': 0.001}
2022-02-19 18:56:27,038 INFO Save best model: ./weights/2022-02-19_18-56-27--experiment--epoch_13--val_loss_3.369119554758072.pt
2022-02-19 18:56:27,069 INFO {'time': '0:00:17.16', 'epoch': 13, 'train_loss': 2.013856624191553, 'val_loss': 3.369119554758072, 'patience': 0, 'lr': 0.001}
2022-02-19 18:56:44,092 INFO {'time': '0:00:17.02', 'epoch': 14, 'train_loss': 1.9388298972587754, 'val_loss': 3.3745158314704895, 'patience': 0, 'lr': 0.001}
2022-02-19 18:57:01,435 INFO {'time': '0:00:17.34', 'epoch': 15, 'train_loss': 1.8426517368938429, 'val_loss': 3.3831247687339783, 'patience': 1, 'lr': 0.001}
2022-02-19 18:57:18,528 INFO {'time': '0:00:17.09', 'epoch': 16, 'train_loss': 1.751304646945735, 'val_loss': 3.467366576194763, 'patience': 2, 'lr': 0.001}
2022-02-19 18:57:35,464 INFO {'time': '0:00:16.94', 'epoch': 17, 'train_loss': 1.681736251856262, 'val_loss': 3.438371419906616, 'patience': 3, 'lr': 0.001}
2022-02-19 18:57:52,598 INFO {'time': '0:00:17.13', 'epoch': 18, 'train_loss': 1.6105415469224234, 'val_loss': 3.4407562613487244, 'patience': 0, 'lr': 0.0001}
2022-02-19 18:58:09,817 INFO {'time': '0:00:17.22', 'epoch': 19, 'train_loss': 1.4858117802027564, 'val_loss': 3.4612559974193573, 'patience': 1, 'lr': 0.0001}
2022-02-19 18:58:27,308 INFO {'time': '0:00:17.49', 'epoch': 20, 'train_loss': 1.457812064544745, 'val_loss': 3.4632680416107178, 'patience': 2, 'lr': 0.0001}
```

**Help:**

```bash
$ python bin/train --help
Usage: train [OPTIONS]

Options:
  --dataset-path TEXT             Dataset path (Default: ./dataset)
  --batch-size INTEGER            Batch size (Default: 128)
  --origin-language-model TEXT    Origin language (Default: de_core_news_sm
  --target-language-model TEXT    Target language (Default: en_core_web_sm)
  --origin-min-freq INTEGER       Origin language word min frequency (Default:
                                  2)
  --target-min-freq INTEGER       Target language word min frequency (Default:
                                  2)
  --source-embedding-dim INTEGER  Source language embedding dimension
                                  (Default: 256)
  --target-embedding-dim INTEGER  Target language embedding dimension
                                  (Default: 256)
  --rnn-hidden-state-dim INTEGER  Rnn hidden state dimension (Default: 256)
  --dropout FLOAT                 Dropout (Default: 0.5)
  --learning-rate FLOAT           Learning rate (Default: 0.001)
  --epochs INTEGER                Epochs (Default: 20)
  --reduce-lr-patience INTEGER    Reduce learning rate on plateau patience
                                  (Default: 4)
  --reduce-lr-factor FLOAT        Reduce learning rate on plateau factor
                                  (Default: 0.0015)
  --reduce-lr-min FLOAT           Reduce learning rate on plateau factor
                                  (Default: 0.0001)
  --early-stop-patience INTEGER   Early stop patience (Default: 5)
  --device TEXT                   Device used to train and optimize model.
                                  Values: gpu(Default) or cpu(Fallback).
  --help                          Show this message and exit
```

## Evaluation

```bash
$ python bin/eval \
  --weights-path ./weights/2022-02-19_18-56-27--experiment--epoch_13--val_loss_3.369119554758072.pt                                                                            1 ✘  3s   attention   19:03:54  

2022-02-19 19:05:41,228 INFO Test set loss: 3.4043412804603577
```

**Help:**

```bash
$ python bin/eval --help
Usage: eval [OPTIONS]

Options:
  --dataset-path TEXT             Dataset path (Default: ./dataset)
  --batch-size INTEGER            Batch size (Default: 128)
  --origin-language-model TEXT    Origin language (Default: de_core_news_sm
  --target-language-model TEXT    Target language (Default: en_core_web_sm)
  --origin-min-freq INTEGER       Origin language word min frequency (Default:
                                  2)
  --target-min-freq INTEGER       Target language word min frequency (Default:
                                  2)
  --weights-path TEXT             Weights file path
  --source-embedding-dim INTEGER  Source language embedding dimension
                                  (Default: 256)
  --target-embedding-dim INTEGER  Target language embedding dimension
                                  (Default: 256)
  --rnn-hidden-state-dim INTEGER  Rnn hidden state dimension (Default: 256)
  --dropout FLOAT                 Dropout (Default: 0.5)
  --learning-rate FLOAT           Learning rate (Default: 0.001)
  --device TEXT                   Device used to train and optimize model.
                                  Values: gpu(Default) or cpu(Fallback).
  --help                          Show this message and exit
```
