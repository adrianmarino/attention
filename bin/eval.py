#!/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import sys

import torch

sys.path.append('./src')

from pytorch_common.callbacks import SaveBestModel, EarlyStop
from pytorch_common.callbacks.output import Logger
from pytorch_common.util import get_device, set_device_name, trainable_params_count

import logging

from torch import nn
from torch.optim import Adam

from model import Loss, AttentionSeqToSeqModel, ModelManager

from logger import initialize_logger
import click

from torchtext.legacy.data import BucketIterator

from data import DatasetLoader
from field_factory import FieldFactory


# -----------------------------------------------------------------------------
#
#
#
#
#
# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
@click.command()
@click.option('--dataset-path', default='./dataset', help='Dataset path')
@click.option('--batch-size', default=128, help='Batch size')
@click.option('--origin-language', default='de', help='Origin language')
@click.option('--target-language', default='en', help='Target language')
@click.option('--origin-min-freq', default=2, help='Origin language word min frequency')
@click.option('--target-min-freq', default=2, help='Target language word min frequency')
@click.option('--weights-path', help='weights path')
@click.option(
    '--device',
    default='gpu',
    help='Device used to train and optimize model. Values: gpu(Default) or cpu(Fallback).'
)
def main(
        dataset_path,
        batch_size,
        origin_language,
        target_language,
        origin_min_freq,
        target_min_freq,
        weights_path,
        device
):
    initialize_logger()
    set_device_name(device)

    source_field = FieldFactory.create_from_news_model(origin_language)
    target_field = FieldFactory.create_from_web_model(target_language)

    loader = DatasetLoader(source_field, target_field)

    train_data = loader.load(f'{dataset_path}/train.json')
    test_data = loader.load(f'{dataset_path}/test.json')

    test_iterator = BucketIterator.splits(
        test_data,
        batch_size=batch_size,
        sort=False,
        device=get_device()
    )

    source_field.build_vocab(train_data, min_freq=origin_min_freq)
    target_field.build_vocab(train_data, min_freq=target_min_freq)

    embedding_dim = 256
    dropout = 0.5
    hidden_state_dim = 512

    model = AttentionSeqToSeqModel(
        source_vocab_dim=len(source_field.vocab),
        target_vocab_dim=len(target_field.vocab),
        enc_embedding_dim=embedding_dim,
        dec_embedding_dim=embedding_dim,
        enc_dropout=dropout,
        dec_dropout=dropout,
        enc_hidden_state_dim=hidden_state_dim,
        dec_hidden_state_dim=hidden_state_dim
    )
    model.load_state_dict(torch.load(weights_path))
    model.to(get_device())

    model_manager = ModelManager(
        model,
        optimizer=Adam(model.parameters(), lr=0.001),
        loss_fn=Loss(
            loss_fn=nn.CrossEntropyLoss(ignore_index=target_field.vocab.stoi[target_field.pad_token]),
            target_vocab_dim=len(target_field.vocab)
        )
    )

    logging.info(f'Validation loss: {model_manager.validation(test_iterator)}')

if __name__ == '__main__':
    main()
# -----------------------------------------------------------------------------
