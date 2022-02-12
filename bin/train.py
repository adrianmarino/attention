#!/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import sys

from device_utils import set_device_name, get_device
from model.encoder import Encoder

sys.path.append('./src')

from logger import initialize_logger
import click

from torchtext.legacy.data import BucketIterator

from data.dataset_loader import DatasetLoader
from field_factory import FieldFactory

sys.path.append('./src')


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
@click.option('--dataset-path', default='../dataset', help='Dataset path')
@click.option('--batch-size', default=128, help='Batch size')
@click.option('--origin-language', default='de', help='Origin language')
@click.option('--target-language', default='en', help='Target language')
@click.option('--origin-min-freq', default=2, help='Origin language word min frequency')
@click.option('--target-min-freq', default=2, help='Target language word min frequency')
@click.option(
    '--device',
    default='cpu',
    help='Device used to train and optimize model. Values: gpu(Default) or cpu(Fallback).'
)
def main(
        dataset_path,
        batch_size,
        origin_language,
        target_language,
        origin_min_freq,
        target_min_freq,
        device
):
    initialize_logger()
    set_device_name(device)

    source_field = FieldFactory.create_from_news_model(origin_language)
    target_field = FieldFactory.create_from_web_model(target_language)

    loader = DatasetLoader(source_field, target_field)

    train_data = loader.load(f'{dataset_path}/train.json')
    valid_data = loader.load(f'{dataset_path}/valid.json')
    test_data = loader.load(f'{dataset_path}/test.json')

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=batch_size,
        sort=False,
        device=get_device()
    )

    source_field.build_vocab(train_data, min_freq=origin_min_freq)
    target_field.build_vocab(train_data, min_freq=target_min_freq)

    vocab_dim = len(source_field.vocab)
    embedding_dim = 256
    dropout = 0.5
    enc_hidden_state_dim = dec_hidden_state_dim = 512

    encoder = Encoder(vocab_dim, embedding_dim, dropout, enc_hidden_state_dim, dec_hidden_state_dim).to(device)

    for batch in train_iterator:
        output = encoder(batch.source)
        print(len(output))
        print(output[0].size())
        print(output[1].size())


if __name__ == '__main__':
    main()
# -----------------------------------------------------------------------------
