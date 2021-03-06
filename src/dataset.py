import torch
import torchtext
from torchtext.data import Field, BucketIterator
from torchtext.data import TabularDataset, Iterator

import pandas as pd

from utils import formula_to_list
import config

torch.manual_seed(config.SEED)
torch.cuda.manual_seed(config.SEED)

# Tokenizer for formulas
formula_tokenize = formula_to_list

# Tokenizer for names
def name_tokenize(name):
    token = []
    for char in name:
        token.append(char)
    return token


FORMULA_TEXT = Field(
    tokenize=formula_tokenize,
    batch_first=True,
    include_lengths=True,
    pad_token=config.PAD_TOKEN,
    init_token=None,
    eos_token=config.EOS_TOKEN,
)

NAME_TEXT = Field(
    tokenize=name_tokenize,
    batch_first=True,
    include_lengths=True,
    pad_token=config.PAD_TOKEN,
    init_token=config.SOS_TOKEN,
    eos_token=config.EOS_TOKEN,
)

data_fields = [("formula", FORMULA_TEXT), ("name", NAME_TEXT)]

train_data, val_data = TabularDataset.splits(
    config.PATH,
    train="train.csv",
    validation="valid.csv",
    format="csv",
    fields=data_fields,
)

# Form vocabulary from the dataset
FORMULA_TEXT.build_vocab(train_data, val_data)
NAME_TEXT.build_vocab(train_data, val_data)

# Iterator
train_iter = BucketIterator(
    train_data,
    batch_size=config.BATCH_SIZE,
    train=True,
    sort_within_batch=True,
    sort_key=lambda x: (len(x.formula), len(x.name)),
    repeat=False,
    device=config.DEVICE,
)
valid_iter = Iterator(
    val_data, batch_size=1, train=False, sort=False, repeat=False, device=config.DEVICE
)
