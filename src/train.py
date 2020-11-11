import argparse
import math

import torch
from torch import nn

from model import make_model

from dataset import FORMULA_TEXT, NAME_TEXT
from dataset import train_iter, valid_iter

from utils import rebatch, LossCompute

import config

# Function for training the model for one epoch
def run_epoch(data_iter, model, loss_compute, train=True):
    total_tokens = 0
    total_loss = 0
    print_tokens = 0

    for i, batch in enumerate(data_iter, 1):

        out, _, pre_output = model.forward(
            batch.src,
            batch.trg,
            batch.src_mask,
            batch.trg_mask,
            batch.src_lengths,
            batch.trg_lengths,
        )
        loss = loss_compute(pre_output, batch.trg_y, batch.nseqs)
        total_loss += loss
        total_tokens += batch.ntokens

    if train:
        print(f"Training Loss: {math.exp(total_loss / total_tokens)}")
    else:
        print(f"Validation Loss: {math.exp(total_loss / total_tokens)}")


# Function to train the model
def train(model, num_epochs=100, lr=0.00001):
    if config.USE_CUDA:
        model.cuda()

    criterion = nn.NLLLoss(reduction="sum", ignore_index=PAD_INDEX)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):

        print("Epoch: ", epoch)
        model.train()
        run_epoch(
            (rebatch(PAD_INDEX, b) for b in train_iter),
            model,
            LossCompute(model.generator, criterion, optim),
        )

        model.eval()
        with torch.no_grad():
            run_epoch(
                (rebatch(PAD_INDEX, b) for b in valid_iter),
                model,
                LossCompute(model.generator, criterion, None),
                train=False
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--learning_rate",
        "-lr",
        type=float,
        help=f"Learning rate for trainig (default: {config.LEARNING_RATE})",
        default=config.LEARNING_RATE,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help=f"Epochs for trainig (default: {config.EPOCHS})",
        default=config.EPOCHS,
    )
    parser.add_argument(
        "--save",
        type=bool,
        help="Save model after training (default: True)",
        default=config.SAVE_MODEL,
    )
    args = parser.parse_args()

    PAD_INDEX = FORMULA_TEXT.vocab.stoi[config.PAD_TOKEN]
    src_vocab = len(FORMULA_TEXT.vocab)
    trg_vocab = len(NAME_TEXT.vocab)

    model = make_model(
        src_vocab,
        trg_vocab,
        emb_size=config.EMB_SIZE,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
    )

    train(model, num_epochs=args.epochs, lr=args.learning_rate)

    if args.save:
        torch.save(model.state_dict(), config.MODEL_PATH)