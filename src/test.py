import argparse

import torch
from torchtext.data import Iterator

from model import make_model
from dataset import FORMULA_TEXT, NAME_TEXT

from utils import rebatch, predict_output
from utils import formula_to_list
from utils import tokens_to_name

import config

torch.manual_seed(config.SEED)
torch.cuda.manual_seed(config.SEED)

# Converts inputs to appropriate forms as needed for predictions
def input_to_batch(inp):
    try:
        inp = formula_to_list(inp)
    except:
        raise ValueError("Input does not contains valid elements")

    src = torch.tensor([[FORMULA_TEXT.vocab.stoi[i] for i in inp]])
    src_len = torch.tensor([len(inp)])
    src_mask = torch.tensor([[[True for _ in range(len(inp))]]])

    if config.USE_CUDA:
        src = src.to(config.DEVICE)
        src_len = src_len.to(config.DEVICE)
        src_mask = src_mask.to(config.DEVICE)

    return src, src_len, src_mask


PAD_INDEX = FORMULA_TEXT.vocab.stoi[config.PAD_TOKEN]
src_vocab = len(FORMULA_TEXT.vocab)
trg_vocab = len(NAME_TEXT.vocab)

# Make Model
model = make_model(
    src_vocab,
    trg_vocab,
    emb_size=config.EMB_SIZE,
    hidden_size=config.HIDDEN_SIZE,
    num_layers=config.NUM_LAYERS,
    dropout=config.DROPOUT,
)

if config.USE_CUDA:
    model.load_state_dict(torch.load(config.MODEL_PATH))
else:
    model.load_state_dict(
        torch.load(config.MODEL_PATH, map_location=torch.device("cpu"))
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("formula", type=str, help="formula to predict its name")
    args = parser.parse_args()

    src, src_len, src_mask = input_to_batch(args.formula)

    # Predict the outputs
    pred, _ = predict_output(
        model,
        src,
        src_mask,
        src_len,
        max_len=25,
        sos_index=NAME_TEXT.vocab.stoi[config.SOS_TOKEN],
        eos_index=NAME_TEXT.vocab.stoi[config.EOS_TOKEN],
    )

    # Convert tokens(predictions) to name
    output = tokens_to_name(pred, NAME_TEXT)
    print(output)