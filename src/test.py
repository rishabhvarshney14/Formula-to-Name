import torch

from model import make_model
from dataset import FORMULA_TEXT, NAME_TEXT
from utils import rebatch, predict_output
from utils import formula_to_list, Batch
from utils import tokens_to_name
from torchtext.data import Iterator

import config

PAD_INDEX = FORMULA_TEXT.vocab.stoi[config.PAD_TOKEN]
src_vocab = len(FORMULA_TEXT.vocab)
trg_vocab = len(NAME_TEXT.vocab)

model = model = make_model(src_vocab, trg_vocab, emb_size=config.EMB_SIZE, 
                        hidden_size=config.HIDDEN_SIZE, num_layers=config.NUM_LAYERS,
                        dropout=config.DROPOUT)

model.load_state_dict(torch.load(config.MODEL_PATH))

if __name__ == '__main__':
    inp = 'C2H5OH'
    inp = formula_to_list(inp)
    src = torch.tensor([[FORMULA_TEXT.vocab.stoi[i] for i in inp]]).to(config.DEVICE)
    src_len = torch.tensor([len(inp)]).to(config.DEVICE)
    src_mask = torch.tensor([[[True for _ in range(len(inp))]]]).to(config.DEVICE)

    pred, attention = predict_output(
        model, src, src_mask, src_len, max_len=25,
        sos_index=NAME_TEXT.vocab.stoi[config.SOS_TOKEN],
        eos_index=NAME_TEXT.vocab.stoi[config.EOS_TOKEN])
    
    output = tokens_to_name(pred, NAME_TEXT)
    print(output)