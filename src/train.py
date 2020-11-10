import time
import math

import torch
from torch import nn

from model import make_model

from dataset import FORMULA_TEXT, NAME_TEXT
from dataset import train_iter, valid_iter

from utils import rebatch, LossCompute

import config

def run_epoch(data_iter, model, loss_compute, print_every=50):
    """Standard Training and Logging Function"""

    start = time.time()
    total_tokens = 0
    total_loss = 0
    print_tokens = 0

    for i, batch in enumerate(data_iter, 1):
        
        out, _, pre_output = model.forward(batch.src, batch.trg,
                                           batch.src_mask, batch.trg_mask,
                                           batch.src_lengths, batch.trg_lengths)
        loss = loss_compute(pre_output, batch.trg_y, batch.nseqs)
        total_loss += loss
        total_tokens += batch.ntokens
        print_tokens += batch.ntokens
        
        if model.training and i % print_every == 0:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                    (i, loss / batch.nseqs, print_tokens / elapsed))
            start = time.time()
            print_tokens = 0

    return math.exp(total_loss / float(total_tokens))

def train(model, num_epochs=10, lr=0.0003):
    if config.USE_CUDA:
        model.cuda()

    # optionally add label smoothing; see the Annotated Transformer
    criterion = nn.NLLLoss(reduction="sum", ignore_index=PAD_INDEX)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    
    dev_perplexities = []

    for epoch in range(num_epochs):
      
        print("Epoch", epoch)
        model.train()
        train_perplexity = run_epoch((rebatch(PAD_INDEX, b) for b in train_iter), 
                                     model,
                                     LossCompute(model.generator, criterion, optim))
        
        model.eval()
        with torch.no_grad():
            dev_perplexity = run_epoch((rebatch(PAD_INDEX, b) for b in valid_iter), 
                                       model, 
                                       LossCompute(model.generator, criterion, None))
            print("Validation Loss: %f" % dev_perplexity)
            dev_perplexities.append(dev_perplexity)
        
    return dev_perplexities

if __name__ == '__main__':
    PAD_INDEX = FORMULA_TEXT.vocab.stoi[config.PAD_TOKEN]
    src_vocab = len(FORMULA_TEXT.vocab)
    trg_vocab = len(NAME_TEXT.vocab)

    model = make_model(src_vocab, trg_vocab, emb_size=config.EMB_SIZE, 
                        hidden_size=config.HIDDEN_SIZE, num_layers=config.NUM_LAYERS,
                        dropout=config.DROPOUT)
    
    train(model, num_epochs=config.EPOCHS, lr=config.LEARNING_RATE)

    if config.SAVE_MODEL:
        torch.save(model.state_dict(), config.MODEL_PATH)