import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

import config

torch.manual_seed(config.SEED)
torch.cuda.manual_seed(config.SEED)


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.rnn = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

    def forward(self, x, mask, lengths):
        if config.USE_CUDA:
            lengths = lengths.to("cpu")

        packed = pack_padded_sequence(x, lengths, batch_first=True)
        output, final = self.rnn(packed)
        output, _ = pad_packed_sequence(output, batch_first=True)

        fwd_final = final[0 : final.size(0) : 2]
        bwd_final = final[1 : final.size(0) : 2]
        final = torch.cat([fwd_final, bwd_final], dim=2)

        return output, final


class Decoder(nn.Module):
    def __init__(
        self, emb_size, hidden_size, attention, num_layers=1, dropout=0.5, bridge=True
    ):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention = attention
        self.dropout = dropout

        self.rnn = nn.GRU(
            emb_size + 2 * hidden_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout,
        )

        # to initialize from the final encoder state
        self.bridge = (
            nn.Linear(2 * hidden_size, hidden_size, bias=True) if bridge else None
        )

        self.dropout_layer = nn.Dropout(p=dropout)
        self.pre_output_layer = nn.Linear(
            hidden_size + 2 * hidden_size + emb_size, hidden_size, bias=False
        )

    def forward_step(self, prev_embed, encoder_hidden, src_mask, proj_key, hidden):
        """Perform a single decoder step (1 word)"""

        # compute context vector using attention mechanism
        query = hidden[-1].unsqueeze(1)
        context, attn_probs = self.attention(
            query=query, proj_key=proj_key, value=encoder_hidden, mask=src_mask
        )

        # update rnn hidden state
        rnn_input = torch.cat([prev_embed, context], dim=2)
        output, hidden = self.rnn(rnn_input, hidden)

        pre_output = torch.cat([prev_embed, output, context], dim=2)
        pre_output = self.dropout_layer(pre_output)
        pre_output = self.pre_output_layer(pre_output)

        return output, hidden, pre_output

    def forward(
        self,
        trg_embed,
        encoder_hidden,
        encoder_final,
        src_mask,
        trg_mask,
        hidden=None,
        max_len=None,
    ):
        """Unroll the decoder one step at a time."""

        # the maximum number of steps to unroll the RNN
        if max_len is None:
            max_len = trg_mask.size(-1)

        # initialize decoder hidden state
        if hidden is None:
            hidden = self.init_hidden(encoder_final)

        proj_key = self.attention.key_layer(encoder_hidden)

        decoder_states = []
        pre_output_vectors = []

        # unroll the decoder RNN for max_len steps
        for i in range(max_len):
            prev_embed = trg_embed[:, i].unsqueeze(1)
            output, hidden, pre_output = self.forward_step(
                prev_embed, encoder_hidden, src_mask, proj_key, hidden
            )
            decoder_states.append(output)
            pre_output_vectors.append(pre_output)

        decoder_states = torch.cat(decoder_states, dim=1)
        pre_output_vectors = torch.cat(pre_output_vectors, dim=1)
        return decoder_states, hidden, pre_output_vectors

    def init_hidden(self, encoder_final):
        return torch.tanh(self.bridge(encoder_final))


class Attention(nn.Module):
    def __init__(self, hidden_size, key_size=None, query_size=None):
        super(Attention, self).__init__()

        key_size = 2 * hidden_size if key_size is None else key_size
        query_size = hidden_size if query_size is None else query_size

        self.key_layer = nn.Linear(key_size, hidden_size, bias=False)
        self.query_layer = nn.Linear(query_size, hidden_size, bias=False)
        self.energy_layer = nn.Linear(hidden_size, 1, bias=False)

        self.alphas = None

    def forward(self, mask, query=None, proj_key=None, value=None):
        query = self.query_layer(query)

        scores = self.energy_layer(torch.tanh(query + proj_key))
        scores = scores.squeeze(2).unsqueeze(1)

        scores.data.masked_fill_(mask == 0, -float("inf"))

        alphas = F.softmax(scores, dim=-1)
        self.alphas = alphas

        context = torch.bmm(alphas, value)

        return context, alphas


class Generator(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, trg_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.generator = generator

    def forward(self, src, trg, src_mask, trg_mask, src_lengths, trg_lengths):
        encoder_hidden, encoder_final = self.encode(src, src_mask, src_lengths)
        return self.decode(encoder_hidden, encoder_final, src_mask, trg, trg_mask)

    def encode(self, src, src_mask, src_lengths):
        return self.encoder(self.src_embed(src), src_mask, src_lengths)

    def decode(
        self,
        encoder_hidden,
        encoder_final,
        src_mask,
        trg,
        trg_mask,
        decoder_hidden=None,
    ):
        return self.decoder(
            self.trg_embed(trg),
            encoder_hidden,
            encoder_final,
            src_mask,
            trg_mask,
            hidden=decoder_hidden,
        )


def make_model(
    src_vocab, tgt_vocab, emb_size=256, hidden_size=512, num_layers=1, dropout=0.1
):
    attention = Attention(hidden_size)

    model = EncoderDecoder(
        Encoder(emb_size, hidden_size, num_layers=num_layers, dropout=dropout),
        Decoder(
            emb_size, hidden_size, attention, num_layers=num_layers, dropout=dropout
        ),
        nn.Embedding(src_vocab, emb_size),
        nn.Embedding(tgt_vocab, emb_size),
        Generator(hidden_size, tgt_vocab),
    )

    return model.cuda() if config.USE_CUDA else model