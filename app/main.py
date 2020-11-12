import sys

sys.path.insert(0, "src/")

import streamlit as st

import torch

from utils import predict_output, tokens_to_name
from test import input_to_batch, model
from dataset import FORMULA_TEXT, NAME_TEXT

from elements_list import elements_name

import config as config

torch.manual_seed(config.SEED)
torch.cuda.manual_seed(config.SEED)

st.set_page_config(page_title="Formula to Name")

PAD_INDEX = FORMULA_TEXT.vocab.stoi[config.PAD_TOKEN]
src_vocab = len(FORMULA_TEXT.vocab)
trg_vocab = len(NAME_TEXT.vocab)

st.title("Formula to Name Predictor!")
st.write("Predict the chemical name of any chemical formula")

st.sidebar.write("Available elements:")
for element, name in elements_name.items():
    st.sidebar.text(f"{element}: {name}")

formula = st.text_input("Enter Formula (Please check side bar for Elements):")

if formula:
    src, src_len, src_mask = input_to_batch(formula)

    pred, _ = predict_output(
        model,
        src,
        src_mask,
        src_len,
        max_len=20,
        sos_index=NAME_TEXT.vocab.stoi[config.SOS_TOKEN],
        eos_index=NAME_TEXT.vocab.stoi[config.EOS_TOKEN],
    )

    output = tokens_to_name(pred, NAME_TEXT)

    st.write(f"Pedicted Name: {output}")

st.write("Example:")
st.markdown("```input: CH54OHCl``` (Note: This chemical does not exist.)")
st.markdown("```output: 2,4,6-chloroxybenzyl```")

st.markdown(
    "Check source code on [GitHub](https://github.com/rishabhvarshney14/Formula-to-Name)"
)
