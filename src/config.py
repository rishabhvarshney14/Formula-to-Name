import torch

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if USE_CUDA else "cpu")

PATH = "data/"
TRAIN_PATH = "data/train.csv"
VALID_PATH = "data/valid.csv"

URL = "https://en.wikipedia.org/wiki/Glossary_of_chemical_formulae"

PAD_TOKEN = "<pad>"
SOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"

MAX_SEQ_LEN = 30
BATCH_SIZE = 32

EMB_SIZE = 256
HIDDEN_SIZE = 256
NUM_LAYERS = 1
DROPOUT = 0.2

EPOCHS = 10
LEARNING_RATE = 0.0003

SAVE_MODEL = True
MODEL_PATH = 'model/model'