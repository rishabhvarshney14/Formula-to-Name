import torch

SEED = 42

# set DEVICE to cuda if GPU is available else to CPU
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if USE_CUDA else "cpu")

# Path to dataset
PATH = "data/"
TRAIN_PATH = "data/train.csv"
VALID_PATH = "data/valid.csv"

# URL for the wikipedia page
URL = "https://en.wikipedia.org/wiki/Glossary_of_chemical_formulae"

PAD_TOKEN = "<pad>"
SOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"

MAX_SEQ_LEN = 20
BATCH_SIZE = 32

EMB_SIZE = 256
HIDDEN_SIZE = 256
NUM_LAYERS = 1
DROPOUT = 0.2

EPOCHS = 40
LEARNING_RATE = 0.0001

# Set SAVE_MODEL to False for testing
SAVE_MODEL = True
MODEL_PATH = "model/model"
