import torch
import torch.nn as nn
from .early_stopping import EarlyStopper

# Constants
EPOCHS = 200
LEARNING_RATE = 0.001
PATIENCE = 5
BATCH_SIZE = 32

# Reusables
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_fn = nn.CrossEntropyLoss()
early_stopper = EarlyStopper(patience=PATIENCE)