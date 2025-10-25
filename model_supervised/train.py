import torch
from torch.utils.data import DataLoader

from load_data import ProcessedDatasets

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')

train_ds = ProcessedDatasets().train
test_ds = ProcessedDatasets().test
vocab_size = ProcessedDatasets().vocab_size

GPT_CONFIG = {
    "vocab_size": vocab_size, # Vocabulary size
    "context_length": train_ds.input_seq.shape[1], # Context length
    "emb_dim": 768, # Embedding dimension
    "n_heads": 12, # Number of attention heads
    "n_layers": 12, # Number of layers
    "drop_rate": 0.1, # Dropout rate
    "qkv_bias": False # Query-Key-Value bias
}

torch.manual_seed(1337)

train_loader = DataLoader(
    dataset=train_ds,
    batch_size=32,
    shuffle=True,
    drop_last=True,
    num_workers=0
)

test_loader = DataLoader(
    dataset=test_ds,
    batch_size=32,
    shuffle=False,
    drop_last=True,
    num_workers=0
)