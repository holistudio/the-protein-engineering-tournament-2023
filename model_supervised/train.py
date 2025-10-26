import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from load_data import ProcessedDatasets
from gpt import GPTModel

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')

torch.manual_seed(1337)

BATCH_SIZE = 32
EPOCHS = 5000
EVAL_INTERVAL = 100
EVAL_ITERS = 200
LR = 1e-3

GPT_CONFIG = {
    "emb_dim": 768, # Embedding dimension
    "n_heads": 12, # Number of attention heads
    "n_layers": 12, # Number of layers
    "drop_rate": 0.1, # Dropout rate
    "qkv_bias": False # Query-Key-Value bias
}

train_ds = ProcessedDatasets().train
test_ds = ProcessedDatasets().test
vocab_size = ProcessedDatasets().vocab_size

GPT_CONFIG["vocab_size"] = vocab_size # Vocabulary size
GPT_CONFIG["context_length"] = train_ds.input_seq.shape[1] # Context length
GPT_CONFIG["output_size"] = train_ds.targets.shape[1] # Output dimensions

train_loader = DataLoader(
    dataset=train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=0
)

test_loader = DataLoader(
    dataset=test_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=0
)

model = GPTModel(GPT_CONFIG)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

def train_epoch(model, batch, optim):
    X, y = batch

    model.train()

    # forward pass
    preds = model(X)

    # loss
    loss = F.mse_loss(preds,y)

    # clear gradient
    optim.zero_grad()

    # backprop
    loss.backward()

    # optimizer step
    optim.step()

    model.eval()

    return model

@torch.no_grad()
def estimate_loss(model, train_dl, test_dl):
    out = {}
    for split in ['train', 'test']:
        if split == 'train':
            dl = train_dl
        else:
            dl = test_dl
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, y = next(iter(dl))
            preds = model(X)
            loss = F.mse_loss(preds,y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    return out