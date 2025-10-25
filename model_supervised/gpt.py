import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from load_data import ProcessedDatasets

train_ds = ProcessedDatasets().train
test_ds = ProcessedDatasets().test
vocab_size = ProcessedDatasets().vocab_size

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

batch = next(iter(train_loader))
print(batch[0].shape, batch[1].shape, vocab_size)