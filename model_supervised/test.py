import datetime
import json

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from load_data import ProcessedDatasets
from gpt import GPTModel

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')

torch.manual_seed(1337)

FILE_PREFIX = 'GPT_emb256_nh4_nl4'

with open(f"{FILE_PREFIX}_cfg.json", 'r') as f:
    GPT_CONFIG = json.load(f)

model = GPTModel(GPT_CONFIG)
model.to(device)
model.load_state_dict(torch.load(f"{FILE_PREFIX}.pth.tar"))

test_df = ProcessedDatasets().test_df
# print(test_df.head())
col_names = list(test_df.columns)
input_cols = col_names[1]
target_cols = col_names[2:]
y_test  = torch.tensor(test_df[target_cols].to_numpy(), device=device, dtype=float) # NOTE: NOT NORMALIZED

test_ds = ProcessedDatasets().test # NOTE: normalized values
X_test = test_ds.input_seq # NOTE: tokenized input sequences
mu = ProcessedDatasets().mu # use for recover values to real-world
std = ProcessedDatasets().std

vocab_size = ProcessedDatasets().vocab_size


def spearman_corr(pred, y):
    # Get ranks of pred and y using double argsort trick
    pred_rank = torch.argsort(torch.argsort(pred))
    y_rank = torch.argsort(torch.argsort(y))

    # Convert to float for correlation computation
    pred_rank = pred_rank.float()
    y_rank = y_rank.float()

    # Normalize (zero mean and unit variance)
    pred_rank = (pred_rank - pred_rank.mean()) / pred_rank.std()
    y_rank = (y_rank - y_rank.mean()) / y_rank.std()

    # Pearson correlation of ranks
    return (pred_rank * y_rank).mean()


# give X_test to the model to predict preds
preds = model(X_test)[:, -1, :]

# recover preds to real-world values
preds = preds * std + mu

# store preds in a dataframe similar to test_df
preds_df = test_df.copy(deep=True)
preds_np = preds.detach().cpu().numpy()
preds_df.iloc[:, -3:] = preds_np
preds_df.to_pickle(f"{FILE_PREFIX}_preds.pkl")
# print(preds_df.head())

print("Target, Spearman coeff")
# for each column in tensors
for c in range(preds.shape[-1]):
    feature_vals = preds[:, c]

    # compute spearman coefficient
    sp = spearman_corr(feature_vals, y_test[:, c])

    print(f"{target_cols[c]}, {sp}")

