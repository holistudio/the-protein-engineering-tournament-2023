import torch
from torch.utils.data import Dataset

import tiktoken

import pandas as pd

import os

MECMUP_FILTER = False

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')

dataset = 'Alkaline phosphatase PafA'
data_dir  = os.path.join('..', 'in_silico_supervised', 'input', f'{dataset}  (In Silico_ Supervised)')

train_csv = os.path.join(data_dir, 'train.csv')
test_csv  = os.path.join(data_dir, 'test (with values).csv')

train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)

col_names = list(train_df.columns)
input_cols = col_names[1]
target_cols = col_names[2:]

# TODO: filter train_df so that only records with ['methyl phosphodiester (MecMUP)'] < 75

def tokenize_seqs(data, encoding='gpt2'):
    seqs = data.to_numpy()
    tokenizer = tiktoken.get_encoding(encoding)
    enc_seqs = [tokenizer.encode(seq) for seq in seqs]
    return enc_seqs, tokenizer.n_vocab

def seq_tensor(enc_seqs, max_len):
    enc_seqs_tensor = torch.zeros((len(enc_seqs),max_seq_len), dtype=int, device=device)
    for i, enc_seq in enumerate(enc_seqs):
        for j, id in enumerate(enc_seq):
            enc_seqs_tensor[i][j] = id
    return enc_seqs_tensor

X_train, vocab_size = tokenize_seqs(train_df[input_cols])
X_test, _  = tokenize_seqs(test_df[input_cols])

max_seq_len = max(max([len(enc_seq) for enc_seq in X_train]), max([len(enc_seq) for enc_seq in X_test]))

X_train = seq_tensor(X_train, max_seq_len)
X_test = seq_tensor(X_test, max_seq_len)
# print(X_train.shape, X_test.shape)

y_train = torch.tensor(train_df[target_cols].to_numpy(), device=device, dtype=float)
y_test  = torch.tensor(test_df[target_cols].to_numpy(), device=device, dtype=float)
# print(y_train.shape, y_test.shape)

mu_train = y_train.mean(dim=0)
std_train = y_train.std(dim=0)

y_train = (y_train - mu_train) / std_train
y_test = (y_test - mu_train) / std_train # since test mean and std are not known

class SeqTargetDataset(Dataset):
    def __init__(self, X, y):
        self.input_seq = X
        self.targets = y
    
    def __getitem__(self, index):
        x_item = self.input_seq[index]
        y_item = self.targets[index]
        return x_item, y_item
    
    def __len__(self):
        return self.targets.shape[0]
    
train_ds = SeqTargetDataset(X=X_train,y=y_train)
test_ds = SeqTargetDataset(X=X_test,y=y_test)


class ProcessedDatasets(object):
    def __init__(self):
        self.train = train_ds
        self.test = test_ds
        self.mu = mu_train
        self.std = std_train
        self.vocab_size = vocab_size
