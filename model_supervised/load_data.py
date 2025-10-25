import torch
import pandas as pd

import os


dataset = 'Alkaline phosphatase PafA'
data_dir  = os.path.join('..', 'in_silico_supervised', 'input', f'{dataset}  (In Silico_ Supervised)')

train_csv = os.path.join(data_dir, 'train.csv')
test_csv  = os.path.join(data_dir, 'test (with values).csv')

train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)

col_names = list(train_df.columns)
input_cols = col_names[:2]
target_cols = col_names[2:]