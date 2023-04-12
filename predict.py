import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import fasttext.util
import argparse
from torchmetrics import MeanAbsolutePercentageError
from main import df2data_private, data2loader, get_forum_dict, get_author_dict

parser = argparse.ArgumentParser()
parser.add_argument("--normalization", type=str, default=None)
parser.add_argument("--ckpt", type=str, required=True)
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_df = pd.read_csv("./dataset/intern_homework_train_dataset.csv")
ft = fasttext.load_model("dataset/cc.zh.300.bin")
print(len(ft.words))
forum_dict = get_forum_dict(train_df)
private_test = pd.read_csv("./dataset/intern_homework_private_test_dataset.csv")
private_test_data = df2data_private(private_test, drop_columns=["author_id"], forum_dict=forum_dict, normalization=args.normalization)
private_test_data = np.expand_dims(private_test_data, axis=0)

model = nn.Sequential(
        nn.Linear(private_test_data.shape[2], 1024),
        nn.ReLU(),
        nn.Linear(1024, 2048),
        nn.ReLU(),
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Linear(128, 1)
    )

name = args.ckpt
a = pickle.load(open(name, "rb"))
min_mse = np.inf
best_weights = None
id = 0
for i in range(5):
    print(a[i]["best_mse"])
    if a[i]["best_mse"] < min_mse:
        min_mse = a[i]["best_mse"]
        best_weights = a[i]["best_weights"]
        id = i
print("best_error: ", min_mse)
model.load_state_dict(best_weights)
model = model.to(device)

model.eval()
preds = model(torch.tensor(private_test_data, dtype=torch.float32 ).to(device))
preds = preds.detach().cpu().numpy()
# reduce preds dimension from (1, 10000, 1) to (10000, )
preds = np.squeeze(preds)
print(preds.shape)
print(preds[:5])
print(private_test_data[:5])

pd = pd.DataFrame(preds)
pd.columns = ["like_count_24h"]
pd.to_csv("submission.csv", index=False)
print(pd.shape)
print(pd.head(5))