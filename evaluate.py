import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import argparse
from torchmetrics import MeanAbsolutePercentageError
from main import df2data, data2loader, get_forum_dict, get_author_dict


parser = argparse.ArgumentParser()
parser.add_argument("--normalization", type=str, default=None)
parser.add_argument("--ckpt", type=str, required=True)
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_df = pd.read_csv("./dataset/intern_homework_train_dataset.csv")
public_test = pd.read_csv("./dataset/intern_homework_public_test_dataset.csv")
forum_dict = get_forum_dict(train_df)
# author_dict = get_author_dict(train_df)
public_test_data, public_test_labels = df2data(public_test, drop_columns=["author_id"], forum_dict=forum_dict, author_dict=None, normalization=args.normalization)
public_test_dataloader = data2loader(public_test_data, public_test_labels, batch_size=16, shuffle=False)

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

# Construct model
model = nn.Sequential(
        nn.Linear(public_test_data.shape[1], 1024),
        nn.ReLU(),
        nn.Linear(1024, 2048),
        nn.ReLU(),
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Linear(128, 1)
    )
sum_p = 0
for p in model.parameters():
    sum_p += p.numel()
print(sum_p / 1000000, "M Parameters")

loss_fn = MeanAbsolutePercentageError().to(device)  # mean square error

model.load_state_dict(best_weights)

model = model.to(device)

model.eval()
eval_loss = 0
for i, data in enumerate(public_test_dataloader):
    inputs, labels = data
    y_pred = model(inputs)
    if args.normalization == "std":
        y_pred = y_pred * public_test.like_count_24h.std() + public_test.like_count_24h.mean()
        labels = labels *public_test.like_count_24h.std() + public_test.like_count_24h.mean()
    elif args.normalization == "minmax":
        y_pred = y_pred * (public_test.like_count_24h.max() - public_test.like_count_24h.min()) + public_test.like_count_24h.min()
        labels = labels * (public_test.like_count_24h.max() - public_test.like_count_24h.min()) + public_test.like_count_24h.min()
    print(y_pred)
    print(labels)
    mse = loss_fn(y_pred, labels)
    mse = float(mse)
    eval_loss += mse
    if i % 100 == 0:
        print(y_pred, labels)
eval_loss /= len(public_test_dataloader)
print("loss: ", eval_loss)


