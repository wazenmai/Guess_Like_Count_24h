import pickle
import numpy as np
import pandas as pd
import torch.nn as nn
from torchmetrics import MeanAbsolutePercentageError
from main import df2data, data2loader, get_forum_dict, get_author_dict

# train_df = pd.read_csv("./dataset/intern_homework_train_dataset.csv")
public_test = pd.read_csv("./dataset/intern_homework_public_test_dataset.csv")
# forum_dict = get_forum_dict(train_df)
# author_dict = get_author_dict(train_df)
public_test_data, public_test_labels = df2data(public_test, drop_columns=["author_id", "forum_id"])
public_test_dataloader = data2loader(public_test_data, public_test_labels, batch_size=16, shuffle=False)

a = pickle.load(open("model_2-3.pkl", "rb"))
min_mse = np.inf
best_weights = None
for i in range(5):
    print(a[i]["best_mse"])
    if a[i]["best_mse"] < min_mse:
        min_mse = a[i]["best_mse"]
        best_weights = a[i]["best_weights"]
print(min_mse)

# Construct model
model = nn.Sequential(
    nn.Linear(public_test_data.shape[1], 512),
    nn.ReLU(),
    nn.Linear(512, 1024),
    nn.ReLU(),
    nn.Linear(1024, 256),
    nn.ReLU(),
    nn.Linear(256, 1)
)

loss_fn = MeanAbsolutePercentageError()  # mean square error

model.load_state_dict(best_weights)

model.eval()
eval_loss = 0
for i, data in enumerate(public_test_dataloader):
    inputs, labels = data
    y_pred = model(inputs)
    mse = loss_fn(y_pred, labels)
    mse = float(mse)
    eval_loss += mse
    if i % 100 == 0:
        print(y_pred, labels)
eval_loss /= len(public_test_dataloader)
print("loss: ", eval_loss)

