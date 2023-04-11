import pandas as pd
import fasttext.util
import numpy as np
import copy
import pickle
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics import MeanAbsolutePercentageError
from sklearn.model_selection import KFold
from datetime import datetime
from statistics import mean 


result = pd.read_csv("./dataset/example_result.csv")
train_df = pd.read_csv("./dataset/intern_homework_train_dataset.csv")
public_test = pd.read_csv("./dataset/intern_homework_public_test_dataset.csv")
private_test = pd.read_csv("./dataset/intern_homework_private_test_dataset.csv")

ft = fasttext.load_model('cc.zh.300.bin')

MEDIAN = train_df.like_count_24h.median()


def func(value):
    return ''.join(value.splitlines())

def process_time(created_at):
    time = datetime.strptime(created_at, '%Y-%m-%d %H:%M:%S UTC')
    weekday = time.weekday()
    weekday_vec = [1 if i == weekday else 0 for i in range(7)]
    period = 0
    
    if time.hour >= 0 and time.hour < 4:
        period = 0
    elif time.hour >= 4 and time.hour < 8:
        period = 1
    elif time.hour >= 8 and time.hour < 12:
        period = 2
    elif time.hour >= 12 and time.hour < 16:
        period = 3
    elif time.hour >= 16 and time.hour < 20:
        period = 4
    else:
        period = 5
    period_vec = [1 if i == period else 0 for i in range(6)]
    return weekday_vec, period_vec

def get_forum_dict(df):
    forum_dict = {}
    for i in range(df.shape[0]):
        if df.iloc[i].forum_id not in forum_dict:
            forum_dict[df.iloc[i].forum_id] = [df.iloc[i].like_count_24h, 1]
        else:
            forum_dict[df.iloc[i].forum_id][0] += df.iloc[i].like_count_24h
            forum_dict[df.iloc[i].forum_id][1] += 1
    for k in forum_dict:
        forum_dict[k] = forum_dict[k][0] / forum_dict[k][1]
    return forum_dict

def get_author_dict(df):
    author_dict = {}
    for i in range(df.shape[0]):
        if df.iloc[i].author_id not in author_dict:
            author_dict[df.iloc[i].author_id] = [df.iloc[i].like_count_24h, 1]
        else:
            author_dict[df.iloc[i].author_id][0] += df.iloc[i].like_count_24h
            author_dict[df.iloc[i].author_id][1] += 1
    for k in author_dict:
        author_dict[k] = author_dict[k][0] / author_dict[k][1]
    return author_dict

def get_like_mean(info_dict, info_id):
    if info_id in info_dict:
        return info_dict[info_id]
    return MEDIAN

def df2data(org_df, drop_columns=None, forum_dict=None, author_dict=None):
    if drop_columns is not None:
        df = org_df.drop(columns=drop_columns) # created_at
    else:
        df = org_df
    # df = org_df.drop(columns=["forum_id", "author_id", "created_at"])
    data = []
    labels = []
    for index, row in df.iterrows():
        data_array = []
        for column in df.columns:
            if column == "title":
                title = func(row[column].strip())
                title_vec = ft.get_sentence_vector(title).tolist()
                data_array.extend(title_vec)
            elif column == "created_at":
                weekday, period = process_time(row[column])
                data_array.extend(weekday)
                data_array.extend(period)
            elif column == "forum_id" and forum_dict is not None:
                data_array.append(get_like_mean(forum_dict, row[column]))
            elif column == "author_id" and author_dict is not None:
                data_array.append(get_like_mean(author_dict, row[column]))
            else:
                data_array.append(row[column])
        data.append(data_array[:-1])
        labels.append(data_array[-1])
    data = np.asarray(data)
    labels = np.asarray(labels)
    labels = np.expand_dims(labels, axis=1)
    print(data.shape, labels.shape)
    return data, labels

def data2loader(data, label, batch_size, shuffle):
    tensor_x = torch.Tensor(data) # transform to torch tensor
    tensor_y = torch.Tensor(label)

    my_dataset = TensorDataset(tensor_x,tensor_y)
    my_dataloader = DataLoader(my_dataset, batch_size=batch_size, shuffle=shuffle)
    return my_dataloader

def train(n_epochs, train_dataloader, eval_dataloader, data_dim):
    # Construct model
    model = nn.Sequential(
        nn.Linear(data_dim, 512),
        nn.ReLU(),
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Linear(1024, 256),
        nn.ReLU(),
        nn.Linear(256, 1)
    )
    
    # loss function and optimizer
    loss_fn = MeanAbsolutePercentageError()  # mean square error
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Hold the best model
    best_mse = np.inf   # init to infinity
    best_weights = None
    train_losses = []
    eval_losses = []
    
    for epoch in range(n_epochs):
        running_loss = 0.0
        model.train()
        for i, data in enumerate(train_dataloader):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            running_loss += loss.item()

        # evaluate accuracy at end of each epoch
        model.eval()
        eval_loss = 0
        for i, data in enumerate(eval_dataloader):
            inputs, labels = data
            y_pred = model(inputs)
            mse = loss_fn(y_pred, labels)
            mse = float(mse)
            eval_loss += mse
        eval_loss /= len(eval_dataloader)
        eval_losses.append(eval_loss)
        train_loss = running_loss / len(train_dataloader)
        print(f"Epoch {epoch}, training loss: {train_loss}, eval loss: {eval_loss}")
        if eval_loss < best_mse:
            best_mse = eval_loss
            best_weights = copy.deepcopy(model.state_dict())
 
    return {"best_mse": best_mse,
            "best_weights": best_weights,
            "train_losses": train_losses,
            "eval_losses": eval_losses}

def draw(train_losses, eval_losses):
    epoch_average_loss = []
    sample_per_epoch = len(train_losses) // n_epochs
    for i in range(0, len(train_losses), sample_per_epoch):
        epoch_average_loss.append(mean(train_losses[i:i+sample_per_epoch]))
    # print(epoch_average_loss)

    plt.plot(epoch_average_loss, label='train')
    plt.plot(eval_losses, color='orange', label='test')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.show()

def main():
    # forum_dict = get_forum_dict(train_df)
    # author_dict = get_author_dict(train_df)

    # training parameters
    n_epochs = 100   # number of epochs to run
    kfold_result = []

    kf = KFold(n_splits = 5, shuffle = True, random_state = 2)
    for train_index, test_index in kf.split(train_df):
        train_data, train_label = df2data(train_df.iloc[train_index], drop_columns=["author_id", "forum_id"])
        train_dataloader = data2loader(train_data, train_label, 32, True)
        print(len(train_dataloader))
        
        data_dim = train_data.shape[1]

    #     test_data, test_label = df2data(public_test)
        test_data, test_label = df2data(train_df.iloc[test_index], drop_columns=["author_id", "forum_id"])
        test_dataloader = data2loader(test_data, test_label, 32, False)
        print(len(test_dataloader))
        
        kfold_result.append(train(n_epochs, train_dataloader, test_dataloader, data_dim))

    print("load in pkl")
    with open("model_2-3.pkl", "wb") as f:
        pickle.dump(kfold_result, f)
    print("loaded")

if __name__ == "__main__":
    main()