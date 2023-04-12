import xgboost as xgb
import pickle
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_percentage_error as MAPE
from main import df2data, data2loader, get_forum_dict, get_author_dict

train_df = pd.read_csv("./dataset/intern_homework_train_dataset.csv")
public_test = pd.read_csv("./dataset/intern_homework_public_test_dataset.csv")
forum_dict = get_forum_dict(train_df)
# author_dict = get_author_dict(train_df)

param_test = {
    'max_depth': list(range(3, 10, 2)),
    'min_child_weight': list(range(1, 6, 2))
}

train_data, train_labels = df2data(train_df, drop_columns=["author_id", "forum_id", "created_at", "title", "like_count_1h", "comment_count_1h"])
public_test_data, public_test_labels = df2data(public_test, drop_columns=["author_id", "forum_id"])

# for max_depth in param_test["max_depth"]:
#     for min_child_weight in param_test["min_child_weight"]:
#         print("max_depth: ", max_depth, "min_child_weight: ", min_child_weight)
#         model = xgb.XGBRegressor(
#             max_depth=max_depth,
#             min_child_weight=min_child_weight,
#             learning_rate=0.1,
#             n_estimators=1000,
#             objective='reg:squarederror',
#             booster='gbtree',
#             gamma=0,
#             subsample=0.8,
#             colsample_bytree=0.8,
#             reg_alpha=0,
#             reg_lambda=1,
#             scale_pos_weight=1,
#             seed=42,
#         )
#         model.fit(train_data, train_labels)
#         y_train = model.predict(train_data)
#         y_train = np.expand_dims(y_train, axis=1)
#         train_mape = MAPE(train_labels, y_train)
#         y_pred = model.predict(public_test_data)
#         y_pred = np.expand_dims(y_pred, axis=1)
#         mape = MAPE(public_test_labels, y_pred)
#         print(f"training loss: {train_mape}, test loss: {mape}")

model = xgb.XGBRegressor(
    max_depth=6,
    n_estimators=800,
    objective='reg:squarederror',
    booster='gbtree',
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0,
    reg_lambda=1,
    scale_pos_weight=1,
    seed=42,
)
model.fit(train_data, train_labels)
y_train = model.predict(train_data)
y_train = np.expand_dims(y_train, axis=1)
train_mape = MAPE(train_labels, y_train)
y_pred = model.predict(public_test_data)
y_pred = np.expand_dims(y_pred, axis=1)
mape = MAPE(public_test_labels, y_pred)
print(f"training loss: {train_mape}, test loss: {mape}")
print(model.feature_importances_)

