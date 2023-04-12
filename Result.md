
```
# Construct model
model = nn.Sequential(
    nn.Linear(data_dim, 500),
    nn.ReLU(),
    nn.Linear(500, 100),
    nn.ReLU(),
    nn.Linear(100, 10),
    nn.ReLU(),
    nn.Linear(10, 1)
)
# 0.215121 M Parameters
```
- included all information: model can not be trained
- drop created_at, author_id, forum_id: 
    - public_test_loss:  0.3128125036478043
- drop author_id, forum_id: public_test_loss: 
    - weekday as one-hot, period as one hot
    - public_test_loss:  0.30869804294109343
- add forum mean and author mean:
    - include one-hot created_at
    - not include author_id and forum_id
    - public_test_loss: 0.4030624141216278
- add forum mean and drop author_id:
    - public_test_loss: 0.30803462636470796

```
model = nn.Sequential(
    nn.Linear(data_dim, 512),
    nn.ReLU(),
    nn.Linear(512, 1024),
    nn.ReLU(),
    nn.Linear(1024, 256),
    nn.ReLU(),
    nn.Linear(256, 1)
)
# 0.948737 M Parameters
```
1. add forum mean and drop author_id:
    - loss:  0.3053166998147964
2. drop forum, author, created_at:
    - loss:  0.3110107444524765
3. drop forum and author id:
    - loss:  0.3073045672655106
4. drop forum and author id, standardize like, comment and forum_stats count, training loss use squared error:
    - loss: 0.7434546798229218
5. based on 4, change standardization to normalization, add regularization, reg_strength = 0.001:
    - loss:  0.5577323312759399
6. based on 5, use denormalized the predicted value and use MAPE when training
    - 
7. based on 6, use denormalized value for evaluation, train on GPU
    - loss:  0.4106618849754333

```
model = nn.Sequential(
        nn.Linear(data_dim, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 1)
    )
# 0.977921 M Parameters
```
With regularization
1. No normalization, drop author_id, forum_id
    - loss:  0.3080337948083878
2. No normalization, drop author_id, forum_id, created_at:
    - loss:  0.3114083482027054

```
model = nn.Sequential(
        nn.Linear(data_dim, 1024),
        nn.ReLU(),
        nn.Linear(1024, 2048),
        nn.ReLU(),
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Linear(128, 1)
    )
# 3.535617 M Parameters
```

Without Regularization
1. drop author_id, use forum mean:
    - loss:  0.304154496049881
2. drop author_id, forum_id: 
    - loss:  0.3080479504108429
3. Based on 1, add regularization, reg_strength = 0.001
    - loss:  0.3085175615549087
4. Based on 3, use simplified chinese:
    - loss:  0.3069587200641632
5. Based on 1, use simplified chinese:
    - loss:  0.30490859050750735
6. Based on 5, use learning rate = 0.002:
    - loss:  0.3046501812696457
7. Based on 1, use SGD:
    - loss:  0.33205798792839053
8. Based on 1, use SGD with learning rate 0.0005:
    - loss:  0.3439116889476776
9. Based on 1, fix the chinese vector
    - loss:  0.307127286362648
10. Based on 9, use standardization data
    - loss:  0.36994197726249695
11. Based on 9, use normalization (minmax) data
    - loss:  0.45771909070014954


model = xgb.XGBRegressor(
    n_estimators=1000,
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
```
1. drop author_id, forum_id:
    - training loss: 0.4103661758675394
    - test loss: 0.8469147813858188
2. Based on 1, max_depth = 8, n_estimator = 800
    - training loss: 0.1522033147612761
    - test loss: 0.8909031225740309
    
