
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
1. add forum mean and drop author_id:
    - loss:  0.3053166998147964
2. drop forum, author, created_at:
    - loss:  0.3110107444524765
3. drop forum and author id:
    - loss:  0.3073045672655106
    
