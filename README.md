# Guess Like Count After 24 hr

## Train the model
You can also tune the model by setting various parameter.
- `normalization`
- `regularize`
- `epoch`
- `batch_size`

If you don't want to use normalization, just don't add it to argument.
```
python3 main.py --normalization {normalization_method} --output {output_name.pkl}
```
Here I save the `best_loss`, `best_weights`, `train_losses`, `eval_losses` in output (pkl file).

Note: If you failed to run the code, please check whether you download `cc.zh.300.bin`.

## Evaluate the model
```
python3 evaluate.py --normalization {normalization_method} --ckpt {checkpoint_name.pkl}
```

## Generate Prediction (result.csv)
```
python3 predict.py --normalization {normalization_method} --ckpt {checkpoint_name.pkl}
```

For the best model checkpoint, please download from https://github.com/wazenmai/Guess_Like_Count_24h/blob/main/best_model.pkl

---

# Other Notes

## Download FastText Model

Please run `download.py` or run below code.
```
import fasttext.util
fasttext.util.download_model('zh', if_exists='ignore')
```

## Data Analysis
For dataset analysis, please see `Analysis.ipynb` for more details. It also include the training curve and the information about fasttext chinese embedding.
I include following analysis in the code.
1. Numerial Analysis
2. Relationship between Created Time and Like Count
3. Like Count Analysis
4. Comment Count Analysis
5. Relationship between Like Count and Comment Count
6. Hot Topic Analysis

## Feature Importance
```
python3 feature_importance.py --ckpt {checkpoint_name.pkl}
```
