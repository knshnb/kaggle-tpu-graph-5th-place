# 5th Place Solution of Kaggle Fast or Slow Competition
This is knshnb's solution for ["Google - Fast or Slow? Predict AI Model Runtime"](https://www.kaggle.com/competitions/predict-ai-model-runtime)

## Run
### Dataset Preparetion
1. Download the competition dataset under `input/`.
```
$ ls -F input
npz_all/  pb/  sample_submission.csv
```
2. Run the following command and make sure npy files are generated under `input/npz_all/npz/layout-config/`.
```
$ python -m scripts.preprocess_layout_data
```

### Training and Inference
Layout model:
```
python -m scripts.train --save_model --config_path config/default.yaml --exp_name layout-exp
```

Tile model:
```
python -m scripts.train --save_model --config_path config/tile.yaml --exp_name tile-exp
```

By running the above commands, the trained models and inference results will be saved under `result/layout-exp/0/` and `result/tile-exp/0/` respectively.

### Submission
By specifying the directories generated in the previous step, you can make a submission file for the competition.

Example:
```
python -m scripts.make_submission \
--layout_model_dirs result/layout-exp/0 \
--tile_model_dirs result/tile-exp/0
```

## Links
- For an overview of our key ideas and detailed explanation, please also refer to [5th Place Solution: GNN with Invariant Dimension Features](https://www.kaggle.com/competitions/predict-ai-model-runtime/discussion/456093)
