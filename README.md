# recsys-challenge-2024

This is the code implementation of [RecSys Challenge 2024](https://recsys.eb.dk/) focusing on online news recommendation by Ekstra Bladet and JP/Politikens Hus A/S ("Ekstra Bladet").

Our organization is **hrec** (captain: doubleQ).

## Usage

Machine: 36 cores, 2T memory, 2 * V100

### Unzip the data files in `./data/`

    data/
    ├── image_embeddings.parquet
    ├── contrastive_vector.parquet
    ├── small
    │   ├── train
    │   ├── validation
    │   ├── articles.parquet
    ├── large
    │   ├── train
    │   ├── validation
    │   ├── articles.parquet
    ├── ebnerd_testset
    │   ├── test
    │   ├── articles.parquet

### Generate base features
- `python preprocess.py`
- The base features will be in `preprocessed/large`

### Generate din and dcn prediction scores based on the preprocessed features
- `cd fuxictr`
- `python convert_data.py` # generate fuxictr features in `Ebnerd_large_data/`
- `python run_expid.py --config config/ebnerd_large_tuner_config --expid DCN_Ebnerd_large_001 --gpu 0`
- `python run_expid.py --config config/ebnerd_large_tuner_config --expid DIN_Ebnerd_large_001 --gpu 0`
This step will generate scores in `fuxictr/features`

### Train xgboost/lightgbm with pairwise loss or BCE base on these features 

- `python main.py --model_name xgb --mode large --rank` will get result in `result/xgb_submit.zip`
- `python main.py --model_name xgb --mode large` will get result in `result/xgb_submit_binary.zip`

### ensemble to get final result
-- `python en.py --files result/xgb_submit.csv result/xgb_submit_binary.csv --weights 0.5 0.25 --output final_submit`
-- The final result will generated within `result/final_submit.zip`
