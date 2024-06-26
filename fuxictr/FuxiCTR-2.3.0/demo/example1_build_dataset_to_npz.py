import sys
sys.path.append('../')
import os
from fuxictr import datasets
from fuxictr.preprocess import FeatureProcessor, build_dataset
from fuxictr.utils import load_dataset_config, set_logger, print_to_json
import logging

if __name__ == '__main__':
    # Load params from config files
    config_dir = './config/example1_config'
    dataset_id = 'tiny_example1'
    params = load_dataset_config(config_dir, dataset_id)

    # set up logger
    set_logger(params)
    logging.info("Params: " + print_to_json(params))

    # Set up feature encoder
    feature_encoder = FeatureProcessor(feature_cols=params["feature_cols"],
                                       label_col=params["label_col"],
                                       dataset_id=dataset_id, 
                                       data_root=params["data_root"])
                                       
    # Build dataset
    build_dataset(feature_encoder, 
                  train_data=params["train_data"],
                  valid_data=params["valid_data"],
                  test_data=params["test_data"])
