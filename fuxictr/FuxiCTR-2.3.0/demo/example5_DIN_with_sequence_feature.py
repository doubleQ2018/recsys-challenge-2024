import sys
sys.path.append('../')
import os
import logging
from datetime import datetime
from fuxictr import datasets
from fuxictr.utils import load_config, set_logger, print_to_json
from fuxictr.features import FeatureMap
from fuxictr.pytorch.torch_utils import seed_everything
from fuxictr.pytorch.dataloaders import RankDataLoader
from model_zoo import DIN


if __name__ == '__main__':
    # Load params from config files
    config_dir = './config/example5_config'
    experiment_id = 'DIN_test'
    params = load_config(config_dir, experiment_id)

    # set up logger and random seed
    set_logger(params)
    logging.info("Params: " + print_to_json(params))
    seed_everything(seed=params['seed'])

    # Load feature_map from json
    data_dir = os.path.join(params['data_root'], params['dataset_id'])
    feature_map = FeatureMap(params['dataset_id'], data_dir)
    feature_map.load(os.path.join(data_dir, "feature_map.json"), params)
    logging.info("Feature specs: " + print_to_json(feature_map.features))
    
    # Get train and validation data generators
    train_gen, valid_gen = RankDataLoader(feature_map, 
                                          stage='train', 
                                          train_data=params['train_data'],
                                          valid_data=params['valid_data'],
                                          batch_size=params['batch_size'],
                                          shuffle=params['shuffle']).make_iterator()

    # Model initialization and fitting
    model = DIN(feature_map, **params)
    model.fit(train_gen, validation_data=valid_gen, epochs=params['epochs'])
    
    logging.info('***** Validation evaluation *****')
    model.evaluate(valid_gen)

    logging.info('***** Test evaluation *****')
    test_gen = RankDataLoader(feature_map, 
                              stage='test',
                              test_data=params['test_data'],
                              batch_size=params['batch_size'],
                              shuffle=False).make_iterator()
    model.evaluate(test_gen)
