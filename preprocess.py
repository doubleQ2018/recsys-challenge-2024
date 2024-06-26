# -*- coding:utf-8 -*-
#========================================================================
# Copyright (C) 2024 All rights reserved.
# Author: doubleQ
# File Name: preprocess.py
# Created Date: 2024-06-25
# Description:
# =======================================================================

from util import *

# train test data preprocess
mode = 'small'

def data_generate(mode):
    data_path = f'preprocessed/{mode}'
    os.makedirs(data_path, exist_ok=True)
    if mode == 'large':
        use_feats = load_top_features('preprocessed/feature_importance.csv')
        feature_map = save_data('large', 'train', f'{data_path}/train.pkl', filter_feats=use_feats)
        save_data('large', 'validation', f'{data_path}/valid.pkl', feature_map, filter_feats=use_feats)
        save_data('ebnerd_testset', 'test', f'{data_path}/test.pkl', feature_map, filter_feats=use_feats)
    else:
        use_feats = None
        feature_map = save_data('small', 'train', f'{data_path}/train.pkl', filter_feats=use_feats)
        save_data('small', 'validation', f'{data_path}/valid.pkl', feature_map, filter_feats=use_feats)
    return

if __name__ == '__main__':
    os.makedirs('preprocessed', exist_ok=True)

    '''
    start = time.time()
    data_generate('small')
    print('small data cost {:.1f} mins'.format((time.time()-start)/60))
    x_train = load_file(f'preprocessed/small/train.pkl')
    print(x_train.columns)
    groups = x_train.groupby(['impression_id', 'user_id'])['article_id'].count().values
    y_train = x_train[target]
    x_train = x_train.drop(drop_list, axis=1 )
    model, feature_importance_df = train_model('xgb', x_train, y_train, cat_features=[], groups=groups)
    print(feature_importance_df.iloc[:50])
    feature_importance_df.to_csv('preprocessed/feature_importance.csv', index=False)
    '''
    start = time.time()
    data_generate('large')
    print('large data cost {:.1f} mins'.format((time.time()-start)/60))
