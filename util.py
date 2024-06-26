# -*- coding:utf-8 -*-
#========================================================================
# Copyright (C) 2022 All rights reserved.
# Author: doubleQ
# File Name: util.py
# Created Date: 2024-06-25
# Description:
# =======================================================================

import time
import datetime
import gc
import io
import os
import sys
import base64
import glob
import itertools
import numpy as np
import pandas as pd
import tempfile
import random
import sklearn
import pickle
import shutil
from itertools import combinations
from functools import reduce
import polars as pl
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, KFold, GridSearchCV, GroupKFold
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from argparse import ArgumentParser
import multiprocessing

from lightgbm import early_stopping
import lightgbm as lgb
from lightgbm.sklearn import LGBMRanker
import xgboost as xgb
from catboost import CatBoostRegressor, CatBoostClassifier, CatBoostRanker, Pool

import warnings
warnings.filterwarnings('ignore')

base_dir = './data/'
sparse_feat = ['device_type', 'is_sso_user', 'gender', 'postcode', 
        'age', 'is_subscriber', 'premium', 'article_type', 'category']
target = 'label'
drop_list = ['impression_id', 'label', 'impression_time', 'published_time', 'last_modified_time', 'user_id', 'article_id'] + \
        ['total_seconds', 'hour', 'day_hour', 'day_minute']
SEED = 2023

def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            elif str(col_type)[:5] == 'float':
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df

def func(grouped_df):
    if sum(grouped_df["label"]) == 0 or sum(grouped_df["label"]) == len(grouped_df["label"]):
        return 1.0
    return roc_auc_score(grouped_df["label"], grouped_df["score"])

def gauc(eval_df):
    groups_iter = eval_df.groupby(["impression_id", "user_id"])
    imp, df_groups = zip(*groups_iter)
    pool = multiprocessing.Pool()
    result = pool.map(func, df_groups)
    pool.close()
    pool.join()
    return np.mean(result)

def grank(scores):
    tmp = [(i, s) for i, s in enumerate(scores)]
    tmp = sorted(tmp, key=lambda y: y[-1], reverse=True)
    rank = [(i+1, t[0]) for i, t in enumerate(tmp)]
    rank = [str(r[0]) for r in sorted(rank, key=lambda y: y[-1])]
    rank = "[" + ",".join(rank) + "]"
    return rank

def gen_submit(df):
    df = ( df
            .group_by(["impression_id", "user_id"], maintain_order=True)
            .agg(
                pl.col("score").apply(grank).alias("rank")
            )
        )
    return df

def train_model(model_name, train_x, train_y, cat_features=[], init_model=None, groups=None, epochs=200):

    if model_name == 'lgb':
        clf = lgb
    elif model_name == 'xgb':
        clf = xgb
    else:
        clf = CatBoostClassifier

    if model_name == "lgb":
        if groups is None:
            train_matrix = clf.Dataset(train_x, label=train_y)
            params = {
                'boosting_type': 'gbdt',
                'objective': 'binary',
                'metric': 'auc',
                'max_depth': 6,
                'num_leaves': 2 ** 6,
                'feature_fraction': 0.8,  
                'bagging_fraction': 0.8, 
                'learning_rate': 0.1,  
                'verbose': -1,
                'feature_fraction_seed':SEED,
                'bagging_seed':SEED,
                'seed': SEED,
                'n_jobs': 8
            }

            model = clf.train(params, train_matrix, num_boost_round=epochs, valid_sets=[train_matrix], 
                    categorical_feature=cat_features, callbacks=[lgb.log_evaluation(10)])
            importances = pd.DataFrame({'features': model.feature_name(),
                                    'importances': model.feature_importance()})
            importances.sort_values('importances',ascending=False,inplace=True)
        else:
            ranker = LGBMRanker(
                learning_rate=0.08,
                objective="lambdarank",
                metric="ndcg",
                boosting_type="gbdt",
                importance_type='gain',
                verbose=-1,
                num_threads=20,
                n_estimators=epochs,
                reg_alpha=0.0,
                reg_lambda=1,
                subsample=0.7,
                colsample_bytree=0.7,
                subsample_freq=1,
                device='cpu'
            )
            model = ranker.fit(
                    train_x,
                    train_y,
                    categorical_feature=cat_features,
                    group=groups,
                    eval_set=[(train_x, train_y)], eval_group=[groups], eval_at=5, callbacks=[lgb.log_evaluation(30)]
                )
            importances = pd.DataFrame({'features': model.feature_name_,
                                    'importances': model.feature_importances_})
            importances.sort_values('importances',ascending=False,inplace=True)

    if model_name == "xgb":
        enable= (len(cat_features) > 0)
        if enable:
            train_x[cat_features] = train_x[cat_features].astype('category')
        train_matrix = clf.DMatrix(train_x, label=train_y, enable_categorical=enable)
        if groups is not None:
            train_matrix.set_group(groups)

        params = {'booster': 'gbtree',
                  'objective': 'binary:logistic' if groups is None else 'rank:pairwise',
                  'eval_metric': 'auc' if groups is None else 'ndcg@5',
                  'lambdarank_pair_method': 'topk',
                  'gamma': 1,
                  'min_child_weight': 1.5,
                  'learning_rate': 0.1,
                  'max_depth': 8,
                  'lambda': 10,
                  'subsample': 0.7,
                  'colsample_bytree': 0.8,
                  'colsample_bylevel': 0.8,
                  'eta': 0.1,
                  'seed': SEED,
                  'nthread': 8,
                  'enable_categorical': enable
                  }


        model = clf.train(params, train_matrix, num_boost_round=epochs, 
                evals=[(train_matrix, 'train')], verbose_eval=10, xgb_model=init_model)
        importances = pd.DataFrame(list(model.get_fscore().items()),
            columns=['features', 'importances']).sort_values('importances', ascending=False)
        importances.sort_values('importances',ascending=False,inplace=True)

    if model_name == "cat":
        if groups is not None:
            g = []
            for i, v in enumerate(groups):
                g += [i] * v
            groups = g
            clf = CatBoostRanker
        train_pool = Pool(train_x, train_y, cat_features=cat_features, group_id=groups)
        params = {
                'n_estimators': epochs,
                'learning_rate': 0.1,
                'eval_metric':'AUC' if groups is None else 'NDCG:top=5',
                'loss_function':'Logloss' if groups is None else 'LambdaMart',
                'random_seed':SEED,
                'metric_period':10,
                #'one_hot_max_size': 254,
                'od_wait':500,
                'depth': 8,
                #'task_type': 'GPU',
                'verbose': True,
                }
        model = clf(**params)
        model.fit(train_pool, eval_set=train_pool)
        importances = pd.DataFrame({'features':train_x.columns,
                                'importances':model.feature_importances_})
        importances.sort_values('importances',ascending=False,inplace=True)
    return model, importances

def predict(model_name, data_path, model_path, drop_list=None, feats=None, train_score=None, add_score=None):
    chunk_size = 1000000
    model = load_file(model_save)
    df = load_file(data_path)
    impression_ids = df['impression_id'].tolist()
    user_ids = df['user_id'].tolist()
    labels = df[target].tolist()
    if add_score is not None: 
        df = pd.concat([df, add_score], axis=1)
    if train_score is not None:
        df['train_score'] = train_score
    if feats is not None:
        df = df[feats]
    else:
        df = df.drop(drop_list, axis=1)
    preds = []
    for start in range(0, df.shape[0], chunk_size):
        test_x = df.iloc[start:start + chunk_size]
        if model_name == 'lgb':
            test_pred = model.predict(test_x)
        elif model_name == 'xgb':      
            test_matrix = xgb.DMatrix(test_x)
            test_pred = model.predict(test_matrix)
        else:
            try:
                test_pred = model.predict(test_x)
            except:
                test_pred = model.predict_proba(test_x)[:,1]
        preds.extend(list(test_pred))
    return impression_ids, user_ids, preds, labels

def save_file(file_path, model):
    with open(file_path, 'wb') as fout:
        pickle.dump(model, fout, protocol=4)

def load_file(file_path):
    with open(file_path, 'rb') as fin:
        return pickle.load(fin)

def time_feature(df, field_name):
    ''' 添加时间特征 '''
    df = df.with_columns([
        pl.col(field_name).dt.day().alias('day').cast(pl.Int32),
        pl.col(field_name).dt.hour().alias('hour').cast(pl.Int32),
        pl.col(field_name).dt.minute().alias('minute').cast(pl.Int32),
        pl.col(field_name).dt.weekday().alias('dayofweek').cast(pl.Int32),
        pl.col(field_name).dt.ordinal_day().alias('dayofyear').cast(pl.Int32),
        (pl.col(field_name).dt.timestamp("ms")//1000).alias('total_seconds').cast(pl.Int32),
    ]).with_columns([
        (pl.col('hour') + pl.col('minute') / 60).alias('hour_minute').cast(pl.Float32),
        (pl.col('hour') // 6).alias('6hour').cast(pl.Int32),
        (pl.col('hour') // 12).alias('12hour').cast(pl.Int32),
        ((pl.col('hour') + 6) % 24 // 12).alias('is_day').cast(pl.Int32),
    ]).with_columns([
        (pl.col('dayofweek')*24 + pl.col('hour')).alias('day_hour').cast(pl.Int32),
        (pl.col('dayofweek')*24*60 + pl.col('hour')*60 + pl.col('minute')).alias('day_minute').cast(pl.Int32),
    ])

    return df


def hist_feature(df, df_history):
    dense = ['sentiment_score', 'sentiment_label', 'click_gap']
    df = ( df
        .join(
            df_history.group_by('user_id').agg(
                 [pl.col(c).n_unique().alias(f'{c}_nunique_gby_user').cast(pl.Float32) 
                     for c in ['article_id', 'article_type', 'category']])  
            , on='user_id', how='left')
        .join(
            df_history.group_by('user_id').agg(
                 [pl.col(c).mean().alias(f'{c}_mean_gby_user').cast(pl.Float32) for c in dense])  
            , on='user_id', how='left')
        .join(
            df_history.group_by('user_id').agg(
                 [pl.col(c).min().alias(f'{c}_min_gby_user').cast(pl.Float32) for c in dense])  
            , on='user_id', how='left')
        .join(
            df_history.group_by('user_id').agg(
                 [pl.col(c).max().alias(f'{c}_max_gby_user').cast(pl.Float32) for c in dense])  
            , on='user_id', how='left')
        .join(
            df_history.group_by('user_id').agg(
                 [pl.col(c).std().alias(f'{c}_std_gby_user').cast(pl.Float32) for c in dense])  
            , on='user_id', how='left')
        .join(
            df_history.group_by('article_id').agg(
                 [pl.col(c).n_unique().alias(f'{c}_nunique_gby_article').cast(pl.Float32) 
                     for c in ['user_id']])  
            , on='article_id', how='left')
        .join(
            df_history.group_by('user_id').len(name='user_count')
            , on='user_id', how='left')
        .join(
            df_history.group_by('article_id').len(name='article_count')
            , on='article_id', how='left')
        .join(
            df_history.group_by('article_type').len(name='article_type_count')
            , on='article_type', how='left')
        .join(
            df_history.group_by('category').len(name='category_count')
            , on='category', how='left')
        .join(
            df_history.group_by(['user_id', 'article_id']).len(name='user_article_count')
            , on=['user_id', 'article_id'], how='left')
        .join(
            df_history.group_by(['user_id', 'article_type']).len(name='user_article_type_count')
            , on=['user_id', 'article_type'], how='left')
        .join(
            df_history.group_by(['user_id', 'category']).len(name='user_category_count')
            , on=['user_id', 'category'], how='left')
        )

    return df 

def with_features(df):
    dense = ['view_gap', 'total_inviews', 'total_pageviews', 'total_read_time', 'read_time', 'premium', 'sentiment_score', 'user_read_time_mean', 'user_scroll_percentage_mean', 'session_len']
    df = ( df
        .with_row_index("index").sort('impression_time')
        .with_columns([
           pl.col('impression_time').shift().over('article_id').alias('prev_impression'),
           pl.col('impression_time').shift(-1).over('article_id').alias('next_impression'),
        ])
        .with_columns([
           ((pl.col('impression_time') - pl.col('prev_impression')).dt.total_seconds()).alias('article_before'),
           ((pl.col('impression_time') - pl.col('next_impression')).dt.total_seconds()).alias('article_after'),
        ])
        .with_columns([
           pl.col('impression_time').shift().over(['user_id', 'article_id']).alias('prev_impression'),
           pl.col('impression_time').shift(-1).over('user_id', 'article_id').alias('next_impression'),
        ])
        .with_columns([
           ((pl.col('impression_time') - pl.col('prev_impression')).dt.total_seconds()).alias('user_article_before'),
           ((pl.col('impression_time') - pl.col('next_impression')).dt.total_seconds()).alias('user_article_after'),
        ])
        .with_columns([
           pl.col('impression_time').shift(2).over(['user_id', 'article_id']).alias('prev_impression'),
           pl.col('impression_time').shift(-2).over('user_id', 'article_id').alias('next_impression'),
        ])
        .with_columns([
           ((pl.col('impression_time') - pl.col('prev_impression')).dt.total_seconds()).alias('user_article_before_2'),
           ((pl.col('impression_time') - pl.col('next_impression')).dt.total_seconds()).alias('user_article_after_2'),
        ])
        .with_columns(
            [ pl.col(c).rolling_mean(window_size=3).alias(f'{c}_mean3') for c in ['article_before', 'article_after'] 
              + ['user_article_before', 'user_article_after'] + dense]  
            +[ pl.col(c).rolling_mean(window_size=5).alias(f'{c}_mean5') for c in ['article_before', 'article_after'] 
              + ['user_article_before', 'user_article_after'] + dense]  
            +[ pl.col(c).rolling_mean(window_size=7).alias(f'{c}_mean7') for c in ['article_before', 'article_after'] 
              + ['user_article_before', 'user_article_after'] + dense]  
        )
        .with_columns([
            pl.col('impression_id').len().over('user_id').alias(name='full_user_count'),
            pl.col('impression_id').len().over('article_id').alias(name='full_article_count'),
            pl.col('impression_id').len().over(['dayofyear', 'user_id']).alias(name='day_user_count'),
            pl.col('impression_id').len().over(['dayofyear', 'article_id']).alias(name='day_article_count'),
            pl.col('impression_id').len().over(['day_hour', 'user_id']).alias(name='hour_user_count'),
            pl.col('impression_id').len().over(['day_hour', 'article_id']).alias(name='hour_article_count'),
            pl.col('impression_id').len().over(['day_minute', 'user_id']).alias(name='minute_user_count'),
            pl.col('impression_id').len().over(['day_minute', 'article_id']).alias(name='minute_article_count'),
            pl.col('impression_id').len().over(['total_seconds', 'user_id']).alias(name='second_user_count'),
            pl.col('impression_id').len().over(['total_seconds', 'article_id']).alias(name='second_article_count'),
            pl.col('impression_id').len().over(['dayofyear', 'user_id', 'article_type']).alias(name='day_user_type_count'),
            pl.col('impression_id').len().over(['dayofyear', 'user_id', 'category']).alias(name='day_user_category_count'),
        ])
    )

    exprs = []
    for pivot in ['user_id', 'article_id']:
        exprs += [pl.col(c).n_unique().over(pivot).alias(f'{c}_nunique_gby_{pivot}').cast(pl.Float32) 
                 for c in ['article_id', 'article_type', 'category', 'user_id', 'device_type', 'postcode', 'age']]  
        exprs += [pl.col(c).min().over(pivot).alias(f'{c}_min_gby_{pivot}').cast(pl.Float32) for c in dense] 
        exprs += [pl.col(c).max().over(pivot).alias(f'{c}_max_gby_{pivot}').cast(pl.Float32) for c in dense] 
        exprs += [pl.col(c).mean().over(pivot).alias(f'{c}_mean_gby_{pivot}').cast(pl.Float32) for c in dense] 
        exprs += [pl.col(c).median().over(pivot).alias(f'{c}_median_gby_{pivot}').cast(pl.Float32) for c in dense] 
        exprs += [pl.col(c).std().over(pivot).alias(f'{c}_std_gby_{pivot}').cast(pl.Float32) for c in dense] 
        exprs += [pl.col(c).skew().over(pivot).alias(f'{c}_skew_{pivot}').cast(pl.Float32) for c in dense]
        exprs += [pl.col(c).first().over(pivot).alias(f'{c}_first_{pivot}').cast(pl.Float32) for c in dense]
        exprs += [pl.col(c).last().over(pivot).alias(f'{c}_last_{pivot}').cast(pl.Float32) for c in dense]
        exprs += [(pl.col(c).max() - pl.col(c).min()).over(pivot).alias(f'{c}_maxmin_{pivot}').cast(pl.Float32) for c in dense]
        for time in ['dayofyear', 'day_hour', 'day_minute', 'total_seconds']:
            exprs += [pl.col(c).n_unique().over([time, pivot]).alias(f'{c}_nunique_gby_{pivot}_{time}').cast(pl.Float32) 
                 for c in ['article_id', 'article_type', 'category', 'user_id', 'device_type', 'postcode', 'age']]
            exprs += [pl.col(c).min().over([time, pivot]).alias(f'{c}_min_gby_{pivot}_{time}').cast(pl.Float32) for c in dense]
            exprs += [pl.col(c).max().over([time, pivot]).alias(f'{c}_max_gby_{pivot}_{time}').cast(pl.Float32) for c in dense]
            exprs += [pl.col(c).mean().over([time, pivot]).alias(f'{c}_mean_gby_{pivot}_{time}').cast(pl.Float32) for c in dense]
            exprs += [pl.col(c).median().over([time, pivot]).alias(f'{c}_median_gby_{pivot}_{time}').cast(pl.Float32) for c in dense]
            exprs += [pl.col(c).std().over([time, pivot]).alias(f'{c}_std_gby_{pivot}_{time}').cast(pl.Float32) for c in dense]
            exprs += [pl.col(c).skew().over([time, pivot]).alias(f'{c}_skew_{pivot}_{time}').cast(pl.Float32) for c in dense]
            exprs += [pl.col(c).first().over([time, pivot]).alias(f'{c}_first_{pivot}_{time}').cast(pl.Float32) for c in dense]
            exprs += [pl.col(c).last().over([time, pivot]).alias(f'{c}_last_{pivot}_{time}').cast(pl.Float32) for c in dense]
            exprs += [(pl.col(c).max() - pl.col(c).min()).over([time, pivot]).alias(f'{c}_maxmin_{pivot}_{time}').cast(pl.Float32) for c in dense]
    df = ( df
        .with_columns(
            exprs    
        )
        .sort('index')
        .drop(['index', 'prev_impression', 'next_impression'])
    )
    
    return df

def polars_data(dataset, split, filter_feats=None):
    
    df_article = (
         pl.scan_parquet(f'{base_dir}/{dataset}/articles.parquet')
         .with_columns([
             pl.col(c).len().alias(f'{c}_len').cast(pl.Float32) 
                 for c in ['title', 'subtitle', 'body', 'ner_clusters', 'entity_groups', 'topics']
         ])
         .with_columns([
             pl.col('sentiment_label').map_elements(
                 lambda x: 1 if x=='Positive' else (0 if x=='Neutral' else 0), return_dtype=pl.Int8),
             pl.col('premium').cast(pl.Float32),
             (pl.col('total_read_time')/pl.col('total_pageviews')).cast(pl.Float32).alias('read_pageviews'),
             (pl.col('total_read_time')/pl.col('total_inviews')).cast(pl.Float32).alias('read_inviews'),
             (pl.col('total_read_time')/pl.col('body_len')).cast(pl.Float32).alias('read_time_word'),
         ])
         .drop(['title', 'subtitle', 'body', 'image_ids', 'url', 'ner_clusters', 
             'entity_groups', 'topics', 'category_str', 'subcategory'])

    )
    user_history = (
         pl.scan_parquet(f'{base_dir}/{dataset}/{split}/history.parquet')
         .with_columns([
             pl.col('read_time_fixed').list.mean().alias('user_read_time_mean'),
             pl.col('scroll_percentage_fixed').list.mean().alias('user_scroll_percentage_mean'),
             pl.col('read_time_fixed').list.median().alias('user_read_time_median'),
             pl.col('scroll_percentage_fixed').list.median().alias('user_scroll_percentage_median'),
             pl.col('impression_time_fixed').list.diff().list.median().dt.total_seconds().alias('user_impression_diff'),
         ])
         .select(['user_id', 'user_read_time_mean', 'user_scroll_percentage_mean'])
    )
    df_behavior = (
         pl.scan_parquet(f'{base_dir}/{dataset}/{split}/behaviors.parquet')
         .with_row_index("index").sort('impression_time')
         .with_columns([
            pl.col('impression_time').shift().over('user_id').alias('prev_impression'),
            pl.col('impression_time').shift(-1).over('user_id').alias('next_impression'),
         ])
         .with_columns([
            ((pl.col('impression_time') - pl.col('prev_impression')).dt.total_seconds()).alias('user_before'),
            ((pl.col('impression_time') - pl.col('next_impression')).dt.total_seconds()).alias('user_after'),
         ])
         .sort('index')
         .drop(['index', 'prev_impression', 'next_impression'])
         .with_columns([
            pl.col('article_ids_inview').list.len().alias('session_len'),
         ])
         .explode(['article_ids_inview'])
         .with_columns(
            label = pl.col('article_ids_inview').is_in(pl.col('article_ids_clicked')).cast(pl.Int8) if split!='test' else -1
         )
         .select(['impression_id', 'label', 'impression_time', 'device_type', 'article_ids_inview', 'user_id', 
             'is_sso_user', 'gender', 'postcode', 'age', 'is_subscriber', 'read_time', 'user_before', 'user_after', 'session_len'])
         .rename({'article_ids_inview': 'article_id'})
         .join(df_article, on='article_id', how="left")
         .with_columns(
             ((pl.col('impression_time') - pl.col('published_time')).dt.total_seconds()).alias('view_gap'),
         )
         .join(user_history, on='user_id', how="left")
    )
    df_history = (
         pl.scan_parquet(f'{base_dir}/{dataset}/{split}/history.parquet')
         .explode(['article_id_fixed', 'impression_time_fixed', 'read_time_fixed', 'scroll_percentage_fixed'])
         .rename(lambda x: x[:-6] if x.endswith('_fixed') else x)
         .join(df_article, on='article_id', how="left")
         .with_columns(
             ((pl.col('impression_time') - pl.col('published_time')).dt.total_seconds()).alias('click_gap'),
         )
     )
    df_behavior = time_feature(df_behavior, 'impression_time')
    df_history = time_feature(df_history, 'impression_time')
    df_behavior = hist_feature(df_behavior, df_history)
    df_behavior = with_features(df_behavior)
    if filter_feats is not None:
        filter_feats += ['impression_id', 'user_id', 'article_id', 'label']
    df_behavior = df_behavior.select(
            [col for col in df_behavior.columns if col in filter_feats] if filter_feats is not None else pl.all())
    return df_behavior.collect()

def save_data(dataset, split, save_path, feature_map=None, filter_feats=None):
    data = polars_data(dataset, split, filter_feats=filter_feats)
    maps = dict()
    if filter_feats is not None:
        encoder_feat = [f for f in sparse_feat if f in filter_feats]
    else:
        encoder_feat = sparse_feat
    if feature_map is None:
        for feat in encoder_feat:
            le = { v: k+1 for k, v in enumerate(sorted(list(set(data[feat].drop_nulls())))) }
            maps[feat] = le
            data = data.with_columns( pl.col(feat).replace(le, default=0).alias(feat) )
    else:
        for feat in encoder_feat:
            le = feature_map[feat]
            data = data.with_columns( pl.col(feat).replace(le, default=0).alias(feat) )
    data = data.to_pandas()
    data = reduce_mem_usage(data)
    if save_path.endswith('pkl'):
        save_file(save_path, data)
    else:
        data.to_csv(save_path, index=False)
    return maps

def load_top_features(feature_path, topk=300):
    important_features = pd.read_csv(feature_path)
    feats = important_features.iloc[:topk]['features'].tolist()
    return feats

