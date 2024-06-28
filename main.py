# -*- coding:utf-8 -*-
#========================================================================
# Copyright (C) 2024 All rights reserved.
# Author: doubleQ
# File Name: train.py
# Created Date: 2024-06-25
# Description:
# =======================================================================

from argparse import ArgumentParser
import warnings

from util import *
warnings.filterwarnings('ignore')

fuxictr_features = ['DCN_Ebnerd_large_001', 'DIN_Ebnerd_large_001']

def dnn_score(dataset, split):
    data = pd.DataFrame()
    for feature_dir in fuxictr_features:
        scores = pd.read_csv(f'fuxictr/features/{feature_dir}/{dataset}_{split}.csv')
        data[feature_dir] = scores['score']
    data['en_score'] = data[fuxictr_features].values.sum(axis=1)
    return data

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default='xgb')
    parser.add_argument("--mode", type=str, default='small')
    parser.add_argument("--tag", type=str, default='submit')
    parser.add_argument("--rank", action='store_true', default=False)
    args = parser.parse_args()
    print(args)
    model_name = args.model_name

    
    train_dnn = dnn_score('large', 'train')
    valid_dnn = dnn_score('large', 'validation')
    test_dnn = dnn_score('ebnerd_testset', 'test')

    use_feats = load_top_features('preprocessed/feature_importance.csv')
    use_feats += fuxictr_features + ['en_score']
    print('use features:', use_feats)

    data_path = 'preprocessed/large'
    
    print('train on train data')
    x_train = load_file(f'{data_path}/train.pkl')
    x_train = pd.concat([x_train, train_dnn], axis=1)
    if args.rank:
        groups = x_train.groupby(['impression_id', 'user_id'])['article_id'].count().values
    else:
        args.tag += '_binary'
        groups = None
    os.makedirs('models', exist_ok=True)
    model_save = 'models/{}_{}.pkl'.format(model_name, args.tag)

    y_train = x_train[target]
    x_train = x_train[use_feats]
    sparse_feat = [c for c in sparse_feat if c in use_feats]
    model, feature_importance_df = train_model(model_name, x_train, y_train, cat_features=[], groups=groups)
    save_file(model_save, model)
    _, _, valid_score, _ = predict(model_name, f'{data_path}/valid.pkl', model_save, feats=use_feats, add_score=valid_dnn)
    _, _, test_score, _ = predict(model_name, f'{data_path}/test.pkl', model_save, feats=use_feats, add_score=test_dnn)

    print('train on valid data')
    x_train = load_file(f'{data_path}/valid.pkl')
    x_train['train_score'] = valid_score
    x_train = pd.concat([x_train, valid_dnn], axis=1)
    use_feats += ['train_score']
    if args.rank:
        groups = x_train.groupby(['impression_id', 'user_id'])['article_id'].count().values
    else:
        args.tag += '_binary'
        groups = None
    y_train = x_train[target]
    x_train = x_train[use_feats]
    model, feature_importance_df = train_model(model_name, x_train, y_train, cat_features=[], groups=groups, epochs=800)
    save_file(model_save, model)

    print('predicting')
    impression_ids, user_ids, preds, labels = predict(model_name, f'{data_path}/test.pkl', model_save, 
            feats=use_feats, train_score=test_score, add_score=test_dnn)
    ans = pl.DataFrame({
        'impression_id': impression_ids,
        'user_id': user_ids,
        'score': preds
    })

    os.makedirs('result', exist_ok=True)
    submit_tag = '{}'.format(args.tag)
    ans.write_csv('result/{}_{}.csv'.format(model_name, submit_tag))
    ans = gen_submit(ans)
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(f'{tmpdir}/predictions.txt', "w") as fout:
            out = [str(row['impression_id']) + " " + row['rank'] for row in ans.iter_rows(named=True)]
            fout.write("\n".join(out))
        shutil.make_archive('result/{}_{}'.format(model_name, submit_tag), 'zip', tmpdir, 'predictions.txt')
