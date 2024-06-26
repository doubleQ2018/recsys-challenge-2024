import polars as pl
import numpy as np
import pickle
import os
from pandas.core.common import flatten
from datetime import datetime
from sklearn.decomposition import PCA
import gc
from keras_preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from datetime import datetime
import json


# Download the datasets and put them to the following folders
base_dir = '../data'
dataset = 'large'
dataset_version = f"Ebnerd_{dataset}_data"

train_path = f"{base_dir}/{dataset}/train/"
dev_path = f"{base_dir}/{dataset}/validation/"
if dataset == 'large':
    test_path = f"{base_dir}/ebnerd_testset/test/"
else:
    test_path = dev_path
image_emb_path = f"{base_dir}/image_embeddings.parquet"
contrast_emb_path = f"{base_dir}/contrastive_vector.parquet"
MIN_COUNT = 10

feature_vocab_size = dict()
feature_vocab_dict = dict()
feature_max_len = dict()

print("Preprocess news info...")
os.makedirs(dataset_version, exist_ok=True)
train_news_file = os.path.join(train_path, "../articles.parquet")
train_news = pl.scan_parquet(train_news_file)
test_news_file = os.path.join(test_path, "../articles.parquet")
test_news = pl.scan_parquet(test_news_file)
news = pl.concat([train_news, test_news]).collect()
news = news.unique(subset=['article_id'])
num_items = len(news)
news = news.with_columns(pl.arange(1, 1 + num_items).alias("item_index"))
article_id2index = dict(zip(news.select("article_id").to_series().to_list(),
                            news.select("item_index").to_series().to_list()))
feature_vocab_size["item_index"] = len(article_id2index) + 1
print("Number of items: ", num_items)

def onehot_encoding(df, column, min_count=MIN_COUNT):
    vocab = df.select(column).to_series().to_pandas().value_counts(dropna=True).to_dict()
    vocab = dict((k, v) for k, v in vocab.items() if v >= min_count)
    vocab2index = dict((k, idx + 1) for idx, k in enumerate(vocab.keys()))
    oov_id = len(vocab2index) + 1
    feature_vocab_dict[column] = vocab2index
    feature_vocab_size[column] = oov_id + 1
    return vocab2index

def multihot_encoding(df, column, min_count=MIN_COUNT):
    vocab = df.select(column).to_series().explode().to_pandas().value_counts(dropna=True).to_dict()
    vocab = dict((k, v) for k, v in vocab.items() if v >= min_count)
    vocab2index = dict((k, idx + 1) for idx, k in enumerate(vocab.keys()))
    oov_id = len(vocab2index) + 1
    feature_vocab_dict[column] = vocab2index
    feature_vocab_size[column] = oov_id + 1
    return vocab2index

def encoding_and_padding(input_list, vocab2index, max_len=5):
    oov_id = len(vocab2index) + 1
    output_list = [vocab2index.get(x, oov_id) for x in input_list if x in vocab2index]
    output_list = output_list[:max_len]
    output_list = list(pad_sequences([output_list], maxlen=max_len, value=0,
                       padding="post", truncating="post")[0])
    return output_list

def numeric_encoding(df, column):
    values = df.select(column).to_series().to_numpy().reshape(-1, 1)
    scaler = MinMaxScaler()
    scaler.fit(values)
    return scaler

news = news.select(["item_index", 'article_id', 'published_time', 'last_modified_time', 'premium',
                    'article_type', 'ner_clusters', 'topics', 'category', 'subcategory',
                    'total_inviews', 'total_pageviews', 'total_read_time',
                    'sentiment_score', 'sentiment_label'])
multihot_columns = ["ner_clusters", "topics", "subcategory"]
for column in multihot_columns:
    vocab2index = multihot_encoding(news, column)
    if column in ["ner_clusters", "topics"]:
        max_len = 10
    else:
        max_len = 5
    feature_max_len[column] = max_len
    news = news.with_columns(
        pl.col(column).apply(lambda x: encoding_and_padding(x, vocab2index, max_len))
    )

onehot_columns = ['premium', 'article_type', 'category', 'sentiment_label']
for column in onehot_columns:
    vocab2index = onehot_encoding(news, column)
    news = news.with_columns(
        pl.col(column).apply(lambda x: vocab2index.get(x, len(vocab2index) + 1))
        .fill_null(len(vocab2index) + 1)
    )

numeric_columns = ['total_inviews', 'total_pageviews', 'total_read_time', 'sentiment_score']
def log_transform(x):
    return np.log10(1 + x)

for column in numeric_columns:
    news = news.with_columns(pl.col(column).fill_null(0))
    if column.startswith("total_"):
        news = news.with_columns(pl.col(column).apply(log_transform, return_dtype=pl.Float32))
    scaler = numeric_encoding(news, column)
    news = news.with_columns(pl.col(column).apply(lambda x: scaler.transform([[x]])[0][0]))

print("Preprocess behavior history data...")
train_hist = pl.scan_parquet(os.path.join(train_path, "history.parquet"))
valid_hist = pl.scan_parquet(os.path.join(dev_path, "history.parquet"))
test_hist = pl.scan_parquet(os.path.join(test_path, "history.parquet"))
hist_df = pl.concat([train_hist, valid_hist, test_hist]).collect()
hist_df = hist_df.rename({"article_id_fixed": "hist_id", 
                          "read_time_fixed": "hist_read_time",
                          "impression_time_fixed": "hist_time",
                          "scroll_percentage_fixed": "hist_scroll_percent"})
user_df = hist_df.unique(subset=["user_id"])
num_users = len(user_df)
user_id2index = dict(zip(user_df.select("user_id").to_series().to_list(),
                         range(1, 1 + num_users)))
feature_vocab_size["user_index"] = len(user_id2index) + 1
print("Number of users: ", num_users)
hist_df = (
    hist_df.with_columns(pl.col("user_id").apply(lambda x: user_id2index.get(x)).alias("user_index"))
    .explode(['hist_id', "hist_read_time", "hist_time", "hist_scroll_percent"])
    .unique(subset=["user_index", 'hist_id'])
    .fill_null(0)
    .with_columns(pl.col("hist_id").apply(lambda x: article_id2index.get(x)))
    .drop_nulls(subset="hist_id")
    .sort(by=["user_index", "hist_time"])
)
user_seq_ids = hist_df.groupby("user_index", maintain_order=True).agg(pl.col('hist_id'))
user_info = pd.DataFrame(
    {"user_index": [0] + user_seq_ids.select("user_index").to_series().to_list(),
     "user_seq_ids": [[]] + user_seq_ids.select("hist_id").to_series().to_list()}
)
user_info.to_parquet(f"./{dataset_version}/user_info.parquet")
print("user_info:", user_info.head())

train_seq_df = (
    hist_df.filter(pl.col("hist_time") < datetime.strptime("2023-05-18 07:00:00", "%Y-%m-%d %H:%M:%S"))
    .groupby("user_index", maintain_order=True).agg(pl.col('hist_id')) #"%Y-%m-%dT%H:%M:%S.%f"
    .with_columns(pl.col("hist_id").list.lengths().alias("seq_len"))
)
valid_seq_df = (
    hist_df.filter(pl.col("hist_time") < datetime.strptime("2023-05-25 07:00:00", "%Y-%m-%d %H:%M:%S"))
    .groupby("user_index", maintain_order=True).agg(pl.col('hist_id'))
    .with_columns(pl.col("hist_id").list.lengths().alias("seq_len"))
)
test_seq_df = user_seq_ids.with_columns(pl.col("hist_id").list.lengths().alias("seq_len"))

def join_data(sample_df, split="test"):
    if split == "test" and dataset == "large":
        sample_df = (
            sample_df.rename({"article_ids_inview": "article_id"})
            .explode('article_id')
            .with_columns(
                pl.lit(None).alias("trigger_id"),
                pl.lit(0).alias("click")
            )
            .collect()
        )
    else:
        sample_df = (
            sample_df.rename({"article_id": "trigger_id"})
            .rename({"article_ids_inview": "article_id"})
            .explode('article_id')
            .with_columns(pl.col("article_id").is_in(pl.col("article_ids_clicked")).cast(pl.Int8).alias("click"))
            .drop(["article_ids_clicked"])
            .collect()
        )
    sample_df = (
        sample_df.with_columns(
            pl.col("article_id").apply(lambda x: article_id2index.get(x)).alias("item_index"),
            pl.col("user_id").apply(lambda x: user_id2index.get(x)).alias("user_index")
        )
        .drop_nulls(subset="item_index")
        .drop_nulls(subset="user_index")
        .join(news.select(["published_time", "item_index"]), on='item_index', how="left")
        .with_columns(
            publish_days=(pl.col('impression_time') - pl.col('published_time')).dt.days().cast(pl.Int32),
            publish_hours=(pl.col('impression_time') - pl.col('published_time')).dt.hours().cast(pl.Int32),
            impression_hour=pl.col('impression_time').dt.hour().cast(pl.Int32),
            impression_weekday=pl.col('impression_time').dt.weekday().cast(pl.Int32)
        )
        .with_columns(
            pl.col("publish_days").clip_max(3).alias("pulish_3day"),
            pl.col("publish_days").clip_max(7).alias("pulish_7day"),
            pl.col("publish_days").clip_max(30),
            pl.col("publish_hours").clip_max(24)
        )
    )
    return sample_df

train_df = pl.scan_parquet(os.path.join(train_path, "behaviors.parquet"))
valid_df = pl.scan_parquet(os.path.join(dev_path, "behaviors.parquet"))
test_df = pl.scan_parquet(os.path.join(test_path, "behaviors.parquet"))
train_df = join_data(train_df, split="train")
valid_df = join_data(valid_df, split="valid")
test_df = join_data(test_df, split="test")
train_df = train_df.join(train_seq_df.select(["user_index", "seq_len"]), on='user_index', how="left")
valid_df = valid_df.join(valid_seq_df.select(["user_index", "seq_len"]), on='user_index', how="left")
test_df = test_df.join(test_seq_df.select(["user_index", "seq_len"]), on='user_index', how="left")

data_columns = ["article_id", "user_id", "trigger_id", "device_type", "is_sso_user", "gender", "postcode",
                "age", "is_subscriber", "impression_hour", "impression_weekday", "pulish_3day", 
                "pulish_7day", "publish_days", "publish_hours"]
for column in data_columns:
    vocab2index = onehot_encoding(train_df, column)
    oov_id = len(vocab2index) + 1
    train_df = train_df.with_columns(
        pl.col(column).apply(lambda x: vocab2index.get(x, oov_id), return_dtype=pl.Int32)
        .fill_null(oov_id)
    )
    valid_df = valid_df.with_columns(
        pl.col(column).apply(lambda x: vocab2index.get(x, oov_id), return_dtype=pl.Int32)
        .fill_null(oov_id)
    )
    test_df = test_df.with_columns(
        pl.col(column).apply(lambda x: vocab2index.get(x, oov_id), return_dtype=pl.Int32)
        .fill_null(oov_id)
    )
    if column == "article_id":
        news = news.with_columns(pl.col("article_id").apply(lambda x: vocab2index.get(x, oov_id)))

item_info = pd.DataFrame()
for column in ["item_index", "article_id"] + onehot_columns + numeric_columns:
    item_info[column] = [0] + news.select(column).to_series().to_list()
item_info["image_id"] = item_info["item_index"]
item_info["text_id"] = item_info["item_index"]
feature_vocab_size["image_id"] = feature_vocab_size["item_index"]
feature_vocab_size["text_id"] = feature_vocab_size["item_index"]
for column in multihot_columns:
    item_info[column] = [[0] * feature_max_len[column]] + news.select(column).to_series().to_list()
item_info.to_parquet(f"./{dataset_version}/item_info.parquet")
print("item_info:\n", item_info.head())

def load_file(file_path):
    with open(file_path, 'rb') as fin:
        return pickle.load(fin)

print("Saving data files...")
data_columns.remove("article_id")
selected_columns = ["click", "user_index", "impression_id", "item_index", "seq_len"] + data_columns
train_df = train_df.select(selected_columns)
print("train_df", train_df.head())
train_samples = train_df.shape[0]
train_df.write_parquet(f"./{dataset_version}/train.parquet")
del train_df
gc.collect()

valid_df = valid_df.select(selected_columns)
print("valid_df", valid_df.head())
valid_samples = valid_df.shape[0]
valid_df.write_parquet(f"./{dataset_version}/valid.parquet")
del valid_df
gc.collect()

test_df = test_df.select(selected_columns)
print("test_df", test_df.head())
test_samples = test_df.shape[0]
test_df.write_parquet(f"./{dataset_version}/test.parquet")
del test_df
gc.collect()

sample_size_dict = {"total": train_samples + valid_samples + test_samples, 
                    "train": train_samples, "valid": valid_samples, "test": test_samples}
with open(f"./{dataset_version}/data_size.json", "w") as fd:
    fd.write(json.dumps({"sample_size": sample_size_dict,
                         "vocab_size": feature_vocab_size,
                         "numeric_features": numeric_columns,
                         "label": ["click"],
                         "max_len": feature_max_len}, indent=4))
print("sample size:", sample_size_dict)
print("vocab size:", feature_vocab_size)

print("Preprocess pretrained embeddings...")
image_emb_df = pl.read_parquet(image_emb_path)
pca = PCA(n_components=64)
image_emb = pca.fit_transform(np.array(image_emb_df["image_embedding"].to_list()))
print("image_embedding.shape", image_emb.shape)
image_emb_df = image_emb_df.with_columns(
    pl.col("article_id").apply(lambda x: article_id2index.get(x, 0)).alias("key"),
    pl.Series(image_emb).alias("value")
)
print("Save image_emb_dim64.parquet...")
image_emb_df.select(["key", "value"]).write_parquet(f"./{dataset_version}/image_emb_dim64.parquet")

contrast_emb_df = pl.read_parquet(contrast_emb_path)
contrast_emb = pca.fit_transform(np.array(contrast_emb_df["contrastive_vector"].to_list()))
print("contrast_emb.shape", contrast_emb.shape)
contrast_emb_df = contrast_emb_df.with_columns(
    pl.col("article_id").apply(lambda x: article_id2index.get(x, 0)).alias("key"),
    pl.Series(contrast_emb).alias("value")
)
print("Save contrast_emb_dim64.parquet...")
contrast_emb_df.select(["key", "value"]).write_parquet(f"./{dataset_version}/contrast_emb_dim64.parquet")

has_columns = selected_columns + onehot_columns + numeric_columns

def load_file(file_path):
    with open(file_path, 'rb') as fin:
        return pickle.load(fin)

def merge_data(df, scaler=None, split='train'):
    pkl_path = f'../preprocessed/large/{split}.pkl'
    data = load_file(pkl_path)
    important_features = pd.read_csv(f'../preprocessed/feature_importance.csv')
    feats = important_features.iloc[:100]['features'].tolist()
    feats = [f for f in feats if f not in has_columns and not f.endswith('_score')]
    data = data[feats].fillna(0)
    if scaler is None:
        scaler = MinMaxScaler(clip=True)
        scaler.fit(data)
    data = pd.DataFrame(scaler.transform(data), columns=data.columns, index=data.index)
    data = pl.from_pandas(data)
    df = pl.concat([df, data], how="horizontal")
    for col in feats:
        print("            - {" + "name: {}, active: True, dtype: float, type: numeric, embedding_dim: 1".format(col) + "}")
    return df, scaler


train_df = pl.read_parquet(f"./{dataset_version}/train.parquet")
train_df, scaler = merge_data(train_df, split='train')
train_df.write_parquet(f"./{dataset_version}/train.parquet")
del train_df
gc.collect()

valid_df = pl.read_parquet(f"./{dataset_version}/valid.parquet")
valid_df, _ = merge_data(valid_df, scaler=scaler, split='valid')
valid_df.write_parquet(f"./{dataset_version}/valid.parquet")
del valid_df
gc.collect()

test_df = pl.read_parquet(f"./{dataset_version}/test.parquet")
test_df, _ = merge_data(test_df, scaler=scaler, split='test')
print("test_df", test_df.head())
test_samples = test_df.shape[0]
test_df.write_parquet(f"./{dataset_version}/test.parquet")
del test_df
gc.collect()

print("fuxictr data generete done.")
