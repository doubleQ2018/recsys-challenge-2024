# =========================================================================
# Copyright (C) 2024. FuxiCTR Authors. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================


import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from keras_preprocessing.sequence import pad_sequences
import pandas as pd
import polars as pl
import torch
import logging


class ParquetDataset(Dataset):
    def __init__(self, data_path):
        self.column_index = dict()
        self.darray = self.load_data(data_path)
        
    def __getitem__(self, index):
        return self.darray[index, :]
    
    def __len__(self):
        return self.darray.shape[0]

    def load_data(self, data_path):
        df = pl.read_parquet(data_path)
        data_arrays = []
        idx = 0
        for col in df.columns:
            if df[col].dtype != pl.List:
                array = np.array(df[col])
                self.column_index[col] = idx
                idx += 1
            else:
                array = np.array(df[col].to_list())
                seq_len = array.shape[1]
                self.column_index[col] = [i + idx for i in range(seq_len)]
                idx += seq_len
            data_arrays.append(array)
        return np.column_stack(data_arrays)


class LongSequenceDataLoader(DataLoader):
    def __init__(self, feature_map, data_path, batch_size=32, shuffle=False,
                 num_workers=1, max_len=50, **kwargs):
        #if not data_path.endswith(".parquet"):
        #    data_path += ".parquet"
        self.dataset = ParquetDataset(data_path)
        column_index = self.dataset.column_index
        super().__init__(dataset=self.dataset, batch_size=batch_size,
                         shuffle=shuffle, num_workers=num_workers,
                         collate_fn=BatchCollator(feature_map, max_len, column_index))
        self.num_samples = len(self.dataset)
        self.num_blocks = 1
        self.num_batches = int(np.ceil(self.num_samples / self.batch_size))

    def __len__(self):
        return self.num_batches


class BatchCollator(object):
    def __init__(self, feature_map, max_len, column_index):
        self.feature_map = feature_map
        dataset_id = feature_map.dataset_id
        self.user_info = pd.read_parquet(f"{dataset_id}/user_info.parquet").set_index("user_index")
        self.item_info = pd.read_parquet(f"{dataset_id}/item_info.parquet").set_index("item_index")
        self.max_len = max_len
        self.column_index = column_index

    def __call__(self, batch):
        batch_tensor = default_collate(batch)
        all_cols = set(list(self.feature_map.features.keys()) + self.feature_map.labels)
        batch_dict = dict()
        for col, idx in self.column_index.items():
            if col in all_cols:
                batch_dict[col] = batch_tensor[:, idx]
        user_index = batch_dict["user_index"].tolist()
        user_seqs = self.user_info["user_seq_ids"].iloc[user_index].tolist()
        seq_lens = batch_dict["seq_len"].int().tolist()
        batch_seqs = self.padding_seqs(user_seqs, seq_lens)
        mask = (torch.from_numpy(batch_seqs) > 0).float()
        # batch_seqs = np.array(batch_seqs)
        item_index = batch_dict["item_index"].numpy().reshape(-1, 1)
        batch_items = np.hstack([batch_seqs, item_index]).flatten()
        item_info = self.item_info.iloc[batch_items]
        item_dict = dict()
        for col in item_info.columns:
            if col in all_cols:
                item_dict[col] = torch.from_numpy(np.array(item_info[col].to_list()))
        return batch_dict, item_dict, mask
    
    def padding_seqs(self, user_seqs, seq_lens):
        batch_seqs = []
        for seq, l in zip(user_seqs, seq_lens):
            batch_seqs.append(seq[:l])
        max_len = min(max(seq_lens), self.max_len)
        batch_seqs = pad_sequences(batch_seqs, maxlen=max_len,
                                   value=0, padding="pre", truncating="pre")
        return batch_seqs
