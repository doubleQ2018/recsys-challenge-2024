# =========================================================================
# Copyright (C) 2023-2024. FuxiCTR Authors. All rights reserved.
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
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
from itertools import chain
from torch.utils.data.dataloader import default_collate
from torch.utils.data import IterDataPipe, DataLoader, get_worker_info
import glob
import os


class NpzIterDataPipe(IterDataPipe):
    def __init__(self, data_blocks, feature_map):
        self.feature_map = feature_map
        self.data_blocks = data_blocks
        
    def load_data(self, data_path):
        data_dict = np.load(data_path)
        all_cols = list(self.feature_map.features.keys()) + self.feature_map.labels
        data_arrays = [data_dict[col] for col in all_cols]
        return np.column_stack(data_arrays)

    def read_block(self, data_block):
        darray = self.load_data(data_block)
        for idx in range(darray.shape[0]):
            yield darray[idx, :]

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None: # single-process data loading
            block_list = self.data_blocks
        else: # in a worker process
            block_list = [
                block
                for idx, block in enumerate(self.data_blocks)
                if idx % worker_info.num_workers == worker_info.id
            ]
        return chain.from_iterable(map(self.read_block, block_list))


class NpzBlockDataLoader(DataLoader):
    def __init__(self, feature_map, data_path, split="train", batch_size=32, shuffle=False,
                 num_workers=1, buffer_size=100000, **kwargs):
        if not data_path.endswith("npz"):
            data_path = os.path.join(data_path, "*.npz")
        data_blocks = sorted(glob.glob(data_path)) # sort by part name
        assert len(data_blocks) > 0, f"invalid data_path: {data_path}"
        self.data_blocks = data_blocks
        self.num_blocks = len(self.data_blocks)
        self.feature_map = feature_map
        self.batch_size = batch_size
        self.num_batches, self.num_samples = self.count_batches_and_samples()
        datapipe = NpzIterDataPipe(self.data_blocks, feature_map)
        if shuffle:
            datapipe = datapipe.shuffle(buffer_size=buffer_size)
        elif split == "test":
            num_workers = 1 # multiple workers cannot keep the order of data reading
        super().__init__(dataset=datapipe,
                         batch_size=batch_size,
                         num_workers=num_workers,
                         collate_fn=BatchCollator(feature_map))

    def __len__(self):
        return self.num_batches

    def count_batches_and_samples(self):
        num_samples = 0
        for block_path in self.data_blocks:
            block_size = np.load(block_path)[self.feature_map.labels[0]].shape[0]
            num_samples += block_size
        num_batches = int(np.ceil(num_samples / self.batch_size))
        return num_batches, num_samples


class BatchCollator(object):
    def __init__(self, feature_map):
        self.feature_map = feature_map

    def __call__(self, batch):
        batch_tensor = default_collate(batch)
        all_cols = list(self.feature_map.features.keys()) + self.feature_map.labels
        batch_dict = dict()
        for col in all_cols:
            batch_dict[col] = batch_tensor[:, self.feature_map.get_column_index(col)]
        return batch_dict
