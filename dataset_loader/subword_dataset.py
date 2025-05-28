import csv

import numpy as np

import torch
from torch.utils.data import Dataset


"""
Load SubWord Dataset.
"""
class SubWord_Dataset(Dataset):
    def __init__(
            self,
            csv_fpath,
            start_token,
            padding_token,
            context_window):
        self.start_token = start_token
        self.padding_token = padding_token

        self.context_window = context_window

        with open(csv_fpath, "r") as f:
            csv_reader = csv.reader(f)
            self.fpaths_list = list(csv_reader)

    def __len__(self):
        return len(self.fpaths_list)

    def __getitem__(self, index):
        np_fpath = self.fpaths_list[index][0]

        # Used in old data format.
        """
        loaded_data = np.load(np_fpath)
        numpy_data = loaded_data["data"]
        """
        numpy_data = np.load(np_fpath)

        numpy_start_token = np.array([self.start_token])

        input_data = np.concatenate([numpy_start_token, numpy_data[:-1]])
        target_data = numpy_data

        in_pad_tokens = [self.padding_token] * (self.context_window - len(input_data))
        numpy_in_pad_tokens = np.array(in_pad_tokens)
        input_data = np.concatenate([input_data, numpy_in_pad_tokens])

        target_pad_tokens = [self.padding_token - 1] * (self.context_window - len(target_data))
        numpy_target_pad_tokens = np.array(target_pad_tokens)
        target_data = np.concatenate([target_data, numpy_target_pad_tokens])

        input_tensor = torch.from_numpy(input_data).long()
        target_tensor = torch.from_numpy(target_data).long()

        return input_tensor, target_tensor
