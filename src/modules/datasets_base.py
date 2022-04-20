"""
@Desc:
@Reference:
- transformers examples for using BART model
https://github.com/huggingface/transformers/blob/master/examples/pytorch/summarization/README.md
https://discuss.huggingface.co/t/train-bart-for-conditional-generation-e-g-summarization/1904/2
@Notes:

"""
from typing import List
from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.utils.string_utils import rm_extra_spaces

class BaseDataset(Dataset):
    src_suffix = "source.txt"
    tgt_suffix = "target.txt"

    def __init__(
            self,
            tokenizer,
            data_dir,
            max_source_length,
            max_target_length,
            src_file_prefix="train",
            tgt_file_prefix="train",
            device=None
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.src_file_prefix = src_file_prefix
        self.tgt_file_prefix = tgt_file_prefix
        self.src_file = Path(data_dir).joinpath(f"{self.src_file_prefix}.{self.src_suffix}")
        self.tgt_file = Path(data_dir).joinpath(f"{self.tgt_file_prefix}.{self.tgt_suffix}")
        self.src_data: List[str] = None
        self.tgt_data: List[str] = None
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

        self.pad_token_id = self.tokenizer.pad_token_id
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._read_data()

    def _read_clean_lines(self, file_path):
        data = []
        with open(file_path, "r", encoding="utf-8") as fr:
            for line in fr:
                line = rm_extra_spaces(line)
                if len(line) > 0:
                    data.append(line)
        return data

    def _read_data(self):
        self.src_data = self._read_clean_lines(self.src_file)
        self.tgt_data = self._read_clean_lines(self.tgt_file)
        if len(self.src_data) != len(self.tgt_data):
            raise ValueError(f"data size of src_data {len(self.src_data)} should be equal to "
                             f"tgt_data {len(self.tgt_data)}")

    def __getitem__(self, item):
        raise NotImplementedError("You must implement this")

    def collate_fn(self, batch):
        raise NotImplementedError("You must implement this")
