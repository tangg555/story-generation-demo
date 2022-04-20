"""
@Desc:
@Reference:
- transformers examples for using BART model
https://github.com/huggingface/transformers/tree/master/examples/pytorch/summarization
https://discuss.huggingface.co/t/train-bart-for-conditional-generation-e-g-summarization/1904/2
- add_special_tokens
https://huggingface.co/docs/transformers/v4.17.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase
- linecache
https://blog.csdn.net/my2010Sam/article/details/38022041
- torch Dataset
https://pytorch.org/docs/stable/data.html?highlight=dataset#torch.utils.data.Dataset
@Notes:
- add_special_tokens
special_tokens_dict (dictionary str to str or tokenizers.AddedToken) —
Keys should be in the list of predefined special attributes:
[bos_token, eos_token, unk_token, sep_token, pad_token, cls_token, mask_token, additional_special_tokens].
Tokens are only added if they are not already in the vocabulary (tested by checking
if the tokenizer assign the index of the unk_token to them).
- collate_fn
A custom collate_fn can be used to customize collation, e.g., padding sequential data to max length of a batch.
See this section on more about collate_fn.
"""

from src.modules.datasets_base import BaseDataset


class LeadingContextDataset(BaseDataset):
    def __init__(self, tokenizer,
                 data_dir,
                 max_source_length,
                 max_target_length,
                 src_file_prefix="train",
                 tgt_file_prefix="train", ):
        super().__init__(tokenizer, data_dir, max_source_length, max_target_length,
                         src_file_prefix, tgt_file_prefix)

    def __getitem__(self, index):
        source_line = self.src_data[index]
        target_line = self.tgt_data[index]
        assert source_line, f"empty source line for index {index}"
        assert target_line, f"empty tgt line for index {index}"
        return {"src_text": source_line, "tgt_text": target_line, "data_id": index}

    def collate_fn(self, batch):
        batch_encoding = self.tokenizer(
            [x["src_text"] for x in batch],
            add_special_tokens=True,
            truncation=True,
            padding="longest",
            max_length=self.max_source_length,
            return_tensors="pt",
        ).data
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                [x["tgt_text"] for x in batch],
                add_special_tokens=True,
                truncation=True,
                padding="longest",
                max_length=self.max_target_length,
                return_tensors="pt",
            ).data
        batch_encoding["labels"] = labels["input_ids"]
        batch_encoding["ids"] = [x["data_id"] for x in batch]
        return batch_encoding

    def __len__(self):
        return len(self.src_data)
