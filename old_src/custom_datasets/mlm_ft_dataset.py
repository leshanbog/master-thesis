import torch
import random
from torch.utils.data import Dataset

class MLMFTDataset(Dataset):
    def __init__(
        self,
        docs,
        tokenizer,
        max_tokens_text=250,
        filter_date=None,
    ):        
        self.docs = docs 

        self.tokenizer = tokenizer
        self.max_tokens_text = max_tokens_text

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, index):
        text = self.docs[index]
 
        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_tokens_text,
            padding="max_length",
            truncation=True
        )

        return {
            "input_ids": torch.tensor(inputs["input_ids"]),
            "attention_mask": torch.tensor(inputs["attention_mask"]),
        }
