import json
import random

import torch
from torch.utils.data import Dataset


class LentaRiaDataset(Dataset):
    def __init__(
        self,
        path,
        tokenizer,
        max_tokens_text=250,
        max_tokens_title=40
    ):
        with open(path, 'r') as f:
            self.records = [json.loads(x.strip()) for x in f.readlines()]

        self.tokenizer = tokenizer
        self.max_tokens_text = max_tokens_text
        self.max_tokens_title = max_tokens_title

    def __len__(self):
        return len(self.records) * 2

    def __getitem__(self, index):
        record = self.records[index // 2]
        if index % 2 == 1:
            text = record['ria_text']
            title = record['lenta_title']            
        else:
            text = record['lenta_text']
            title = record['ria_title']

        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_tokens_text,
            padding="max_length",
            truncation=True
        )

        outputs = self.tokenizer(
            title,
            add_special_tokens=True,
            max_length=self.max_tokens_title,
            padding="max_length",
            truncation=True
        )

        decoder_input_ids = torch.tensor(outputs["input_ids"])
        decoder_attention_mask = torch.tensor(outputs["attention_mask"])
        labels = decoder_input_ids.clone()

        for i, mask in enumerate(decoder_attention_mask):
            if mask == 0:
                labels[i] = -100

        return {
            "input_ids": torch.tensor(inputs["input_ids"]),
            "attention_mask": torch.tensor(inputs["attention_mask"]),
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "labels": labels
        }

    def get_strings(self, index):
        record = self.records[index // 2]
        if index % 2 == 1:
            text = record['ria_text']
            title = record['lenta_title']            
        else:
            text = record['lenta_text']
            title = record['ria_title']
        return {
            'text': text,
            'title': title,
            'date': record['lenta_date'],
        }


class LentaRiaDatasetClassification(Dataset):
    def __init__(
        self,
        path,
        tokenizer,
        agency_list,
        max_tokens_text=250,
        max_tokens_title=40
    ):
        with open(path, 'r') as f:
            self.records = [json.loads(x.strip()) for x in f.readlines()]

        self.agency_to_target = {a: i for i, a in enumerate(sorted(agency_list))}
        self.target_to_agency = {i: a for a, i in self.agency_to_target.items()}

        self.tokenizer = tokenizer
        self.max_tokens_text = max_tokens_text
        self.max_tokens_title = max_tokens_title

    def __len__(self):
        return len(self.records) * 2

    def __getitem__(self, index):
        record = self.records[index // 2]
        if index % 2 == 1:
            title = record['lenta_title']    
            target = self.agency_to_target['lenta.ru']
        else:
            title = record['ria_title']
            target = self.agency_to_target['РИА Новости']

        inputs = self.tokenizer(
            title,
            add_special_tokens=True,
            max_length=self.max_tokens_title,
            padding="max_length",
            truncation=True
        )

        return {
            "input_ids": torch.tensor(inputs["input_ids"]),
            "attention_mask": torch.tensor(inputs["attention_mask"]),
            "labels": target,
        }
