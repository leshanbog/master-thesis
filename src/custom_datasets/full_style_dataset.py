import torch
import random
import json
from torch.utils.data import Dataset


def is_ok_date(date, filter_date):
    if filter_date is None:
        return True
    
    return date.startswith(filter_date)


def is_ok_agency(agency, agency_list):
    if agency_list is None:
        return True
    
    return agency in agency_list


def set_target_title(tok_otpt, agency_to_special_token_id, target_agency):
    marker = agency_to_special_token_id[target_agency]

    tok_otpt['input_ids'][2:] = tok_otpt['input_ids'][1:-1]
    tok_otpt['input_ids'][1] = marker

    tok_otpt['attention_mask'][2:] = tok_otpt['attention_mask'][1:-1]
    tok_otpt['attention_mask'][1] = 1

    return tok_otpt


class FullStyleDataset(Dataset):
    def __init__(
        self,
        path,
        tokenizer,
        agency_to_special_token_id,
        cur_slice,
        max_tokens_text=250,
        max_tokens_title=48,
    ):        
        with open(path, 'r') as f:
            self.records = [json.loads(x.strip()) for x in f.readlines()]
            assert len(self.records) == 75362
            self.records = self.records[cur_slice]

        self.tokenizer = tokenizer
        self.max_tokens_text = max_tokens_text
        self.max_tokens_title = max_tokens_title

        self.agency_to_special_token_id = agency_to_special_token_id

    def __len__(self):
        return len(self.records) * 4

    def __getitem__(self, index):
        record = self.records[index // 4]
        
        if index % 4 == 0:
            text = record['ria_text']
            title = record['ria_title']
            target_agency = 'РИА Новости'
        elif index % 4 == 1:
            text = record['ria_text']
            title = record['lenta_title']
            target_agency = 'lenta.ru'
        elif index % 4 == 2:
            text = record['lenta_text']
            title = record['ria_title']
            target_agency = 'РИА Новости'
        else:
            text = record['lenta_text']
            title = record['lenta_title']
            target_agency = 'lenta.ru'

        title_tok = self.tokenizer(
            title,
            add_special_tokens=True,
            max_length=self.max_tokens_title,
            padding="max_length",
            truncation=True
        )

        decoder_input_ids = torch.tensor(title_tok["input_ids"])
        decoder_attention_mask = torch.tensor(title_tok["attention_mask"])
        labels = decoder_input_ids.clone()

        for i, mask in enumerate(decoder_attention_mask):
            if mask == 0:
                labels[i] = -100

        text_tok = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_tokens_text,
            padding="max_length",
            truncation=True
        )

        ### Not adding [SEP] in the end
        text_tok = set_target_title(
            text_tok, self.agency_to_special_token_id, target_agency
        )

        return {
            "input_ids": torch.tensor(text_tok["input_ids"]),
            "attention_mask": torch.tensor(text_tok["attention_mask"]),
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "labels": labels,
        }
