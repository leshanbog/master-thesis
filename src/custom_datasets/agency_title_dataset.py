import torch
from torch.utils.data import Dataset


def is_ok_date(date, filter_date):
    if filter_date is None:
        return True
    
    return date.startswith(filter_date)


def is_ok_agency(agency, agency_list):
    if agency_list is None:
        return True
    
    return agency in agency_list

class AgencyTitleDataset(Dataset):
    def __init__(
        self,
        records,
        tokenizer,
        agency_list,
        agency_to_special_token_id=None,
        do_prepend_marker=False,
        max_tokens_text=196,
        max_tokens_title=40,
        filter_date=None
    ):        
        self.records = [
            r for r in records if is_ok_agency(r['agency'], agency_list) and is_ok_date(r['date'], filter_date)
        ]
        
        self.tokenizer = tokenizer
        self.max_tokens_text = max_tokens_text
        self.max_tokens_title = max_tokens_title

        self.agency_to_special_token_id = agency_to_special_token_id
        
        if agency_list:
            self.agency_to_target = {a: i for i, a in enumerate(sorted(agency_list))}
            self.target_to_agency = {i: a for a, i in self.agency_to_target.items()}

        self.do_prepend_marker = do_prepend_marker # True if dataset is used for title_gen, False - for classification

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        title = record.get("title", "")
        text = record["text"]
        target = self.agency_to_target[record["agency"]]

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

        if self.do_prepend_marker:
            text_tok = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=self.max_tokens_text,
                padding="max_length",
                truncation=True
            )

            ### Not adding [SEP] in the end
            text_tok['input_ids'][2:] = text_tok['input_ids'][1:-1]
            text_tok['input_ids'][1] = self.agency_to_special_token_id[record['agency']]

            text_tok['attention_mask'][2:] = text_tok['attention_mask'][1:-1]
            text_tok['attention_mask'][1] = 1

            return {
                "input_ids": torch.tensor(text_tok["input_ids"]),
                "attention_mask": torch.tensor(text_tok["attention_mask"]),
                "decoder_input_ids": decoder_input_ids,
                "decoder_attention_mask": decoder_attention_mask,
                "labels": labels,
            }
        else:
            return {
                "input_ids": decoder_input_ids,
                "attention_mask": decoder_attention_mask,
                "label": target,
            }               

    def get_strings(self, index):
        record = self.records[index]
        return {
            'text': record["text"],
            'title': record.get("title", ""),
            'agency': record.get('agency', '-'),
            'date': record.get('date', '-'),
        }
