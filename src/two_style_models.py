#!/usr/bin/env python
# coding: utf-8

# In[1]:


from utils.training_utils import get_separate_lr_optimizer, init_wandb

from transformers import BertTokenizer, EncoderDecoderModel, Trainer, TrainingArguments, logging
import json
import random
import tqdm
import torch

import os
import wandb

from _jsonnet import evaluate_file as jsonnet_evaluate_file


# In[2]:


def reader(path):
    with open(path, 'r') as f:
        for line in f:
            yield json.loads(line.strip())


# In[3]:


logging.set_verbosity_info()
config = json.loads(jsonnet_evaluate_file('/home/aobuhtijarov/master-thesis/configs/gen_title.jsonnet'))

init_wandb('ria-style-model', config)


# In[4]:


tokenizer_model_path = config["tokenizer_model_path"]
tokenizer = BertTokenizer.from_pretrained(tokenizer_model_path, do_lower_case=False, do_basic_tokenize=False)

max_tokens_text = config["max_tokens_text"]
max_tokens_title = config["max_tokens_title"]


# In[5]:


enc_model_path = config["enc_model_path"]
dec_model_path = config["dec_model_path"]
model = EncoderDecoderModel.from_encoder_decoder_pretrained(enc_model_path, dec_model_path)


# In[8]:


from torch.utils.data import Dataset

class StyleModelDataset(Dataset):
    def __init__(
        self,
        path,
        tokenizer,
        agency, # lenta or ria
        is_train=True,
        max_tokens_text=250,
        max_tokens_title=50
    ):
        with open(path, 'r') as f:
            self.records = [json.loads(x.strip()) for x in f.readlines()]

        if is_train:
            self.records = self.records[:74000]
        else:
            self.records = self.records[74000:]
            
        self.agency = agency
        self.tokenizer = tokenizer
        self.max_tokens_text = max_tokens_text
        self.max_tokens_title = max_tokens_title

    def __len__(self):
        return len(self.records) * 2

    def __getitem__(self, index):
        record = self.records[index // 2]
        if index % 2 == 1:
            text = record['ria_text']
        else:
            text = record['lenta_text']
            
        if self.agency == 'ria':
            title = record['ria_title']
        elif self.agency == 'lenta':
            title = record['lenta_title']

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
        else:
            text = record['lenta_text']
            
        if self.agency == 'ria':
            title = record['ria_title']
        elif self.agency == 'lenta':
            title = record['lenta_title']            

        return {
            'text': text,
            'title': title,
            'date': record['lenta_date'],
        }


# In[9]:


train_dataset = StyleModelDataset('/home/aobuhtijarov/datasets/full_lenta_ria.train.jsonl',
                                tokenizer, 'ria', is_train=True)

val_dataset = StyleModelDataset('/home/aobuhtijarov/datasets/full_lenta_ria.train.jsonl',
                                tokenizer, 'ria', is_train=False)


# In[10]:


len(train_dataset), len(val_dataset)


# In[11]:


wandb.summary.update({
    'Train dataset size': len(train_dataset),
    'Val dataset size': len(val_dataset)
})


# In[12]:


wandb.config.output_model_path = '/home/aobuhtijarov/models/ria_style'


# In[14]:


batch_size = config["batch_size"]
eval_steps = config["eval_steps"]
save_steps = config["save_steps"]
logging_steps = config["logging_steps"]
enc_lr = config["enc_lr"]
dec_lr = config["dec_lr"]
warmup_steps = config["num_warmup_steps"]
max_steps = config["max_steps"]
gradient_accumulation_steps = config["gradient_accumulation_steps"]

opt = get_separate_lr_optimizer(model, enc_lr, dec_lr, warmup_steps, max_steps)

training_args = TrainingArguments(
    output_dir=wandb.config.output_model_path,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    evaluation_strategy='steps',
    do_train=True,
    do_eval=True,
    overwrite_output_dir=False,
    logging_steps=logging_steps,
    save_steps=save_steps,
    eval_steps=eval_steps,
    save_total_limit=1,
    max_steps=max_steps,
    report_to='wandb',
)


# In[15]:


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    optimizers=opt,
)


# In[ ]:


trainer.train()
model.save_pretrained(wandb.config.output_model_path)


# In[ ]:


wandb.finish()
