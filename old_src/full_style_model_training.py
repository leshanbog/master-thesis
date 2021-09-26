#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
import json
import random
import tqdm
import os
import torch
import wandb


# In[ ]:


from _jsonnet import evaluate_file as jsonnet_evaluate_file
from transformers import BertTokenizer, EncoderDecoderModel, Trainer, TrainingArguments, logging

from readers import tg_reader, lenta_reader, ria_reader
from custom_datasets import FullStyleDataset
from utils.training_utils import get_separate_lr_optimizer, init_wandb


# In[ ]:


logging.set_verbosity_info()
config = json.loads(jsonnet_evaluate_file('/home/aobuhtijarov/master-thesis/configs/gen_title.jsonnet'))
init_wandb('full-style', config)

agency_list = ["РИА Новости", "lenta.ru"]
print('Agency list:', agency_list)


# In[ ]:


tokenizer_model_path = config["tokenizer_model_path"]
tokenizer = BertTokenizer.from_pretrained(tokenizer_model_path, do_lower_case=False, do_basic_tokenize=False)

max_tokens_text = config["max_tokens_text"]
max_tokens_title = config["max_tokens_title"]

print("Initializing model...")


# In[ ]:


enc_model_path = config["enc_model_path"]
dec_model_path = config["dec_model_path"]
model = EncoderDecoderModel.from_encoder_decoder_pretrained(enc_model_path, dec_model_path)


# In[ ]:


agency_to_special_token_id = {a: tokenizer.vocab[f'[unused{i+1}]'] for i, a in enumerate(agency_list)}


# In[ ]:


train_dataset = FullStyleDataset('/home/aobuhtijarov/datasets/full_lenta_ria.train.jsonl',
                                 tokenizer, agency_to_special_token_id, slice(0, 72500),
                                 max_tokens_text=max_tokens_text, max_tokens_title=max_tokens_title)

val_dataset = FullStyleDataset('/home/aobuhtijarov/datasets/full_lenta_ria.train.jsonl',
                                 tokenizer, agency_to_special_token_id, slice(72500, 75362),
                                 max_tokens_text=max_tokens_text, max_tokens_title=max_tokens_title)


# In[ ]:


wandb.summary.update({
    'Train dataset size': len(train_dataset),
    'Test dataset size': len(val_dataset),
})


# In[ ]:


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


# In[ ]:


wandb.config.output_model_path = '/home/aobuhtijarov/models/full_style_model/'


# In[ ]:


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


# In[ ]:


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

