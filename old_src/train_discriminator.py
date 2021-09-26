import argparse
import json
import random
import tqdm
import torch
import wandb
import numpy as np
import os

from _jsonnet import evaluate_file as jsonnet_evaluate_file
from transformers import BertTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, logging

from readers import tg_reader, ria_reader, lenta_reader
from custom_datasets import AgencyTitleDatasetClassification, LentaRiaDatasetClassification
from utils.training_utils import get_separate_lr_optimizer, init_wandb


def reader(path):
    with open(path, 'r') as f:
        for line in f:
            yield json.loads(line.strip())


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        'accuracy': np.mean(labels == preds)
    }

def train_discriminator(
    run_name: str,
    model_path: str,
    config_file: str,
    train_file: str,
    train_fraq: float,
    dataset_type: str,
    output_model_path: str,
):
    logging.set_verbosity_info()
    config = json.loads(jsonnet_evaluate_file(config_file))
    init_wandb(run_name, config)
    
    agency_list = config['agency_list']
    print('Agency list:', agency_list)

    max_tokens_text = config["max_tokens_text"]
    max_tokens_title = config["max_tokens_title"]

    tokenizer_model_path = config["tokenizer_model_path"]
    tokenizer = BertTokenizer.from_pretrained(tokenizer_model_path, do_lower_case=False, do_basic_tokenize=False)

    print("Fetching data...")
    if dataset_type == 'tg':
        all_records = [r for r in tqdm.tqdm(tg_reader(train_file, agency_list))]
        full_dataset = AgencyTitleDatasetClassification(
            all_records,
            tokenizer,
            agency_list,
            max_tokens_text=max_tokens_text,
            max_tokens_title=max_tokens_title
        )       
    elif dataset_type == 'lenta-ria':
        lenta_records = [r for r in tqdm.tqdm(lenta_reader(os.path.join(train_file, 'lenta/lenta-ru-news.train.csv')))]
        lenta_records.extend(
            [r for r in tqdm.tqdm(lenta_reader(os.path.join(train_file, 'lenta/lenta-ru-news.val.csv')))]
        )

        ria_records = [r for r in tqdm.tqdm(ria_reader(os.path.join(train_file, 'ria/ria.shuffled.train.json')))]
        ria_records.extend(
            [r for r in tqdm.tqdm(ria_reader(os.path.join(train_file, 'ria/ria.shuffled.val.json')))]
        )

        records = [r for r in reader('/home/aobuhtijarov/datasets/full_lenta_ria.test.jsonl')]

        filter_lenta = [
            {'text': r['lenta_text'], 'title': r['lenta_title'], 'agency': 'lenta.ru', 'date': r['lenta_date']} 
            for r in records
        ]

        filter_ria = [
            {'text': r['ria_text'], 'title': r['ria_title'], 'agency': 'РИА Новости', 'date': r['lenta_date']} 
            for r in records
        ]

        lenta_filter_titles = set(x['title'] for x in filter_lenta)
        ria_filter_titles = set(x['title'] for x in filter_ria)
        lenta_records = [r for r in lenta_records if r['title'] not in lenta_filter_titles]
        ria_records = [r for r in ria_records if r['title'] not in ria_filter_titles]
        
        random.shuffle(ria_records)
        lenta_records = [r for r in lenta_records if r['date'][:4] in ['2010', '2011', '2012', '2013', '2014']]

        all_records = lenta_records + ria_records[:len(lenta_records)]

        random.shuffle(all_records)
        full_dataset = AgencyTitleDatasetClassification(
            all_records,
            tokenizer,
            agency_list,
            max_tokens_text=max_tokens_text,
            max_tokens_title=max_tokens_title
        )        
    elif dataset_type == 'lenta-ria-clusters':
        full_dataset = LentaRiaDatasetClassification(train_file, tokenizer, agency_list,
            max_tokens_text, max_tokens_title)

    print("Building datasets...")
    
    train_size = int(train_fraq * len(full_dataset))
    test_size = int((1-train_fraq) * 0.5 * len(full_dataset))
    
    train_dataset, test_dataset, eval_dataset = \
        torch.utils.data.random_split(full_dataset, [train_size, test_size, len(full_dataset) - train_size - test_size])
    
    wandb.summary.update({
        'Train dataset size': len(train_dataset),
        'Val dataset size': len(eval_dataset),
        'Test dataset size': len(test_dataset),
    })

    print("Initializing model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, 
        num_labels=len(agency_list)
    )

    print("Training model...")
    batch_size = config["batch_size"]
    logging_steps = config["logging_steps"]
    save_steps = config["save_steps"]
    eval_steps = config["eval_steps"]
    warmup_steps = config["num_warmup_steps"]
    gradient_accumulation_steps = config["gradient_accumulation_steps"]
    max_steps = config["max_steps"]
    lr = config["learning_rate"]

    training_args = TrainingArguments(
        output_dir=output_model_path,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        evaluation_strategy='steps',
        learning_rate=lr,
        warmup_steps=warmup_steps,
        overwrite_output_dir=False,
        logging_steps=logging_steps,
        eval_steps=eval_steps,
        save_steps=save_steps,
        max_steps=max_steps,
        save_total_limit=1,
        weight_decay=0.01,
        report_to='wandb',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    
    wandb.summary.update({
        'Test Evaluation': trainer.evaluate(eval_dataset=test_dataset)
    })
    model.save_pretrained(output_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("--train-file", type=str, required=True)
    parser.add_argument("--dataset-type", type=str, required=True, choices=['tg', 'lenta-ria', 'lenta-ria-clusters'])
    parser.add_argument("--train-fraq", type=float, default=0.91)
    parser.add_argument("--output-model-path", type=str, required=True)

    args = parser.parse_args()
    train_discriminator(**vars(args))
