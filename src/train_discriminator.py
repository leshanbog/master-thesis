import argparse
import json
import random
import tqdm
import torch
import numpy as np

from _jsonnet import evaluate_file as jsonnet_evaluate_file
from transformers import BertTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, logging

from readers.tg_reader import tg_reader
from custom_datasets.agency_title_dataset import AgencyTitleDatasetClassification
from utils.training_utils import get_separate_lr_optimizer, init_wandb


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        'accuracy': np.mean(labels == preds)
    }


def train_discriminator(
    run_name: str,
    config_file: str,
    train_file: str,
    train_fraq: float,
    train_sample_rate: float,
    output_model_path: str,
):
    logging.set_verbosity_info()
    config = json.loads(jsonnet_evaluate_file(config_file))
    init_wandb(run_name, config)
    
    agency_list = config['agency_list']
    print('Agency list:', agency_list)

    print("Fetching data...")
    all_records = [r for r in tqdm.tqdm(tg_reader(train_file, agency_list)) if random.random() <= train_sample_rate]

    print("Building datasets...")
    tokenizer_model_path = config["tokenizer_model_path"]
    tokenizer = BertTokenizer.from_pretrained(tokenizer_model_path, do_lower_case=False, do_basic_tokenize=False)

    max_tokens_text = config["max_tokens_text"]
    max_tokens_title = config["max_tokens_title"]

    full_dataset = AgencyTitleDatasetClassification(
        all_records,
        tokenizer,
        agency_list,
        max_tokens_text=max_tokens_text,
        max_tokens_title=max_tokens_title
    )
    
    train_size = int(train_fraq * len(full_dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, len(full_dataset) - train_size])
    
    print("Initializing model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        tokenizer_model_path, 
        num_labels=len(agency_list)
    )

    print("Training model...")
    batch_size = config["batch_size"]
    logging_steps = config["logging_steps"]
    learning_rate = config["learning_rate"]
    warmup_steps = config["num_warmup_steps"]
    num_train_epochs = config["num_train_epochs"]
    gradient_accumulation_steps = config["gradient_accumulation_steps"]
    lr = config["learning_rate"]

    training_args = TrainingArguments(
        output_dir=output_model_path,
        do_train=True,
        do_eval=False,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=lr,
        warmup_steps=warmup_steps,
        overwrite_output_dir=False,
        logging_steps=logging_steps,
        num_train_epochs=num_train_epochs,
        report_to='wandb',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    res = trainer.evaluate(eval_dataset=val_dataset)
    model.save_pretrained(output_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("--train-file", type=str, required=True)
    parser.add_argument("--train-sample-rate", type=float, default=1.0)
    parser.add_argument("--train-fraq", type=float, default=0.8)
    parser.add_argument("--output-model-path", type=str, required=True)

    args = parser.parse_args()
    train_discriminator(**vars(args))
