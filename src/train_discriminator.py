import argparse
import json
import random
import tqdm
import torch
import numpy as np

from _jsonnet import evaluate_file as jsonnet_evaluate_file
from transformers import BertTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, logging

from readers.tg_reader import tg_reader
from datasets.agency_title_dataset import AgencyTitleDataset
from utils.training_utils import init_wandb


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return {
        'accuracy': np.mean(np.argmax(predictions, axis=1) == labels)
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
    
    agency_list = config.pop('agency_list', ['ТАСС', 'РТ на русском'])
    print('Agency list:', agency_list)

    print("Fetching data...")
    all_records = [r for r in tqdm.tqdm(tg_reader(train_file, agency_list)) if random.random() <= train_sample_rate]

    print("Building datasets...")
    tokenizer_model_path = config.pop("tokenizer_model_path")
    tokenizer = BertTokenizer.from_pretrained(tokenizer_model_path, do_lower_case=False, do_basic_tokenize=False)

    max_tokens_text = config.pop("max_tokens_text", 196)
    max_tokens_title = config.pop("max_tokens_title", 48)

    full_dataset = AgencyTitleDataset(
        all_records,
        tokenizer,
        agency_list,
        max_tokens_text=max_tokens_text,
        max_tokens_title=max_tokens_title
    )
    
    train_size = int(train_fraq * len(full_dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, len(full_dataset) - train_size])

    print("Initializing model...")
    model = AutoModelForSequenceClassification.from_pretrained(tokenizer_model_path, 
                                                               num_labels=len(agency_list))

    print("Model: ")
    print(model)

    print("Training model...")
    batch_size = config.pop("batch_size", 8)
    eval_steps = config.pop("eval_steps", 100)
    save_steps = config.pop("save_steps", 100)
    logging_steps = config.pop("logging_steps", 25)
    learning_rate = config.pop("learning_rate", 5e-05)
    warmup_steps = config.pop("num_warmup_steps", 150)
    num_train_epochs = config.pop("num_train_epochs", 5)
    gradient_accumulation_steps = config.pop("gradient_accumulation_steps", 25)

    training_args = TrainingArguments(
       report_to='wandb',
    )

    trainer = Trainer(
    
    )

    trainer.train()
    print(trainer.evaluate())
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
