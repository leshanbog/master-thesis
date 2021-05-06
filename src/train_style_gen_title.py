import argparse
import json
import random
import tqdm
import os
import torch
import wandb

from _jsonnet import evaluate_file as jsonnet_evaluate_file
from transformers import BertTokenizer, EncoderDecoderModel, Trainer, TrainingArguments, logging

from readers import tg_reader, lenta_reader, ria_reader
from custom_datasets.agency_title_dataset import AgencyTitleDatasetGeneration
from utils.training_utils import get_separate_lr_optimizer, init_wandb


def train_style_gen_title(
        run_name: str,
        config_file: str,
        train_file: str,
        dataset_type: str,
        output_model_path: str,
        from_pretrained: str = None,
        checkpoint: str = None
):
    logging.set_verbosity_info()
    config = json.loads(jsonnet_evaluate_file(config_file))
    init_wandb(run_name, config)

    agency_list = config['agency_list']
    print('Agency list:', agency_list)

    tokenizer_model_path = config["tokenizer_model_path"]
    tokenizer = BertTokenizer.from_pretrained(tokenizer_model_path, do_lower_case=False, do_basic_tokenize=False)

    max_tokens_text = config["max_tokens_text"]
    max_tokens_title = config["max_tokens_title"]

    print("Initializing model...")

    if from_pretrained:
        model = EncoderDecoderModel.from_pretrained(from_pretrained)
    else:
        enc_model_path = config["enc_model_path"]
        dec_model_path = config["dec_model_path"]
        model = EncoderDecoderModel.from_encoder_decoder_pretrained(enc_model_path, dec_model_path)

    print("Fetching data...")
    if dataset_type == 'tg':
        all_records = [r for r in tqdm.tqdm(tg_reader(train_file))]
    elif dataset_type == 'lenta-ria':
        lenta_records = [r for r in tqdm.tqdm(lenta_reader(os.path.join(train_file, 'lenta/lenta-ru-news.train.csv')))]
        lenta_records.extend(
            [r for r in tqdm.tqdm(lenta_reader(os.path.join(train_file, 'lenta/lenta-ru-news.val.csv')))]
        )

        ria_records = [r for r in tqdm.tqdm(ria_reader(os.path.join(train_file, 'ria/ria.shuffled.train.json')))]
        ria_records.extend(
            [r for r in tqdm.tqdm(ria_reader(os.path.join(train_file, 'ria/ria.shuffled.val.json')))]
        )

        random.shuffle(ria_records)

        all_records = [r for r in lenta_records if r['date'][:4] in ['2010', '2011', '2012', '2013', '2014']] + \
            ria_records[:220000]

        random.shuffle(all_records)

    print("Building datasets...")

    agency_to_special_token_id = {a: tokenizer.vocab[f'[unused{i+1}]'] for i, a in enumerate(agency_list)}

    full_dataset = AgencyTitleDatasetGeneration(
        all_records, tokenizer, 
        filter_agencies=list(agency_to_special_token_id.keys()), agency_to_special_token_id=agency_to_special_token_id,
        max_tokens_text=max_tokens_text, max_tokens_title=max_tokens_title
    )

    train_size = int(0.93 * len(full_dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset,
                                                               [train_size, len(full_dataset) - train_size])

    print(f"Train dataset length = {len(train_dataset)}\nVal dataset length = {len(val_dataset)}")
    wandb.summary.update({
        'Train dataset size': len(train_dataset),
        'Test dataset size': len(val_dataset),
    })

    print("Training model...")
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
        output_dir=output_model_path,
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
        save_total_limit=2,
        max_steps=max_steps,
        report_to='wandb',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        optimizers=opt,
    )

    trainer.train(checkpoint)
    model.save_pretrained(output_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("--train-file", type=str, required=True)
    parser.add_argument("--dataset-type", type=str, required=True, choices=['tg', 'lenta-ria'])
    parser.add_argument("--output-model-path", type=str, default="models/style_gen_title")
    parser.add_argument("--from-pretrained", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)

    args = parser.parse_args()
    train_style_gen_title(**vars(args))
