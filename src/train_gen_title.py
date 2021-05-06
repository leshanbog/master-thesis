import argparse
import json
import random
import tqdm
import torch

import os
import wandb

from _jsonnet import evaluate_file as jsonnet_evaluate_file
from transformers import BertTokenizer, EncoderDecoderModel, Trainer, TrainingArguments, logging

from readers import ria_reader, tg_reader, lenta_reader
from custom_datasets.gen_title_dataset import GenTitleDataset
from models.bottleneck_encoder_decoder import BottleneckEncoderDecoderModel
from utils.training_utils import get_separate_lr_optimizer, init_wandb


def train_gen_title(
    run_name: str,
    config_file: str,
    train_file: str,
    val_file: str,
    dataset_type: str,
    train_sample_rate: float,
    val_sample_rate: float,
    output_model_path: str,
    enable_bottleneck: bool = False,
    from_pretrained: str = None,
    checkpoint: str = None
):
    logging.set_verbosity_info()
    config = json.loads(jsonnet_evaluate_file(config_file))

    init_wandb(run_name, config)

    tokenizer_model_path = config["tokenizer_model_path"]
    tokenizer = BertTokenizer.from_pretrained(tokenizer_model_path, do_lower_case=False, do_basic_tokenize=False)

    max_tokens_text = config["max_tokens_text"]
    max_tokens_title = config["max_tokens_title"]

    print("Initializing model...")

    cls = BottleneckEncoderDecoderModel if enable_bottleneck else EncoderDecoderModel

    if from_pretrained:
        model = cls.from_pretrained(from_pretrained)
    else:
        enc_model_path = config["enc_model_path"]
        dec_model_path = config["dec_model_path"]
        model = cls.from_encoder_decoder_pretrained(enc_model_path, dec_model_path)

    # print("Model: ")
    # print(model)
    model.cuda()

    if dataset_type == 'ria':
        print("Fetching RIA data...")
        train_records = [r for r in tqdm.tqdm(ria_reader(train_file)) if random.random() <= train_sample_rate]
        val_records = [r for r in tqdm.tqdm(ria_reader(val_file)) if random.random() <= val_sample_rate]

        print("Building datasets...")

        train_dataset = GenTitleDataset(
            train_records,
            tokenizer,
            max_tokens_text=max_tokens_text,
            max_tokens_title=max_tokens_title)
    
        val_dataset = GenTitleDataset(
            val_records,
            tokenizer,
            max_tokens_text=max_tokens_text,
            max_tokens_title=max_tokens_title)
    elif dataset_type == 'tg':
        print("Fetching TG data...")
        all_records = [r for r in tqdm.tqdm(tg_reader(train_file)) if random.random() <= train_sample_rate]

        print("Building datasets...")

        full_dataset = GenTitleDataset(
                        all_records,
                        tokenizer,
                        max_tokens_text=max_tokens_text,
                        max_tokens_title=max_tokens_title)
            
        train_size = int(0.995 * len(full_dataset))
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset,
                                                                   [train_size, len(full_dataset) - train_size])
    elif dataset_type == 'lenta-ria':
        print('Fetching Lenta-RIA data...')
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
            ria_records[:300000]

        random.shuffle(all_records)

        print("Building datasets...")

        full_dataset = GenTitleDataset(
            all_records, tokenizer,
            max_tokens_text=max_tokens_text, max_tokens_title=max_tokens_title
        )
            
        train_size = int(0.99 * len(full_dataset))
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset,
                                                                   [train_size, len(full_dataset) - train_size])

    wandb.summary.update({
        'Train dataset size': len(train_dataset),
        'Val dataset size': len(val_dataset)
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
    parser.add_argument("--val-file", type=str, required=False)
    parser.add_argument("--dataset-type", type=str, choices=('ria', 'tg', 'lenta-ria'), default='ria')
    parser.add_argument("--train-sample-rate", type=float, default=1.0)
    parser.add_argument("--val-sample-rate", type=float, default=1.0)
    parser.add_argument("--output-model-path", type=str, required=True)
    parser.add_argument("--enable-bottleneck", default=False, action='store_true')
    parser.add_argument("--from-pretrained", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)

    args = parser.parse_args()
    train_gen_title(**vars(args))
