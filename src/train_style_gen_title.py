import argparse
import json
import random
import tqdm
import torch

from _jsonnet import evaluate_file as jsonnet_evaluate_file
from transformers import BertTokenizer, EncoderDecoderModel, Trainer, TrainingArguments, logging

from readers.tg_reader import tg_reader
from custom_datasets.agency_title_dataset import AgencyTitleDataset
from utils.training_utils import get_separate_lr_optimizer


def train_style_gen_title(
        config_file: str,
        train_file: str,
        train_sample_rate: float,
        output_model_path: str,
        from_pretrained: str = None,
        checkpoint: str = None
):
    logging.set_verbosity_info()

    config = json.loads(jsonnet_evaluate_file(config_file))

    agency_list = config.pop('agency_list', ['ТАСС', 'РТ на русском'])
    print('Agency list:', agency_list)

    tokenizer_model_path = config.pop("tokenizer_model_path")
    tokenizer = BertTokenizer.from_pretrained(tokenizer_model_path, do_lower_case=False, do_basic_tokenize=False)

    max_tokens_text = config.pop("max_tokens_text", 250)
    max_tokens_title = config.pop("max_tokens_title", 48)

    print("Initializing model...")

    if from_pretrained:
        model = EncoderDecoderModel.from_pretrained(from_pretrained)
    else:
        enc_model_path = config.pop("enc_model_path")
        dec_model_path = config.pop("dec_model_path")
        model = EncoderDecoderModel.from_encoder_decoder_pretrained(enc_model_path, dec_model_path)

    print("Model: ")
    print(model)

    print("Fetching data...")
    all_records = [r for r in tqdm.tqdm(tg_reader(train_file)) if random.random() <= train_sample_rate]

    print("Building datasets...")

    agency_to_special_token_id = {a: tokenizer.vocab[f'[unused{i+1}]'] for i, a in enumerate(agency_list)}

    full_dataset = AgencyTitleDataset(
        all_records,
        tokenizer,
        agency_list,
        agency_to_special_token_id,
        do_prepend_marker=True,
        max_tokens_text=max_tokens_text,
        max_tokens_title=max_tokens_title)

    train_size = int(0.9 * len(full_dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset,
                                                               [train_size, len(full_dataset) - train_size])

    print(f"Train dataset length = {len(train_dataset)}\nVal dataset length = {len(val_dataset)}") 

    print("Training model...")
    batch_size = config.pop("batch_size", 4)
    eval_steps = config.pop("eval_steps", 500)
    save_steps = config.pop("save_steps", 500)
    logging_steps = config.pop("logging_steps", 100)
    enc_lr = config.pop("enc_lr", 5e-5)
    dec_lr = config.pop("dec_lr", 5e-3)
    warmup_steps = config.pop("warmup_steps", 1000)
    max_steps = config.pop("max_steps", 10000)
    gradient_accumulation_steps = config.pop("gradient_accumulation_steps", 125)

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
        max_steps=max_steps
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        optimizers=opt
    )

    trainer.train(checkpoint)
    model.save_pretrained(output_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("--train-file", type=str, required=True)
    parser.add_argument("--train-sample-rate", type=float, default=1.0)
    parser.add_argument("--output-model-path", type=str, default="models/style_gen_title")
    parser.add_argument("--from-pretrained", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)

    args = parser.parse_args()
    train_style_gen_title(**vars(args))