import argparse
import json
import random
import tqdm
import torch

from _jsonnet import evaluate_file as jsonnet_evaluate_file
from transformers import BertTokenizer, EncoderDecoderModel, PreTrainedModel, PretrainedConfig, \
    Trainer, TrainingArguments, logging

from readers.ria_reader import ria_reader
from datasets.gen_title_dataset import GenTitleDataset
from models.bottleneck_encoder_decoder import BottleneckEncoderDecoderModel


def get_separate_lr_optimizer(model, enc_lr, dec_lr, warmup_steps, total_train_steps):
    from transformers import get_linear_schedule_with_warmup
    enc = []
    dec = []

    for name, param in model.named_parameters():
        if name.startswith('encoder'):
            enc.append(param)
        elif name.startswith('decoder'):
            dec.append(param)
        else:
            raise ValueError

    optimizer = torch.optim.AdamW([
        {'params': dec, 'lr': dec_lr},
        {'params': enc, 'lr': enc_lr}
    ])

    return optimizer, get_linear_schedule_with_warmup(optimizer, warmup_steps, total_train_steps)


def train_gen_title(
    config_file: str,
    train_file: str,
    val_file: str,
    train_sample_rate: float,
    val_sample_rate: float,
    output_model_path: str,
    enable_bottleneck: bool = False,
    from_pretrained: str = None,
    checkpoint: str = None
):
    logging.set_verbosity_info()

    config = json.loads(jsonnet_evaluate_file(config_file))

    print("Fetching data...")
    train_records = [r for r in tqdm.tqdm(ria_reader(train_file)) if random.random() <= train_sample_rate]
    val_records = [r for r in tqdm.tqdm(ria_reader(val_file)) if random.random() <= val_sample_rate]

    print("Building datasets...")
    tokenizer_model_path = config.pop("tokenizer_model_path")
    tokenizer = BertTokenizer.from_pretrained(tokenizer_model_path, do_lower_case=False, do_basic_tokenize=False)

    max_tokens_text = config.pop("max_tokens_text", 250)
    max_tokens_title = config.pop("max_tokens_title", 48)

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

    print("Initializing model...")

    cls = BottleneckEncoderDecoderModel if enable_bottleneck else EncoderDecoderModel
    if from_pretrained:
        model = cls.from_pretrained(from_pretrained)
    else:
        enc_model_path = config.pop("enc_model_path")
        dec_model_path = config.pop("dec_model_path")
        model = cls.from_encoder_decoder_pretrained(enc_model_path, dec_model_path)

    print("Model: ")
    print(model)

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
    parser.add_argument("--val-file", type=str, required=True)
    parser.add_argument("--train-sample-rate", type=float, default=1.0)
    parser.add_argument("--val-sample-rate", type=float, default=1.0)
    parser.add_argument("--output-model-path", type=str, default="models/gen_title")
    parser.add_argument("--enable-bottleneck", default=False, action='store_true')
    parser.add_argument("--from-pretrained", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)

    args = parser.parse_args()
    train_gen_title(**vars(args))
