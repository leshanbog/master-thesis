import argparse
import json
import random
import tqdm

from _jsonnet import evaluate_file as jsonnet_evaluate_file
from transformers import AutoTokenizer, EncoderDecoderModel, PreTrainedModel, PretrainedConfig, \
    Trainer, TrainingArguments, logging

from readers.ria_reader import ria_reader
from datasets.gen_title_dataset import GenTitleDataset
from models.bottleneck_encoder_decoder import BottleneckEncoderDecoderModel


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
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_path, do_lower_case=False)

    max_tokens_text = config.pop("max_tokens_text", 196)
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
    batch_size = config.pop("batch_size", 8)
    eval_steps = config.pop("eval_steps", 100)
    save_steps = config.pop("save_steps", 100)
    logging_steps = config.pop("logging_steps", 25)
    learning_rate = config.pop("learning_rate", 5e-05)
    warmup_steps = config.pop("warmup_steps", 150)
    num_train_epochs = config.pop("num_train_epochs", 5)
    gradient_accumulation_steps = config.pop("gradient_accumulation_steps", 25)

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
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        save_total_limit=2,
        num_train_epochs=num_train_epochs
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
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
