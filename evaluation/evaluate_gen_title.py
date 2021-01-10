import argparse
import json
import random
import tqdm

from _jsonnet import evaluate_file as jsonnet_evaluate_file
from transformers import AutoTokenizer, EncoderDecoderModel, logging
import torch

from readers.ria_reader import ria_reader
from datasets.gen_title_dataset import GenTitleDataset
from models.bottleneck_encoder_decoder import BottleneckEncoderDecoderModel



def train_gen_title(
    config_file: str,
    eval_model_file: str,
    test_file: str,
    test_sample_rate: float,
    pred_out_file: str,
    enable_bottleneck: bool = False
):
    logging.set_verbosity_info()

    config = json.loads(jsonnet_evaluate_file(config_file))

    print("Fetching data...")
    test_records = [r for r in ria_reader(test_file) if random.random() <= test_sample_rate]

    print("Building datasets...")
    model_path = config.pop("model_path")
    tokenizer = AutoTokenizer.from_pretrained(model_path, do_lower_case=False)

    max_tokens_text = config.pop("max_tokens_text", 196)
    max_tokens_title = config.pop("max_tokens_title", 48)

    test_dataset = GenTitleDataset(
        test_records,
        tokenizer,
        max_tokens_text=max_tokens_text,
        max_tokens_title=max_tokens_title
    )

    print("Loading model...")
    cls = BottleneckEncoderDecoderModel if enable_bottleneck else EncoderDecoderModel
    model = cls.from_pretrained(eval_model_file)
    model.eval()

    batch_size = config.pop("batch_size", 8)

    with open(pred_out_file, 'w', encoding='utf-8') as f:
        for i in tqdm.trange(0, len(test_dataset), batch_size):
            data = test_dataset[i]
            del data['labels']

            for k in data.keys():
                data[k] = data[k].unsqueeze(0)

            for j in range(i + 1, min(i + batch_size, len(test_dataset))):
                for k in data.keys():
                    data[k] = torch.cat((data[k], test_dataset[j][k].unsqueeze(0)), dim=0)

            output_ids = model.generate(**data, decoder_start_token_id=model.config.decoder.pad_token_id)
            preds = [
                tokenizer.decode(x[1: torch.max(torch.nonzero(x)).item()]) for x in output_ids
            ]

            f.write('\n'.join(preds))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("--eval-model-file", type=str, required=True)
    parser.add_argument("--test-file", type=str, required=True)
    parser.add_argument("--test-sample-rate", type=float, default=1.0)
    parser.add_argument("--pred-out-file", type=str, required=True)
    parser.add_argument("--enable-bottleneck", default=False, action='store_true')

    args = parser.parse_args()
    train_gen_title(**vars(args))
