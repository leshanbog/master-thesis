import argparse
import json
import random
import tqdm
import re
import razdel
import nltk
import wandb

from _jsonnet import evaluate_file as jsonnet_evaluate_file
from transformers import BertTokenizer, EncoderDecoderModel, AutoModelForSequenceClassification, logging
import torch
from sklearn.metrics import classification_report

from readers.tg_reader import tg_reader
from custom_datasets.agency_title_dataset import AgencyTitleDatasetGeneration
from utils.training_utils import init_wandb


def first_sent(x, token_id):
    lx = list(x)
    if token_id in x:
        return x[:lx.index(token_id)]
    return x


def evaluate_style_gen_title(
    existing_run_name: str,
    existing_run_id: str,
    config_file: str,
    gen_model_file: str,
    discr_model_file: str,
    test_file: str,
    test_sample_rate: float,
):
    logging.set_verbosity_info()
    init_wandb(existing_run_name, None, existing_run_id)

    config = json.loads(jsonnet_evaluate_file(config_file))

    tokenizer_model_path = config["tokenizer_model_path"]
    tokenizer = BertTokenizer.from_pretrained(tokenizer_model_path, do_lower_case=False, do_basic_tokenize=False)

    max_tokens_text = config["max_tokens_text"]
    max_tokens_title = config["max_tokens_title"]
    setattr(tokenizer, 'max_tokens_text', max_tokens_text)

    batch_size = config["batch_size"]

    print("Loading model...")
    model = EncoderDecoderModel.from_pretrained(gen_model_file)
    model.eval()
    model.cuda()

    agency_list = config['agency_list']
    discriminator = AutoModelForSequenceClassification.from_pretrained(discr_model_file, num_labels=len(agency_list)).cuda()
    
    print("Fetching TG data...")
    test_records = [r for r in tqdm.tqdm(tg_reader(test_file)) 
        if random.random() <= test_sample_rate]
    
    print("Building datasets...")
    
    
    agency_to_special_token_id = {
        a: tokenizer.vocab[f'[unused{i+1}]'] for i, a in enumerate(agency_list)
    }

    agency_to_target = {a: i for i, a in enumerate(sorted(agency_list))}

    test_dataset = AgencyTitleDatasetGeneration(
        test_records, tokenizer,
        filter_agencies=list(agency_to_special_token_id.keys()),
        agency_to_special_token_id=agency_to_special_token_id,
        max_tokens_text=max_tokens_text, max_tokens_title=max_tokens_title
    )

    print('Dataset size:', len(test_dataset))

    y_pred = []
    y_true = []

    for i in tqdm.trange(0, len(test_dataset), batch_size):
        data = test_dataset[i]
        for k in tuple(data.keys()):
            if k not in ('input_ids', 'attention_mask'):
                del data[k]
            else:
                data[k] = data[k].unsqueeze(0)

        for j in range(i + 1, min(i + batch_size, len(test_dataset))):
            for k in data.keys():
                data[k] = torch.cat((data[k], test_dataset[j][k].unsqueeze(0)), dim=0)

        y_true.extend([ agency_to_target[test_dataset.get_strings(j)['agency']]
            for j in range(i, min(i + batch_size, len(test_dataset)))])

        data['input_ids'] = data['input_ids'].cuda()
        data['attention_mask'] = data['attention_mask'].cuda()

        output_ids = model.generate(
            **data,
            decoder_start_token_id=model.config.decoder.pad_token_id,
            min_length=7,
            max_length=20,
            num_beams=6
        )

        preds = [
            tokenizer.decode(first_sent(x, tokenizer.sep_token_id), skip_special_tokens=True) for x in output_ids
        ]

        for title in preds:
            inp = tokenizer(title, 
                add_special_tokens=True, max_length=max_tokens_title,
                padding='max_length', truncation=True
            )

            logits = discriminator(input_ids=torch.LongTensor(inp['input_ids']).cuda().unsqueeze(0), 
                                   attention_mask=torch.LongTensor(inp['attention_mask']).cuda().unsqueeze(0))[0]
            y_pred.append(torch.argmax(logits).item())

    wandb.summary.update({
        'D-Style': classification_report(y_true, y_pred, output_dict=True)
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--existing-run-name", type=str, required=True)
    parser.add_argument("--existing-run-id", type=str, required=True)
    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("--gen-model-file", type=str, required=True)
    parser.add_argument("--discr-model-file", type=str, required=True)
    parser.add_argument("--test-file", type=str, required=True)
    parser.add_argument("--test-sample-rate", type=float, default=1.0)

    args = parser.parse_args()
    evaluate_style_gen_title(**vars(args))
