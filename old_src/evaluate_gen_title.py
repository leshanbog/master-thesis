import os
import argparse
import json
import random
import tqdm
import re
import razdel
import nltk

from _jsonnet import evaluate_file as jsonnet_evaluate_file
from transformers import BertTokenizer, EncoderDecoderModel, logging
import torch
import wandb

from readers.ria_reader import ria_reader
from readers.tg_reader import tg_reader
from custom_datasets.gen_title_dataset import GenTitleDataset
from custom_datasets.agency_title_dataset import AgencyTitleDatasetGeneration
from models.bottleneck_encoder_decoder import BottleneckEncoderDecoderModel
from utils.gen_title_calculate_metrics import print_metrics, punct_detokenize, postprocess, first_sent
from utils.clusterer import Clusterer
from utils.training_utils import init_wandb


def evaluate_and_print_metrics(
    predicted_path: str,
    gold_path: str,
    language='ru',
    max_count=None,
    is_multiple_ref=False,
    detokenize_after=False,
    tokenize_after=False,
    lower=False,
    are_clusters_used=False,
):
    hyps = []
    refs = []
    
    table = wandb.Table(columns=['Reference', 'Prediction', 'Reference processed', 'Prediction processed'])

    with open(gold_path, "r") as gold, open(predicted_path, "r") as pred:
        for i, (ref, hyp) in enumerate(zip(gold, pred)):

            if i % 500 == 0:
                ref_before = ref
                hyp_before = hyp

            if max_count is not None and i >= max_count:
                break

            ref, hyp = postprocess(ref, hyp, language, is_multiple_ref, detokenize_after, tokenize_after, lower)
            if not hyp:
                print("Empty hyp for ref: ", ref)
                continue
            if not ref:
                continue

            if i % 500 == 0:
                table.add_data(ref_before, hyp_before, ref, hyp)

            refs.append(ref)
            hyps.append(hyp)

    if are_clusters_used:
        wandb.run.summary.update({'Examples with multiple references': table})
    else:
        wandb.run.summary.update({'Examples': table})

    print_metrics(refs, hyps, language=language, are_clusters_used=are_clusters_used)

def make_inference_and_save(
    config_file,
    eval_model_file,
    test_file,
    test_sample_rate,
    enable_bottleneck,
    cluster_model_file,
    clustering_dist_threshold,
    out_path_prefix,
    dataset_type,
    style_model_eval,
):
    config = json.loads(jsonnet_evaluate_file(config_file))

    tokenizer_model_path = config["tokenizer_model_path"]
    tokenizer = BertTokenizer.from_pretrained(tokenizer_model_path, do_lower_case=False, do_basic_tokenize=False)

    max_tokens_text = config["max_tokens_text"]
    max_tokens_title = config["max_tokens_title"]
    setattr(tokenizer, 'max_tokens_text', max_tokens_text)

    batch_size = config["batch_size"]

    print("Loading model...")
    cls = BottleneckEncoderDecoderModel if enable_bottleneck else EncoderDecoderModel
    model = cls.from_pretrained(eval_model_file)
    model.eval()
    model.cuda()

    if cluster_model_file:
        test_sample_rate = 1.
        filter_dates = ('2020-05-12', )
    else:
        filter_dates = None

    if dataset_type == 'ria':
        print("Fetching RIA data...")
        test_records = [r for r in tqdm.tqdm(ria_reader(test_file)) if random.random() <= test_sample_rate]
    else:
        print("Fetching TG data...")
        test_records = [r for r in tqdm.tqdm(tg_reader(test_file, filter_dates=filter_dates)) 
            if random.random() <= test_sample_rate]

    print("Building datasets...")

    if style_model_eval:
        agency_list = config['agency_list']
        agency_to_special_token_id = {a: tokenizer.vocab[f'[unused{i+1}]'] for i, a in enumerate(agency_list)}

        test_dataset = AgencyTitleDatasetGeneration(
            test_records, tokenizer,
            filter_agencies=None, agency_to_special_token_id=agency_to_special_token_id,
            max_tokens_text=max_tokens_text, max_tokens_title=max_tokens_title
        )
    else:
        test_dataset = GenTitleDataset(
            test_records, tokenizer,
            max_tokens_text=max_tokens_text, max_tokens_title=max_tokens_title
        )

    print('Dataset size:', len(test_dataset))

    if cluster_model_file:
        from utils.clustering_utils import get_text_to_vector_func
        clusterer = Clusterer(
            get_text_to_vector_func(
                'bert-FirstCLS',
                BottleneckEncoderDecoderModel.from_pretrained(cluster_model_file),
                tokenizer),
            test_dataset,
            clustering_dist_threshold,
            dates=filter_dates,
        )

        clusterer.perform_clustering()

    with open(out_path_prefix + 'prediction.txt', 'w', encoding='utf-8') as pf, \
            open(out_path_prefix + 'gold.txt', 'w', encoding='utf-8') as gf:
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

            for j in range(i, min(i + batch_size, len(test_dataset))):
                if cluster_model_file:
                    refs = []
                    for r in clusterer.get_cluster_records(j):
                        refs.append(r['title'])

                    gf.write(' s_s '.join(refs) + '\n')
                else:
                    gf.write(test_dataset.get_strings(j)['title'] + '\n')
                pf.write(preds[j - i] + '\n')


def evaluate_gen_title(
    existing_run_name: str,
    existing_run_id: str,
    config_file: str,
    do_inference: bool,
    eval_model_file: str,
    test_file: str,
    test_sample_rate: float,
    out_dir: str,
    dataset_type: str,
    enable_bottleneck: bool = False,
    cluster_model_file: str = None,
    clustering_dist_threshold: float = 0.18,
    style_model_eval: bool = False,
    detokenize_after: bool = False,
    tokenize_after: bool = False
):
    logging.set_verbosity_info()
    init_wandb(existing_run_name, None, existing_run_id)

    out_path_prefix = os.path.join(out_dir, eval_model_file[eval_model_file.index('checkpoint'):])
    if out_path_prefix[-1] == '/':
        out_path_prefix = out_path_prefix[:-1]

    out_path_prefix += '-'

    if do_inference == '1':
        make_inference_and_save(
            config_file, eval_model_file, 
            test_file, test_sample_rate, 
            enable_bottleneck, cluster_model_file, clustering_dist_threshold,
            out_path_prefix, dataset_type,
            style_model_eval
        )

    evaluate_and_print_metrics(
        out_path_prefix + 'prediction.txt',
        out_path_prefix + 'gold.txt',
        detokenize_after=detokenize_after,
        tokenize_after=tokenize_after,
        is_multiple_ref=(cluster_model_file is not None),
        lower=True,
        are_clusters_used=(cluster_model_file is not None)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--existing-run-name", type=str, required=True)
    parser.add_argument("--existing-run-id", type=str, required=True)
    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("--do-inference", type=str, required=True)
    parser.add_argument("--eval-model-file", type=str, required=True)
    parser.add_argument("--test-file", type=str, required=True)
    parser.add_argument("--test-sample-rate", type=float, default=1.0)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--dataset-type", type=str, choices=('ria', 'tg'), required=True)
    parser.add_argument("--enable-bottleneck", default=False, action='store_true')
    parser.add_argument("--cluster-model-file", default=None, type=str)
    parser.add_argument("--clustering-dist-threshold", default=0.18, type=float)
    parser.add_argument("--style-model-eval", default=False, action='store_true')  # it means we evaluating model, that can generate with different styles
    parser.add_argument("--detokenize-after", default=False, action='store_true')
    parser.add_argument("--tokenize-after", default=False, action='store_true')

    args = parser.parse_args()
    evaluate_gen_title(**vars(args))
