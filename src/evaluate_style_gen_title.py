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

from readers.ria_reader import ria_reader
from custom_datasets.gen_title_dataset import GenTitleDataset
from models.bottleneck_encoder_decoder import BottleneckEncoderDecoderModel
from evaluation.gen_title_calculate_metrics import print_metrics



def evaluate_and_print_style_metrics(
    predicted_path: str,
    gold_path: str,
    language='ru',
    max_count=None,
    is_multiple_ref=False,
    detokenize_after=False,
    tokenize_after=False,
    lower=False
):
    pass


def make_inference_and_save(
        config_file,
        eval_model_file,
        test_file,
        test_sample_rate,
        enable_bottleneck,
        out_path_prefix
):
    pass


def evaluate_style_gen_title(
    config_file: str,
    do_inference: bool,
    eval_model_file: str,
    test_file: str,
    test_sample_rate: float,
    out_dir: str,
    enable_bottleneck: bool = False,
    detokenize_after: bool = False,
    tokenize_after: bool = False
):
    logging.set_verbosity_info()
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("--do-inference", type=str, required=True)
    parser.add_argument("--gen-model-file", type=str, required=True)
    parser.add_argument("--discr-model-file", type=str, required=True)
    parser.add_argument("--test-file", type=str, required=True)
    parser.add_argument("--test-sample-rate", type=float, default=1.0)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--enable-bottleneck", default=False, action='store_true')
    parser.add_argument("--detokenize-after", default=False, action='store_true')
    parser.add_argument("--tokenize-after", default=False, action='store_true')

    args = parser.parse_args()
    evaluate_style_gen_title(**vars(args))
