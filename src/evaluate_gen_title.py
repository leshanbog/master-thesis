import argparse
import json
import random
import tqdm
import re

from _jsonnet import evaluate_file as jsonnet_evaluate_file
from transformers import AutoTokenizer, EncoderDecoderModel, logging
import torch

from readers.ria_reader import ria_reader
from datasets.gen_title_dataset import GenTitleDataset
from models.bottleneck_encoder_decoder import BottleneckEncoderDecoderModel


def punct_detokenize(text):
    text = text.strip()
    punctuation = ",.!?:;%"
    closing_punctuation = ")]}"
    opening_punctuation = "([}"
    for ch in punctuation + closing_punctuation:
        text = text.replace(" " + ch, ch)
    for ch in opening_punctuation:
        text = text.replace(ch + " ", ch)
    res = [r'"\s[^"]+\s"', r"'\s[^']+\s'"]
    for r in res:
        for f in re.findall(r, text, re.U):
            text = text.replace(f, f[0] + f[2:-2] + f[-1])
    text = text.replace("' s", "'s").replace(" 's", "'s")
    text = text.strip()
    return text


def postprocess(ref, hyp, language, is_multiple_ref=False, detokenize_after=False, tokenize_after=False, lower=False):
    if is_multiple_ref:
        reference_sents = ref.split(" s_s ")
        decoded_sents = hyp.split("s_s")
        hyp = [w.replace("<", "&lt;").replace(">", "&gt;").strip() for w in decoded_sents]
        ref = [w.replace("<", "&lt;").replace(">", "&gt;").strip() for w in reference_sents]
        hyp = " ".join(hyp)
        ref = " ".join(ref)
    ref = ref.strip()
    hyp = hyp.strip()
    if detokenize_after:
        hyp = punct_detokenize(hyp)
        ref = punct_detokenize(ref)
    if tokenize_after:
        if language == "ru":
            hyp = " ".join([token.text for token in razdel.tokenize(hyp)])
            ref = " ".join([token.text for token in razdel.tokenize(ref)])
        else:
            hyp = " ".join([token for token in nltk.word_tokenize(hyp)])
            ref = " ".join([token for token in nltk.word_tokenize(ref)])
    if lower:
        hyp = hyp.lower()
        ref = ref.lower()
    return ref, hyp


def evaluate_and_print_metrics(
    predicted_path: str,
    gold_path: str,
    metric,
    language,
    max_count=None,
    is_multiple_ref=False,
    detokenize_after=False,
    tokenize_after=False,
    lower=False
):
    hyps = []
    refs = []

    with open(gold_path, "r") as gold, open(predicted_path, "r") as pred:
        for i, (ref, hyp) in enumerate(zip(gold, pred)):
            if max_count is not None and i >= max_count:
                break
            ref, hyp = postprocess(ref, hyp, language, is_multiple_ref, detokenize_after, tokenize_after, lower)
            if not hyp:
                print("Empty hyp for ref: ", ref)
                continue
            if not ref:
                continue
            refs.append(ref)
            hyps.append(hyp)

    print_metrics(refs, hyps, metric=metric, language=language)


def make_inference_and_save(
        config_file,
        eval_model_file,
        test_file,
        test_sample_rate,
        enable_bottleneck,
        out_dir
):
    config = json.loads(jsonnet_evaluate_file(config_file))

    print("Fetching data...")
    test_records = [r for r in ria_reader(test_file) if random.random() <= test_sample_rate]

    print("Building datasets...")
    tokenizer_model_path = config.pop("tokenizer_model_path")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_path, do_lower_case=False)

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

    out_path_prefix = out_dir + '/' + eval_model_file[eval_model_file.index('checkpoint'):] + '-'

    with open(out_path_prefix + 'prediction.txt', 'w', encoding='utf-8') as pf, \
            open(out_path_prefix + 'gold.txt', 'w', encoding='utf-8') as gf:
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

            pf.write('\n'.join(preds))

            gf.write(
                '\n'.join([
                    test_dataset.get_strings(j)['title'] for j in range(i + 1, min(i + batch_size, len(test_dataset)))
                ])
            )


def evaluate_gen_title(
    config_file: str,
    do_inference: bool,
    eval_model_file: str,
    test_file: str,
    test_sample_rate: float,
    out_dir: str,
    enable_bottleneck: bool = False
):
    logging.set_verbosity_info()

    if do_inference:
        make_inference_and_save(config_file, eval_model_file, test_file, test_sample_rate, enable_bottleneck, out_dir)

    evaluate_and_print_metrics()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("--do-inference", type=bool, required=True)
    parser.add_argument("--eval-model-file", type=str, required=True)
    parser.add_argument("--test-file", type=str, required=True)
    parser.add_argument("--test-sample-rate", type=float, default=1.0)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--enable-bottleneck", default=False, action='store_true')

    args = parser.parse_args()
    evaluate_gen_title(**vars(args))