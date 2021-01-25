import argparse
import json
import tqdm
import numpy as np

from _jsonnet import evaluate_file as jsonnet_evaluate_file
from transformers import AutoTokenizer, EncoderDecoderModel, logging
from sklearn.cluster import AgglomerativeClustering


from evaluation.clustering_utils import get_gold_markup, get_data_to_cluster, doc2vec_bert, calc_clustering_metrics
from models.bottleneck_encoder_decoder import BottleneckEncoderDecoderModel


def get_text_to_vector_func(text_to_vec_func, model, tokenizer):
    if text_to_vec_func == 'bert-MeanSum':
        return lambda doc: doc2vec_bert(doc, model, tokenizer, 'MeanSum')
    elif text_to_vec_func == 'bert-FirstCLS':
        return lambda doc: doc2vec_bert(doc, model, tokenizer, 'FirstCLS')
    else:
        raise NotImplementedError


def get_clf_report(embeds, markup, url2record, dist_threshold):
    clustering_model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=dist_threshold,
        linkage="single",
        affinity="cosine"
    )

    clustering_model.fit(embeds)
    labels = clustering_model.labels_

    id2url = dict()
    for i, (url, _) in enumerate(url2record.items()):
        id2url[i] = url

    url2label = dict()
    for i, label in enumerate(labels):
        url2label[id2url[i]] = label

    return calc_clustering_metrics(markup, url2label, url2record, output_dict=True), \
        calc_clustering_metrics(markup, url2label, url2record, output_dict=False)


def get_quality(embeds, markup, url2record, dist_threshold, print_result=False):
    dict_report, str_report = get_clf_report(embeds, markup, url2record, dist_threshold)

    if print_result:
        print(str_report)

    return dict_report['accuracy']


def perform_clustering_eval(config_file,
                            eval_model_file,
                            clustering_data_file,
                            gold_markup_file,
                            enable_bottleneck,
                            text_to_vec_func
):
    logging.set_verbosity_info()

    config = json.loads(jsonnet_evaluate_file(config_file))

    tokenizer_model_path = config.pop("tokenizer_model_path")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_path, do_lower_case=False)

    max_tokens_text = config.pop("max_tokens_text", 196)

    print("Loading model...")
    cls = BottleneckEncoderDecoderModel if enable_bottleneck else EncoderDecoderModel
    model = cls.from_pretrained(eval_model_file)
    model.eval()

    gold_markup = get_gold_markup(gold_markup_file)
    url2record, filename2url = get_data_to_cluster(clustering_data_file)
    setattr(tokenizer, 'max_tokens_text', max_tokens_text)
    text_to_vector_func = get_text_to_vector_func(text_to_vec_func, model, tokenizer)

    print('Calculating embeddings...')
    embeds = np.zeros((len(url2record.items()), 768))

    for i, (url, record) in tqdm.tqdm(enumerate(url2record.items())):
        text = record["title"] + ' ' + record["text"]
        text = text.lower().replace('\xa0', ' ')
        embeds[i] = text_to_vector_func(text).detach().numpy().ravel()

    print('Embeds shape =', embeds.shape)
    assert len(embeds) == len(url2record.items())

    print('Searching for optimal threshold')
    domain = np.logspace(-3, 0, 11)
    quals = [get_quality(embeds, gold_markup, url2record, dist)
            for dist in tqdm.tqdm(domain, total=11)]

    closer_domain = np.linspace(domain[max(0, np.argmax(quals) - 2)], domain[min(np.argmax(quals) + 3, len(domain) - 1)], 9)
    closer_quals = [get_quality(embeds, gold_markup, url2record, dist)
                    for dist in tqdm.tqdm(closer_domain, total=9)]

    best_dist = closer_domain[np.argmax(closer_quals)]

    print('\nBest distance =', best_dist)
    get_quality(embeds, gold_markup, url2record, best_dist, print_result=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("--eval-model-file", type=str, required=True)
    parser.add_argument("--clustering-data-file", type=str, required=True)  # ru_clustering_data.jsonl
    parser.add_argument("--gold-markup-file", type=str, required=True)  # ru_threads_target.tsv
    parser.add_argument("--enable-bottleneck", default=False, action='store_true')
    parser.add_argument("--text-to-vec-func", type=str, default='bert-MeanSum', choices=('bert-MeanSum', 'bert-FirstCLS'))

    args = parser.parse_args()
    perform_clustering_eval(**vars(args))