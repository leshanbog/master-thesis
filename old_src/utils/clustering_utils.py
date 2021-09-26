from razdel import sentenize
import csv
import json
from collections import defaultdict
from sklearn.metrics import classification_report



def read_markup(file_name):
    with open(file_name, "r") as r:
        reader = csv.reader(r, delimiter='\t', quotechar='"')
        header = next(reader)
        for row in reader:
            assert len(header) == len(row)
            record = dict(zip(header, row))
            yield record


def get_gold_markup(markup_path):
    markup = defaultdict(dict)
    for record in read_markup(markup_path):
        first_url = record["INPUT:first_url"]
        second_url = record["INPUT:second_url"]
        quality = int(record["OUTPUT:quality"] == "OK")
        markup[(first_url, second_url)] = quality
    return markup


def get_data_to_cluster(clustering_data_file):
    url2record = dict()
    filename2url = dict()
    with open(clustering_data_file, "r") as r:
        for line in r:
            record = json.loads(line)
            url2record[record["url"]] = record
            filename2url[record["file_name"]] = record["url"]
    return url2record, filename2url


def calc_clustering_metrics(gold_markup, url2label, url2record, output_dict=False):
    not_found_count = 0
    for first_url, second_url in list(gold_markup.keys()):
        not_found_in_labels = first_url not in url2label or second_url not in url2label
        not_found_in_records = first_url not in url2record or second_url not in url2record
        if not_found_in_labels or not_found_in_records:
            not_found_count += 1
            gold_markup.pop((first_url, second_url))

    assert not_found_count == 0

    targets = []
    predictions = []
    for (first_url, second_url), target in gold_markup.items():
        prediction = int(url2label[first_url] == url2label[second_url])
        first = url2record.get(first_url)
        second = url2record.get(second_url)
        targets.append(target)
        predictions.append(prediction)

    return classification_report(targets, predictions, output_dict=output_dict)
