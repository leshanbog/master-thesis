import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"

from readers import lenta_reader, ria_reader_with_date_approx
import tqdm
from transformers import BertTokenizer, EncoderDecoderModel, logging

import pandas as pd
import numpy as np
import json
import random
from sklearn.metrics.pairwise import cosine_distances
from utils.clustering_utils import get_text_to_vector_func
import multiprocessing as mp


def get_embeds_for_records(records, text_to_vector_func):
    embeds = np.zeros((len(records), 768))

    for i in tqdm.trange(len(records)):
        text = records[i]["title"] + ' ' + records[i]["text"]
        text = text.lower().replace('\xa0', ' ').strip()
        embeds[i] = text_to_vector_func(text).detach().cpu().numpy().ravel()
        
    return embeds

clust_model = '/home/aobuhtijarov/models/lenta_ria_clustering_model/checkpoint-6000/'
tokenizer_path = '/home/aobuhtijarov/models/rubert_cased_L-12_H-768_A-12_pt/'


lenta_records = [r for r in tqdm.tqdm(lenta_reader('/home/aobuhtijarov/datasets/lenta/lenta-ru-news.train.csv'))]
lenta_records.extend(
    [r for r in tqdm.tqdm(lenta_reader('/home/aobuhtijarov/datasets/lenta/lenta-ru-news.val.csv'))]
)
lenta_records.extend(
    [r for r in tqdm.tqdm(lenta_reader('/home/aobuhtijarov/datasets/lenta/lenta-ru-news.test.csv'))]
)

ria_records = [r for r in tqdm.tqdm(ria_reader_with_date_approx('/home/aobuhtijarov/datasets/ria/ria.shuffled.train.json'))]
ria_records.extend(
    [r for r in tqdm.tqdm(ria_reader_with_date_approx('/home/aobuhtijarov/datasets/ria/ria.shuffled.val.json'))]
)
ria_records.extend(
    [r for r in tqdm.tqdm(ria_reader_with_date_approx('/home/aobuhtijarov/datasets/ria/ria.shuffled.test.json'))]
)

lenta_records = [r for r in lenta_records if r['date'][:4] in ['2010', '2011', '2012', '2013', '2014']]


model = EncoderDecoderModel.from_pretrained(clust_model)
tokenizer = BertTokenizer.from_pretrained(tokenizer_path, do_lower_case=False, do_basic_tokenize=False)
setattr(tokenizer, 'max_tokens_text', 250)
model.cuda()

text_to_vector_func = get_text_to_vector_func('bert-FirstCLS', model, tokenizer)

lenta_embeds = get_embeds_for_records(lenta_records, text_to_vector_func)
ria_embeds = get_embeds_for_records(ria_records, text_to_vector_func)


def f(start, total, n_jobs):
    print(start, total, n_jobs)
    
    with open(f'/home/aobuhtijarov/datasets/lenta_ria_{start}.jsonl', 'a', encoding='utf-8') as fout:
        for i in range(start, total, n_jobs):
            dist = cosine_distances(lenta_embeds[i].reshape(1, -1), ria_embeds).ravel()

            if i % 10000 == 0:
                print(i)

            top_ria_inds = np.argsort(dist)[:3]
            lenta_month = lenta_records[i]['date'][5:7]
            lenta_day = lenta_records[i]['date'][8:10]

            for j in top_ria_inds:
                ria_month = ria_records[j]['date'][5:7]
                ria_day = ria_records[j]['date'][8:10]

                if ria_month == lenta_month and ria_day == lenta_day:
                    l = lenta_records[i]
                    r = ria_records[j]
                    json.dump({
                        'lenta_text': l['text'],
                        'lenta_title': l['title'],
                        'lenta_date': l['date'],
                        'ria_text': r['text'],
                        'ria_title': r['title'],
                        'ria_date': r['date']
                    }, fout)
                    fout.write('\n')
                    break
    print('Done!')


n_jobs = 25

print('Starting workers')

proc = []

for i in range(n_jobs):
    proc.append(mp.Process(target=f, args=(i, len(lenta_records), n_jobs)))
    proc[-1].start()

for p in proc:
    p.join()
