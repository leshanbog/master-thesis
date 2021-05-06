from bs4 import BeautifulSoup
import json
from .text_normalizer import normalize


def ria_reader(path):
    with open(path, "r", encoding="utf-8") as r:
        for line in r:
            record = json.loads(line.strip())

            assert "title" in record
            assert "text" in record

            record["text"] = BeautifulSoup(record["text"], 'html.parser').text.replace('\xa0', ' ').replace('\n', ' ')

            if not record["text"] or not record["title"] or \
                    record["text"].count(' ') < 20 or record["title"].count(' ') < 5:
                continue

            record['agency'] = 'РИА Новости'
            record['date'] = '2012-01-01 10:00'
            record['text'] = normalize(record['text'])
            record['title'] = normalize(record['title'])
            yield record

def prepend(x):
    x = str(x)
    if len(x) == 2:
        return x
    else:
        return '0' + x

months = [' янв', ' фев', ' мар', ' апр', ' мая', ' июн', ' июл', ' авг', ' сен', ' окт', ' ноя', ' дек']
dates = [str(i) for i in range(1, 32)]

def ria_date_from_text(text):

    try:
        a = text[:70].split('риа новости')[0].split(', ')[1]

        date = '2010-'

        for i, m in enumerate(months):
            if m in a:
                date += prepend(i + 1) + '-'
                date += prepend(a.split(m)[0])

        assert len(date) == 10
        return date
    except:
        return ''


def ria_reader_with_date_approx(path):
    with open(path, "r", encoding="utf-8") as r:
        for line in r:
            record = json.loads(line.strip())

            assert "title" in record
            assert "text" in record

            record["text"] = BeautifulSoup(record["text"], 'html.parser').text.replace('\xa0', ' ').replace('\n', ' ')

            if not record["text"] or not record["title"] or \
                    record["text"].count(' ') < 20 or record["title"].count(' ') < 5:
                continue

            record['agency'] = 'РИА Новости'
            record['date'] = ria_date_from_text(record['text'])

            if not record['date']:
                continue

            record['date'] += ' 10:00'
            record['text'] = normalize(record['text'])
            record['title'] = normalize(record['title'])
            yield record
