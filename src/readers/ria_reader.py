from bs4 import BeautifulSoup
import json


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
            yield record
