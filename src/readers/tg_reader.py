import pandas as pd
from datetime import datetime
from .text_normalizer import normalize


def tg_reader(path, agency_list=None, filter_dates=None):
    chaunks = pd.read_json(path, lines=True, chunksize=1024)
    for data in chaunks:
        for i in range(len(data)):
            text = data.iloc[i]['text'].lower().replace('\xa0', ' ').replace('\n', ' ').strip()
            title = data.iloc[i]['title'].lower().replace('\xa0', ' ').replace('\n', ' ').strip()
            try:
                date = str(datetime.fromtimestamp(data.iloc[i]['timestamp']))
            except TypeError:
                date = str(data.iloc[i]['timestamp'])

            if not text or not title or text.count(' ') < 3 or title.count(' ') < 3 or \
                    (agency_list is not None and data.iloc[i]['site_name'] not in agency_list) or \
                    (filter_dates is not None and not any(date.startswith(d) for d in filter_dates)):
                continue

            yield {
                'text': normalize(text),
                'title': normalize(title),
                'agency': data.iloc[i]['site_name'],
                'date': date,
            }