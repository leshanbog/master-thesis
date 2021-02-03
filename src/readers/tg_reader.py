import pandas as pd


def tg_reader(path, agency_list=None):
    with open(path, "r", encoding="utf-8") as r:
        data = pd.read_json(path, lines=True)
        for i in range(len(data)):
            text = data.iloc[i]['text'].lower().replace('\xa0', ' ').replace('\n', ' ').strip()
            title = data.iloc[i]['title'].lower()

            if not text or not title or text.count(' ') < 3 or title.count(' ') < 3 or \
                    (agency_list is not None and data.iloc[i]['site_name'] not in agency_list):
                continue
                
            yield {
                'text': text,
                'title': title,
                'agency': data.iloc[i]['site_name']
            }
