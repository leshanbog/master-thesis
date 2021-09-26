import csv
from .text_normalizer import normalize

def lenta_reader(path):
    with open(path, "r", encoding="utf-8") as r:
        reader = csv.reader(r, delimiter=",", quotechar='"')
        header = next(reader)
        assert header[1] == "title"
        assert header[2] == "text"
        for row in reader:
            if len(row) < 3:
                continue
            title, text = row[1], row[2]
            if not title or not text:
                continue
            text = text.replace("\xa0", " ").lower()
            title = title.replace("\xa0", " ").lower()

            date = '-'.join(row[0].split('/')[4:7]) + ' 10:00'
            yield {
                'text': normalize(text),
                'title': normalize(title),
                'date': date,
                'agency': 'lenta.ru',
            }
