def normalize(text):
    text = text.replace('«', '"')
    text = text.replace('»', '"')
    text = text.replace('”', '"')
    text = text.replace('”', '"')
    text = text.replace('‘', '\'')
    text = text.replace('\xa0', ' ')
    return text