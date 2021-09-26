import torch


def doc2vec_bert(text, model, tokenizer, mode='MeanSum', device='cuda:0'):
    inp = torch.LongTensor(tokenizer.encode(text))[:tokenizer.max_tokens_text].unsqueeze(0)
    output = model.to(device).encoder(inp.to(device))['last_hidden_state'][0]

    if mode == 'FirstCLS':
        return output[0]
    elif mode == 'MeanSum':
        return output.mean(0)
    else:
        raise Exception('Wrong mode')

def get_text_to_vector_func(text_to_vec_func, model, tokenizer, device='cuda:0'):
    if text_to_vec_func == 'bert-MeanSum':
        return lambda doc: doc2vec_bert(doc, model, tokenizer, 'MeanSum', device)
    elif text_to_vec_func == 'bert-FirstCLS':
        return lambda doc: doc2vec_bert(doc, model, tokenizer, 'FirstCLS', device)
    else:
        raise NotImplementedError
