from collections import Counter

from rouge import Rouge
from nltk.translate.bleu_score import corpus_bleu


def calc_duplicate_n_grams_rate(documents):
    all_ngrams_count = Counter()
    duplicate_ngrams_count = Counter()
    for doc in documents:
        words = doc.split(" ")
        for n in range(1, 5):
            ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
            unique_ngrams = set(ngrams)
            all_ngrams_count[n] += len(ngrams)
            duplicate_ngrams_count[n] += len(ngrams) - len(unique_ngrams)
    return {
        n: duplicate_ngrams_count[n]/all_ngrams_count[n]
            if all_ngrams_count[n] else 0.0
        for n in range(1, 5)
    }


def calc_rouge(refs, hyps, multiple_ref_method):
    r1, r2, rl = 0., 0., 0.
    rouge = Rouge()

    for multiple_refs, hyp in zip(refs, hyps):
        r1_list = [rouge.get_scores(hyp, ref, avg=True)['rouge-1']['f'] for ref in multiple_refs]
        r2_list = [rouge.get_scores(hyp, ref, avg=True)['rouge-2']['f'] for ref in multiple_refs]
        rl_list = [rouge.get_scores(hyp, ref, avg=True)['rouge-l']['f'] for ref in multiple_refs]

        if multiple_ref_method == 'average':
            r1 += sum(r1_list) / len(multiple_refs)
            r2 += sum(r2_list) / len(multiple_refs)
            rl += sum(rl_list) / len(multiple_refs)
        elif multiple_ref_method == 'best':
            r1 += max(r1_list)
            r2 += max(r2_list)
            rl += max(rl_list)
        else:
            raise ValueError("Wrnog method for resolving multiple refs")

    return {
        'r1': r1 / len(refs),
        'r2': r2 / len(refs),
        'rl': rl / len(refs)
    }


def calc_metrics(refs, hyps, language):
    assert len(refs) == len(hyps)
    assert all(type(x) == list and type(x[0]) == str for x in refs)
    assert all(type(x) == str for x in hyps)

    metrics = dict()
    metrics["count"] = len(hyps)
    metrics["bleu"] = corpus_bleu(refs, hyps)
 
    metrics.update(calc_rouge(refs, hyps, multiple_ref_method='best'))

    metrics["duplicate_ngrams"] = dict()
    metrics["duplicate_ngrams"].update(calc_duplicate_n_grams_rate(hyps))
    return metrics


def print_metrics(refs, hyps, language, are_clusters_used=False):
    import wandb
    metrics = calc_metrics(refs, hyps, language=language)

    print(metrics)

    if are_clusters_used:
        wandb.run.summary.update({
            'V2 BLEU': round(metrics['bleu'] * 100.0, 2),

            'V2 ROUGE-1-F': round(metrics['r1'] * 100.0, 2),
            'V2 ROUGE-2-F': round(metrics['r2'] * 100.0, 2),
            'V2 ROUGE-L-F': round(metrics['rl'] * 100.0, 2),
        })
    else:
        wandb.run.summary.update({
            'Count': metrics['count'],
            
            'BLEU': round(metrics["bleu"] * 100.0, 2),

            'ROUGE-1-F': round(metrics['r1'] * 100.0, 2),
            'ROUGE-2-F': round(metrics['r2'] * 100.0, 2),
            'ROUGE-L-F': round(metrics['rl'] * 100.0, 2),

            'Dup 1-grams': round(metrics["duplicate_ngrams"][1] * 100.0, 2),
            'Dup 2-grams': round(metrics["duplicate_ngrams"][2] * 100.0, 2),
            'Dup 3-grams': round(metrics["duplicate_ngrams"][3] * 100.0, 2)
        })
