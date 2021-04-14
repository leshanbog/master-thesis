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


def calc_metrics(refs, hyps, language, metric="all", meteor_jar=None):
    metrics = dict()
    metrics["count"] = len(hyps)
    metrics["ref_example"] = refs[-1]
    metrics["hyp_example"] = hyps[-1]

    many_refs = [[r] if r is not list else r for r in refs]

    metrics["bleu"] = corpus_bleu(many_refs, hyps)

    rouge = Rouge()
    scores = rouge.get_scores(hyps, refs, avg=True)
    metrics.update(scores)

    metrics["duplicate_ngrams"] = dict()
    metrics["duplicate_ngrams"].update(calc_duplicate_n_grams_rate(hyps))
    return metrics

def print_metrics(refs, hyps, language):
    import wandb
    metrics = calc_metrics(refs, hyps, language=language)

    wandb.run.summary.update({
        'Count': metrics['count'],
        
        'BLEU': round(metrics["bleu"] * 100.0, 2),

        'ROUGE-1-F': round(metrics["rouge-1"]['f'] * 100.0, 2),
        'ROUGE-2-F': round(metrics["rouge-2"]['f'] * 100.0, 2),
        'ROUGE-L-F': round(metrics["rouge-l"]['f'] * 100.0, 2),

        'Dup 1-grams': round(metrics["duplicate_ngrams"][1] * 100.0, 2),
        'Dup 2-grams': round(metrics["duplicate_ngrams"][2] * 100.0, 2),
        'Dup 3-grams': round(metrics["duplicate_ngrams"][3] * 100.0, 2)
    })