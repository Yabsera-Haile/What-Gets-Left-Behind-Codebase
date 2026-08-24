import sacrebleu


def _flat(x):
    return x[0] if isinstance(x, (list, tuple)) else x


def chrf_pp_agg(items):
    refs = [_flat(ref) for ref, pred in items]
    preds = [_flat(pred) for ref, pred in items]
    return sacrebleu.corpus_chrf(preds, [refs], word_order=2).score
