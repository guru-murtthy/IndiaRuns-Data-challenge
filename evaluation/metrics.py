import numpy as np

def ndcg_at_k(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gain = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    dcg = np.sum(gain / discounts)
    
    ideal_order = np.argsort(y_true)[::-1]
    ideal_true = np.take(y_true, ideal_order[:k])
    ideal_gain = 2 ** ideal_true - 1
    idcg = np.sum(ideal_gain / discounts)
    
    return dcg / idcg if idcg > 0 else 0.0

def average_precision(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    
    num_hits = 0.0
    score = 0.0
    for i, p in enumerate(y_true):
        if p > 0:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
            
    return score / max(1.0, np.sum(y_true > 0))
