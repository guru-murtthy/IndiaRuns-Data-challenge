import numpy as np

def retrieve_top_k(index, query_emb, k, candidate_ids):
    """
    Search vector index and return similarity dictionary and order list of matching candidate IDs.
    """
    search_k = min(k, index.ntotal)
    similarities, indices = index.search(query_emb.astype('float32'), search_k)
    
    retrieved_cids = []
    sim_dict = {}
    for idx, sim in zip(indices[0], similarities[0]):
        if idx == -1 or idx >= len(candidate_ids):
            continue
        cid = candidate_ids[idx]
        if cid not in sim_dict:
            sim_dict[cid] = float(sim)
            retrieved_cids.append(cid)
            
    return retrieved_cids, sim_dict
