from typing import List, Dict, Any
from pydantic import BaseModel

class RankedCandidate(BaseModel):
    """
    Final output representation for a ranked candidate.
    """
    candidate_id: str
    rank: int
    score: float

class FinalRankGenerator:
    """
    Assigns final ranks to candidates and handles tie-breaking.
    """
    
    @staticmethod
    def generate(sorted_candidates: List[Dict[str, Any]]) -> List[RankedCandidate]:
        """
        Generates the final ranking from a sorted list of candidates.
        Ties in score are broken deterministically using candidate_id.
        
        Args:
            sorted_candidates: A list of candidate dicts containing 'rerank_score'.
            
        Returns:
            A list of RankedCandidate objects.
        """
        # Ensure secondary sort by candidate_id for deterministic tie breaking
        # Since Python's sort is stable, we sort by ID first, then by score descending.
        
        def safe_get_id(c):
            return str(c.get("candidate_id", c.get("id", "")))
            
        sorted_candidates = sorted(sorted_candidates, key=lambda c: safe_get_id(c))
        sorted_candidates = sorted(sorted_candidates, key=lambda c: float(c.get("rerank_score", 0.0)), reverse=True)
        
        final_list = []
        for i, cand in enumerate(sorted_candidates, start=1):
            cand_id = safe_get_id(cand)
            score = float(cand.get("rerank_score", 0.0))
            
            final_list.append(
                RankedCandidate(
                    candidate_id=cand_id,
                    rank=i,
                    score=score
                )
            )
            
        return final_list
