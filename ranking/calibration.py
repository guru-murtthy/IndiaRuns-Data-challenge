import numpy as np
from typing import List

class ScoreCalibration:
    """
    Utility to calibrate raw model scores into normalized ranges (0 to 1).
    Ensures monotonically increasing functions are used where appropriate.
    """
    
    @staticmethod
    def min_max_normalize(scores: List[float]) -> List[float]:
        """
        Normalizes a list of scores to the range [0.0, 1.0].
        If all scores are equal, returns a list of 1.0s.
        """
        if not scores:
            return []
            
        arr = np.array(scores, dtype=np.float32)
        min_val = np.min(arr)
        max_val = np.max(arr)
        
        if np.isclose(min_val, max_val):
            return [1.0] * len(scores)
            
        normalized = (arr - min_val) / (max_val - min_val)
        return normalized.tolist()

    @staticmethod
    def z_score_normalize(scores: List[float]) -> List[float]:
        """
        Normalizes a list of scores using Z-score (standardization).
        Note: The output range is not strictly [0, 1]. Use sigmoid after if needed.
        """
        if not scores:
            return []
            
        arr = np.array(scores, dtype=np.float32)
        std_val = np.std(arr)
        
        if np.isclose(std_val, 0.0):
            return [0.0] * len(scores)
            
        normalized = (arr - np.mean(arr)) / std_val
        return normalized.tolist()

    @staticmethod
    def temperature_scaling(scores: List[float], temperature: float = 1.0) -> List[float]:
        """
        Applies temperature scaling to the scores and passes them through a sigmoid
        to bound them between 0 and 1.
        Higher temperature = softer distribution. Lower temperature = peakier.
        """
        if not scores:
            return []
            
        if temperature <= 0:
            raise ValueError("Temperature must be greater than 0.")
            
        arr = np.array(scores, dtype=np.float32)
        scaled = arr / temperature
        # Sigmoid function for [0, 1] bounds
        probabilities = 1 / (1 + np.exp(-scaled))
        return probabilities.tolist()
