from typing import List, Dict, Any
import numpy as np
from pydantic import BaseModel, Field, ConfigDict

class FeatureVector(BaseModel):
    """
    Represents a computed feature vector for a candidate against a JD.
    """
    features: Dict[str, float] = Field(default_factory=dict)
    
    model_config = ConfigDict(extra="forbid")

    @property
    def feature_names(self) -> List[str]:
        return sorted(self.features.keys())

    def to_numpy(self) -> np.ndarray:
        if not self.features:
            return np.array([], dtype=np.float32)
        names = self.feature_names
        vector = [self.features[name] for name in names]
        return np.array(vector, dtype=np.float32)

    def to_dict(self) -> Dict[str, float]:
        return self.features
