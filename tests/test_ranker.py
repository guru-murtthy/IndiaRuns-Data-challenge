import pytest
from ranking.feature_vector import FeatureVector
from ranking.scorer import RankingScorer
from ranking.calibration import ScoreCalibration

class TestRankerAndCalibration:

    def test_fallback_scorer(self):
        scorer = RankingScorer(weights={"f1": 1.0, "f2": 0.5})
        fv = FeatureVector(features={"f1": 2.0, "f2": 4.0, "f3": 10.0})
        
        score = scorer.score(fv)
        # 2.0*1.0 + 4.0*0.5 + 10.0*0 = 4.0
        assert score == 4.0

    def test_min_max_calibration(self):
        scores = [1.0, 5.0, 9.0]
        norm = ScoreCalibration.min_max_normalize(scores)
        assert norm == [0.0, 0.5, 1.0]

    def test_z_score_calibration(self):
        scores = [0.0, 10.0, 20.0]
        norm = ScoreCalibration.z_score_normalize(scores)
        # mean = 10, std = sqrt(200/3) = ~8.1649
        assert norm[1] == 0.0 # middle element should be exactly 0
        assert norm[0] < 0
        assert norm[2] > 0
        
    def test_temperature_scaling(self):
        scores = [-10.0, 0.0, 10.0]
        norm = ScoreCalibration.temperature_scaling(scores, temperature=1.0)
        assert norm[1] == 0.5
        assert norm[0] < 0.5
        assert norm[2] > 0.5
