"""
ML pipeline for alarm direction: Long vs Short probability.
- features: extract from klines + funding + ticker
- dataset: build from historical data
- train: train classifier, save model
- predict: load model, predict at scan time
"""

from ml.predict import load_predictor, predict_long_probability

__all__ = ["load_predictor", "predict_long_probability"]
