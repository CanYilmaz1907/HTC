"""
Load trained model and predict P(Long) for a scan match.
If no model exists, returns None (no ML line in notification).
"""
from __future__ import annotations

import datetime as dt
from typing import Any, Dict, Optional

import numpy as np

from bybit_client import BybitClient
from config import AppConfig
from ml.features import extract_features_for_match, feature_vector_for_model
from ml.train import load_model_and_scaler


def load_predictor() -> bool:
    """Return True if model and scaler are available."""
    clf, scaler, _ = load_model_and_scaler()
    return clf is not None and scaler is not None


async def predict_long_probability(
    client: BybitClient,
    match: Dict[str, Any],
    tz: dt.tzinfo,
    config: AppConfig,
) -> Optional[float]:
    """
    For a scan match (symbol, last_price, price_change_pct, funding_rate, ...),
    compute ML features and return P(Long) in [0, 1], or None if no model / error.
    """
    clf, scaler, feature_names = load_model_and_scaler()
    if clf is None or scaler is None:
        return None

    symbol = match.get("symbol")
    last_price = match.get("last_price")
    change_5m = match.get("price_change_pct")
    funding_rate = match.get("funding_rate")
    if not symbol or last_price is None or change_5m is None or funding_rate is None:
        return None

    try:
        features = await extract_features_for_match(
            client,
            symbol,
            current_price=float(last_price),
            change_5m=float(change_5m),
            funding_rate=float(funding_rate),
            tz=tz,
        )
    except Exception:
        return None

    # Build vector in model's feature order
    vec = np.array([[features.get(k, 0.0) for k in feature_names]], dtype=np.float64)
    vec = scaler.transform(vec)
    proba = clf.predict_proba(vec)[0]
    # clf.classes_ is [0, 1] for sklearn; proba[1] = P(Long)
    if len(clf.classes_) == 2 and clf.classes_[1] == 1:
        return float(proba[1])
    return float(proba[0])
