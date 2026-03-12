"""
Build training dataset from historical funding + klines.
Run: python -m ml.dataset
Generates ml/dataset.csv and optionally trains (python -m ml.dataset --train).
"""
from __future__ import annotations

import asyncio
import csv
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bybit_client import BybitClient
from config import load_config
import datetime as dt

from ml.features import _parse_float, _pct_return, extract_features_for_match, FEATURE_NAMES
from ml.train import train_and_save


def _parse_ts(ms: Any) -> Optional[int]:
    try:
        return int(ms)
    except (TypeError, ValueError):
        return None


async def build_dataset(client: BybitClient, symbols: List[str], min_5m_pct: float = 2.0) -> List[Dict[str, Any]]:
    """
    For each symbol, get funding history; for each negative funding event,
    check if 5m return at that time was >= min_5m_pct; if so, compute features and target (15m forward return).
    """
    rows = []
    five_min_ms = 5 * 60 * 1000
    fifteen_min_ms = 15 * 60 * 1000

    for sym in symbols:
        # Pull more history by paging backwards with end_time
        fund_hist: List[Dict[str, Any]] = []
        end_time: Optional[int] = None
        for _page in range(6):  # up to ~6*200 = 1200 records (depends on API)
            try:
                chunk = await client.get_funding_history(symbol=sym, limit=200, end_time=end_time)
            except Exception:
                break
            if not chunk:
                break
            fund_hist.extend(chunk)
            # oldest in this chunk is the last item (API returns newest first)
            oldest_ts = _parse_ts(chunk[-1].get("fundingRateTimestamp"))
            if oldest_ts is None:
                break
            end_time = oldest_ts - 1
            await asyncio.sleep(0.05)
        if not fund_hist:
            continue
        for i, rec in enumerate(fund_hist):
            rate = _parse_float(rec.get("fundingRate"))
            if rate is None or rate >= 0:
                continue
            ts_ms = _parse_ts(rec.get("fundingRateTimestamp"))
            if ts_ms is None:
                continue
            # Last closed 5m candle at ts_ms
            end_5 = (ts_ms // five_min_ms) * five_min_ms - five_min_ms
            start_5 = end_5 - 25 * five_min_ms
            try:
                k5 = await client.get_kline(sym, "5", limit=30, start_time=start_5, end_time=end_5 + 1)
            except Exception:
                continue
            if not k5 or len(k5) < 1:
                continue
            k = k5[0]
            if len(k) < 5:
                continue
            o5 = _parse_float(k[1])
            c5 = _parse_float(k[4])
            change_5m = _pct_return(o5, c5)
            if change_5m is None or change_5m < min_5m_pct:
                continue
            # Forward 15m return: close at ts_ms vs close 15m later
            end_15 = ts_ms + fifteen_min_ms
            start_15 = end_15 - 2 * fifteen_min_ms
            try:
                k15 = await client.get_kline(sym, "15", limit=2, start_time=start_15, end_time=end_15 + 1)
            except Exception:
                continue
            if not k15 or len(k15) < 2:
                continue
            close_now = c5
            next_15_start = (ts_ms // fifteen_min_ms + 1) * fifteen_min_ms
            next_15_end = next_15_start + fifteen_min_ms
            try:
                k15_next = await client.get_kline(sym, "15", limit=1, start_time=next_15_start, end_time=next_15_end + 1)
            except Exception:
                continue
            if not k15_next or len(k15_next) < 1 or len(k15_next[0]) < 5:
                continue
            close_15m_later = _parse_float(k15_next[0][4])
            if close_now is None or close_15m_later is None or close_now <= 0:
                continue
            target_pct = (close_15m_later - close_now) / close_now * 100.0
            label = 1 if target_pct > 0 else 0

            now_dt = dt.datetime.fromtimestamp((end_5 + five_min_ms) / 1000.0, tz=dt.timezone.utc)
            # Build full features (disable live-only sources to avoid leakage in training)
            feats = await extract_features_for_match(
                client,
                sym,
                current_price=float(close_now),
                change_5m=float(change_5m),
                funding_rate=float(rate),
                tz=dt.timezone.utc,
                now=now_dt,
                use_live_ticker=False,
                use_live_orderbook=False,
            )

            row: Dict[str, Any] = {
                "symbol": sym,
                "ts": ts_ms,
                "target_15m_pct": target_pct,
                "label": label,
                **{k: feats.get(k, 0.0) for k in FEATURE_NAMES},
            }
            rows.append(row)
        await asyncio.sleep(0.1)

    return rows


async def main() -> None:
    config = load_config()
    async with BybitClient(config.bybit) as client:
        instruments = await client.get_instruments_info()
        symbols = []
        for inst in instruments:
            if inst.get("status") != "Trading":
                continue
            ct = (inst.get("contractType") or "").lower()
            if ct not in ("perpetual", "linearperpetual"):
                continue
            s = inst.get("symbol")
            if s:
                symbols.append(s)

        # Allow overriding symbol cap (default: all)
        cap_raw = (os.getenv("DATASET_SYMBOL_CAP", "") or "").strip()
        if cap_raw:
            try:
                cap = int(cap_raw)
                if cap > 0:
                    symbols = symbols[:cap]
            except ValueError:
                pass

        # Allow overriding threshold for training (default keeps current scan logic)
        min_change = float(os.getenv("DATASET_MIN_5M_PCT", str(2.0)))

        print(f"Building dataset for {len(symbols)} symbols...")
        rows = await build_dataset(client, symbols, min_5m_pct=min_change)
        print(f"Got {len(rows)} samples.")

    dataset_path = Path(__file__).resolve().parent / "dataset.csv"
    with open(dataset_path, "w", newline="", encoding="utf-8") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Wrote {dataset_path}")

    if "--train" in sys.argv and rows:
        train_and_save(dataset_path)
    elif rows:
        print("Run with --train to train and save model.")


if __name__ == "__main__":
    asyncio.run(main())
