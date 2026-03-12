import datetime as dt
from zoneinfo import ZoneInfo

from telegram.ext import Application

from bybit_client import BybitClient
from config import AppConfig, load_config
from scanner import ScanSummary, run_scan
from scheduler import setup_scheduler
from telegram_handler import build_application

try:
    from ml.predict import predict_long_probability
except ImportError:
    predict_long_probability = None  # type: ignore[misc, assignment]


def _get_timezone(config: AppConfig) -> dt.tzinfo:
    tz_name = (config.timezone or "UTC").strip()
    if tz_name.upper() == "UTC":
        return dt.timezone.utc
    try:
        return ZoneInfo(tz_name)
    except Exception:  # noqa: BLE001
        # Fallback to local timezone if invalid name
        return dt.datetime.now().astimezone().tzinfo or dt.timezone.utc


async def run_scan_once(bot_data: dict, *, mode: str = "full") -> ScanSummary:
    """
    Helper used by scheduler and /scan command.
    """
    config: AppConfig = bot_data["config"]
    client: BybitClient = bot_data["bybit_client"]
    tz = _get_timezone(config)
    if mode == "rise_only":
        summary = await run_scan(
            client,
            config.criteria,
            tz,
            require_actual_funding_negative=False,
            direction="up",
        )
    elif mode == "fall_only":
        summary = await run_scan(
            client,
            config.criteria,
            tz,
            require_actual_funding_negative=False,
            direction="down",
        )
    else:
        summary = await run_scan(
            client,
            config.criteria,
            tz,
            require_actual_funding_negative=True,
            direction="up",
        )
    await _enrich_matches_with_ml(bot_data, summary)
    return summary


async def _enrich_matches_with_ml(bot_data: dict, summary: ScanSummary) -> None:
    """Add long_prob to each match if ML model is available."""
    if predict_long_probability is None:
        return
    config: AppConfig = bot_data["config"]
    client: BybitClient = bot_data["bybit_client"]
    tz = _get_timezone(config)
    for m in summary.matches:
        try:
            prob = await predict_long_probability(client, m, tz, config)
            if prob is not None:
                m["long_prob"] = prob
        except Exception:  # noqa: BLE001
            pass


async def _post_init(application: Application) -> None:
    """
    Called by python-telegram-bot once the application is initialized,
    but before polling starts. We use this to setup the scheduler and DB.
    """
    await setup_scheduler(application.bot_data)


def main() -> None:
    config = load_config()
    client = BybitClient(config.bybit)

    application: Application = build_application(config, client)

    # Make application available in bot_data for scheduler and handlers
    application.bot_data["application"] = application

    # Attach post_init hook so scheduler is initialized in the same event loop
    application.post_init = _post_init  # type: ignore[assignment]

    # Blocking call; manages its own event loop internally
    application.run_polling()


if __name__ == "__main__":
    main()

