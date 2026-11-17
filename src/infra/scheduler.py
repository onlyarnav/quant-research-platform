"""
Job scheduler for automated recurring pipeline executions.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any, Callable

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from config.settings import settings
from src.infra.pipeline_models import PipelineRunReport
from src.infra.pipeline_orchestrator import PipelineOrchestrator
from src.infra.pipeline_steps import PipelineContext
from src.utils.logger import get_logger

logger = get_logger(__name__)

TIME_FORMAT_RE = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")


def parse_run_time(time_str: str) -> tuple[int, int]:
    """Parse and validate 'HH:MM' 24-hour time string into (hour, minute)."""
    match = TIME_FORMAT_RE.match(time_str.strip())
    if not match:
        raise ValueError(
            f"Invalid PIPELINE_RUN_TIME '{time_str}'. Expected 24h format 'HH:MM'."
        )
    return int(match.group(1)), int(match.group(2))


def run_once(
    context: PipelineContext | None = None,
    orchestrator: PipelineOrchestrator | None = None,
) -> PipelineRunReport:
    """
    Execute single pipeline batch run immediately.

    Suitable for cron jobs, Windows Task Scheduler, or direct CLI execution.
    """
    orc = orchestrator or PipelineOrchestrator()
    if context is None:
        run_id = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_pipeline"
        context = PipelineContext(
            run_id=run_id,
            asset_class="equity",
            timeframe="daily",
        )

    logger.info("Triggering scheduled run_once with run_id=%s", context.run_id)
    return orc.run(context)


def build_scheduler(
    run_time: str | None = None,
    timezone: str | None = None,
    job: Callable[[], Any] | None = None,
) -> BlockingScheduler:
    """
    Construct a BlockingScheduler configured with daily CronTrigger.

    Args:
        run_time: Execution time in 'HH:MM' format. Defaults to settings.PIPELINE_RUN_TIME.
        timezone: Execution timezone string. Defaults to settings.PIPELINE_TIMEZONE.
        job: Callable to execute. Defaults to run_once.

    Returns:
        Configured, unstarted BlockingScheduler instance.
    """
    time_str = run_time or settings.PIPELINE_RUN_TIME
    tz_str = timezone or settings.PIPELINE_TIMEZONE
    hour, minute = parse_run_time(time_str)

    target_job = job or run_once

    def _safe_job_wrapper() -> None:
        try:
            logger.info("Scheduled trigger firing at %s %s", time_str, tz_str)
            target_job()
        except Exception as exc:
            logger.exception(
                "Scheduled pipeline job failed with unhandled exception: %s", exc
            )

    scheduler = BlockingScheduler()
    trigger = CronTrigger(hour=hour, minute=minute, timezone=tz_str)
    scheduler.add_job(_safe_job_wrapper, trigger=trigger, id="daily_pipeline")
    return scheduler


def start_background_scheduler(
    run_time: str | None = None,
    timezone: str | None = None,
    job: Callable[[], Any] | None = None,
) -> None:
    """
    Build and start the blocking cron scheduler loop with graceful SIGTERM/SIGINT.
    """
    import signal
    import sys

    scheduler = build_scheduler(run_time=run_time, timezone=timezone, job=job)
    logger.info(
        "Starting background scheduler configured for %s (%s)",
        run_time or settings.PIPELINE_RUN_TIME,
        timezone or settings.PIPELINE_TIMEZONE,
    )

    def _handle_exit_signal(signum: int, frame: Any) -> None:
        logger.info("Signal %s received. Shutting down scheduler gracefully.", signum)
        if scheduler.running:
            scheduler.shutdown(wait=False)
        sys.exit(0)

    try:
        signal.signal(signal.SIGTERM, _handle_exit_signal)
        signal.signal(signal.SIGINT, _handle_exit_signal)
    except (ValueError, AttributeError):
        pass

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler received exit signal. Shutting down gracefully.")
        if scheduler.running:
            scheduler.shutdown()
