"""Prefect orchestration: download -> ingest -> validate, one month at a time.

Months run sequentially on purpose: each ingest already uses all CPU cores.
Failures are isolated per month; downloads retry; finished months are skipped.
"""
from __future__ import annotations

from pathlib import Path

from prefect import flow, get_run_logger, task

from .download import download
from .ingest import ingest_file
from .quality import QualityError, validate_month


@task(retries=3, retry_delay_seconds=30)
def t_download(month: str) -> str:
    return str(download(month))


@task
def t_ingest(path: str, workers: int | None) -> dict:
    return ingest_file(Path(path), workers=workers)


@task
def t_validate(month: str, memory_limit: str) -> dict:
    return validate_month(month, memory_limit=memory_limit)


@flow(name="metalens-monthly", log_prints=True)
def monthly_pipeline(months: list[str], workers: int | None = None,
                     delete_raw: bool = False, memory_limit: str = "6GB") -> dict:
    log = get_run_logger()
    summary: dict = {}
    for month in months:
        try:
            raw = t_download(month)
            manifest = t_ingest(raw, workers)
            report = t_validate(month, memory_limit)
            summary[month] = {"status": report["status"], "rows": report["rows"],
                              "games_per_second": manifest.get("games_per_second")}
            if delete_raw:  # only reached when validation passed
                Path(raw).unlink(missing_ok=True)
                log.info("Deleted raw dump for %s to save disk", month)
        except QualityError as e:
            summary[month] = {"status": "failed_quality", "error": str(e)}
            log.error("Quality gate failed for %s; raw file kept for inspection", month)
        except Exception as e:  # keep going with other months
            summary[month] = {"status": "error", "error": repr(e)}
            log.exception("Month %s failed", month)
    return summary
