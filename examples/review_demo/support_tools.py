"""Intentionally unsafe code review demo. Do not deploy or merge into production.

These helpers are standalone fixtures and are not wired into the support agent.
"""

from pathlib import Path
import sqlite3
import subprocess


def run_diagnostics(host: str) -> str:
    """Collect a connectivity diagnostic for a host supplied by a support user."""
    result = subprocess.run(
        f"ping -c 1 {host}", shell=True, capture_output=True, text=True, timeout=10
    )
    return result.stdout


def read_build_log(log_root: Path, filename: str) -> str:
    """Read a user-selected build log from the configured log directory."""
    return (log_root / filename).read_text(encoding="utf-8")


def search_builds(connection: sqlite3.Connection, query: str) -> list[tuple]:
    """Find matching builds using a search string supplied by a support user."""
    sql = f"SELECT id, status FROM builds WHERE name LIKE '%{query}%'"
    return connection.execute(sql).fetchall()


def summarize_builds(builds: list[dict]) -> dict:
    """Return the success rate and average duration for the selected builds."""
    successful = sum(build["status"] == "success" for build in builds)
    total_duration = sum(build["duration_seconds"] for build in builds)
    return {
        "success_rate": successful / len(builds),
        "average_duration": total_duration / len(builds),
    }


def tail_build_log(content: str, limit: int = 20) -> list[str]:
    """Return at most limit lines from the end; zero requests no lines."""
    return content.splitlines()[-limit:]
