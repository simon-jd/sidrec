"""Estimate the warehouse credits used by this repository's Snowflake queries.

    ~/.cache/portfolio-snowflake/.venv-mljobs/bin/python scripts/warehouse_credits.py

PORTFOLIO_XS is shared with other jobs that use the same service user, so the hourly
WAREHOUSE_METERING_HISTORY rows are not ours alone. This script records three things:
- our queries (query tag sidrec-r1*) from INFORMATION_SCHEMA.QUERY_HISTORY_BY_USER;
- two estimates built from them:
  - execution only: the sum of execution times, at 1 credit per hour for XSMALL;
  - an upper bound: the union of our busy intervals, each extended by the 60 s auto-suspend window, as if no
    other job kept the warehouse running;
- the account's metering rows for PORTFOLIO_XS over the same hours, and per-query attributed credits from
  ACCOUNT_USAGE.QUERY_ATTRIBUTION_HISTORY. Both views lag by up to a few hours, so rows may still be missing.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "r1" / "warehouse_credits.json"
SF_HOME = Path(os.environ.get("SIDREC_SF_HOME", "~/.cache/portfolio-snowflake")).expanduser()
WAREHOUSE = "PORTFOLIO_XS"
CREDITS_PER_HOUR = 1.0  # XSMALL
AUTO_SUSPEND_S = 60


def rows(session, sql: str) -> list[dict]:
    return [r.as_dict() for r in session.sql(sql).collect()]


def union_seconds(intervals: list[tuple[datetime, datetime]]) -> float:
    total, cur_start, cur_end = 0.0, None, None
    for start, end in sorted(intervals):
        if cur_end is None or start > cur_end:
            if cur_end is not None:
                total += (cur_end - cur_start).total_seconds()
            cur_start, cur_end = start, end
        else:
            cur_end = max(cur_end, end)
    if cur_end is not None:
        total += (cur_end - cur_start).total_seconds()
    return total


def main() -> None:
    sys.path.insert(0, str(SF_HOME))
    import sf

    s = sf.session()
    s.sql("ALTER SESSION SET QUERY_TAG = 'sidrec-r1-credits'").collect()
    s.sql("ALTER SESSION SET TIMEZONE = 'UTC'").collect()
    ours = rows(
        s,
        """SELECT query_id, query_tag, warehouse_name, start_time, end_time, execution_time,
                  credits_used_cloud_services
           FROM TABLE(INFORMATION_SCHEMA.QUERY_HISTORY_BY_USER(
                  END_TIME_RANGE_START => DATEADD('hour', -12, CURRENT_TIMESTAMP()), RESULT_LIMIT => 10000))
           WHERE query_tag LIKE 'sidrec-r1%' AND execution_status = 'SUCCESS'
           ORDER BY start_time""",
    )
    on_wh = [r for r in ours if r["WAREHOUSE_NAME"] == WAREHOUSE]
    exec_s = sum(int(r["EXECUTION_TIME"]) for r in on_wh) / 1000
    busy = [(r["START_TIME"], r["END_TIME"] + timedelta(seconds=AUTO_SUSPEND_S)) for r in on_wh]
    upper_s = union_seconds(busy)
    cloud = sum(float(r["CREDITS_USED_CLOUD_SERVICES"] or 0) for r in ours)
    first, last = min(r["START_TIME"] for r in ours), max(r["END_TIME"] for r in ours)

    metering = rows(
        s,
        f"""SELECT start_time, end_time, credits_used, credits_used_compute, credits_used_cloud_services
            FROM SNOWFLAKE.ACCOUNT_USAGE.WAREHOUSE_METERING_HISTORY
            WHERE warehouse_name = '{WAREHOUSE}'
              AND start_time >= DATE_TRUNC('hour', '{first.isoformat()}'::TIMESTAMP_TZ)
              AND start_time <= '{last.isoformat()}'::TIMESTAMP_TZ
            ORDER BY start_time""",
    )
    ids = ", ".join(f"'{r['QUERY_ID']}'" for r in on_wh)
    try:
        attributed = rows(
            s,
            f"""SELECT query_id, credits_attributed_compute
                FROM SNOWFLAKE.ACCOUNT_USAGE.QUERY_ATTRIBUTION_HISTORY WHERE query_id IN ({ids})""",
        )
        attribution = {
            "queries_found": len(attributed),
            "credits_attributed_compute": round(
                sum(float(r["CREDITS_ATTRIBUTED_COMPUTE"]) for r in attributed), 6
            ),
        }
    except Exception as exc:  # the view may not be readable with this role
        attribution = {"error": str(exc).splitlines()[0][:200]}
    s.close()

    result = {
        "measured_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "warehouse": WAREHOUSE,
        "size": "XSMALL",
        "credits_per_hour": CREDITS_PER_HOUR,
        "window_utc": [first.isoformat(), last.isoformat()],
        "our_queries": len(ours),
        "our_queries_on_warehouse": len(on_wh),
        "execution_seconds": round(exec_s, 1),
        "estimate_execution_only_credits": round(exec_s / 3600 * CREDITS_PER_HOUR, 4),
        "estimate_upper_bound_credits": round(upper_s / 3600 * CREDITS_PER_HOUR, 4),
        "upper_bound_seconds": round(upper_s, 1),
        "cloud_services_credits": round(cloud, 6),
        "query_attribution_history": attribution,
        "warehouse_metering_history_shared": [{k.lower(): str(v) for k, v in r.items()} for r in metering],
        "note": "PORTFOLIO_XS is shared; metering rows include other jobs and may lag by up to 3 hours.",
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
