from copilot.tools.claims_db import DENIAL_CODE_MEANINGS, InvalidFilterError, get_connection

VALID_METRICS = ("denial_rate", "overturn_rate", "top_denial_codes", "claim_volume")


def _build_where(*, payer, procedure_code, denial_code, start_date, end_date, include_denial_code: bool) -> tuple[str, list]:
    clauses, params = [], []
    if payer:
        clauses.append("payer = ?")
        params.append(payer)
    if procedure_code:
        clauses.append("procedure_code = ?")
        params.append(procedure_code)
    if include_denial_code and denial_code:
        clauses.append("denial_code = ?")
        params.append(denial_code)
    if start_date:
        clauses.append("service_date >= ?")
        params.append(start_date)
    if end_date:
        clauses.append("service_date <= ?")
        params.append(end_date)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    return where, params


def calculate_denial_metrics(
    *,
    metric: str,
    payer: str | None = None,
    procedure_code: str | None = None,
    denial_code: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict:
    if metric not in VALID_METRICS:
        raise InvalidFilterError(f"invalid metric: {metric!r}, must be one of {VALID_METRICS}")

    conn = get_connection()
    try:
        if metric == "denial_rate":
            where, params = _build_where(
                payer=payer, procedure_code=procedure_code, denial_code=None,
                start_date=start_date, end_date=end_date, include_denial_code=False,
            )
            total = conn.execute(f"SELECT COUNT(*) FROM claims {where}", params).fetchone()[0]
            denied = conn.execute(
                f"SELECT COUNT(*) FROM claims {where} {'AND' if where else 'WHERE'} denial_code IS NOT NULL", params,
            ).fetchone()[0]
            rate = round(denied / total * 100, 1) if total else 0.0
            return {"metric": "denial_rate", "total_claims": total, "denied_claims": denied, "denial_rate_pct": rate}

        if metric == "overturn_rate":
            where, params = _build_where(
                payer=payer, procedure_code=procedure_code, denial_code=denial_code,
                start_date=start_date, end_date=end_date, include_denial_code=True,
            )
            appeal_clause = "appeal_filed = 1"
            where = f"{where} AND {appeal_clause}" if where else f"WHERE {appeal_clause}"
            appealed = conn.execute(f"SELECT COUNT(*) FROM claims {where}", params).fetchone()[0]
            overturned = conn.execute(
                f"SELECT COUNT(*) FROM claims {where} AND appeal_outcome = 'overturned'", params,
            ).fetchone()[0]
            rate = round(overturned / appealed * 100, 1) if appealed else 0.0
            return {
                "metric": "overturn_rate", "appealed_claims": appealed,
                "overturned_claims": overturned, "overturn_rate_pct": rate,
            }

        if metric == "top_denial_codes":
            where, params = _build_where(
                payer=payer, procedure_code=procedure_code, denial_code=None,
                start_date=start_date, end_date=end_date, include_denial_code=False,
            )
            where = f"{where} AND denial_code IS NOT NULL" if where else "WHERE denial_code IS NOT NULL"
            rows = conn.execute(
                f"SELECT denial_code, COUNT(*) as c FROM claims {where} GROUP BY denial_code ORDER BY c DESC LIMIT 5",
                params,
            ).fetchall()
            return {
                "metric": "top_denial_codes",
                "results": [
                    {"denial_code": row["denial_code"], "count": row["c"],
                     "meaning": DENIAL_CODE_MEANINGS.get(row["denial_code"], "unknown code")}
                    for row in rows
                ],
            }

        # claim_volume
        where, params = _build_where(
            payer=payer, procedure_code=procedure_code, denial_code=denial_code,
            start_date=start_date, end_date=end_date, include_denial_code=True,
        )
        row = conn.execute(
            f"SELECT COUNT(*) as n, COALESCE(SUM(billed_amount),0) as billed, "
            f"COALESCE(SUM(allowed_amount),0) as allowed FROM claims {where}", params,
        ).fetchone()
        return {
            "metric": "claim_volume", "total_claims": row["n"],
            "total_billed": round(row["billed"], 2), "total_allowed": round(row["allowed"], 2),
        }
    finally:
        conn.close()
