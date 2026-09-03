"""Structured (not raw-SQL) claims lookup.

The model supplies typed filters, never a SQL string - the tool builds a fully
parameterized query internally. This is the same "never let the model touch
a real interpreter directly" posture as the old calculator's AST allowlist:
it's both safer (no injection surface) and more realistic, since production
text-to-SQL systems very rarely give the model a raw SQL execution tool.
"""
from copilot.tools.claims_db import InvalidFilterError, get_connection

MAX_RETURNED_ROWS = 20


def query_claims(
    *,
    payer: str | None = None,
    denial_code: str | None = None,
    procedure_code: str | None = None,
    status: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 10,
) -> dict:
    clauses = []
    params: list = []

    if payer:
        clauses.append("payer = ?")
        params.append(payer)
    if denial_code:
        clauses.append("denial_code = ?")
        params.append(denial_code)
    if procedure_code:
        clauses.append("procedure_code = ?")
        params.append(procedure_code)
    if status:
        if status not in ("paid", "denied", "appealed"):
            raise InvalidFilterError(f"invalid status: {status!r}")
        clauses.append("status = ?")
        params.append(status)
    if start_date:
        clauses.append("service_date >= ?")
        params.append(start_date)
    if end_date:
        clauses.append("service_date <= ?")
        params.append(end_date)

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    capped_limit = max(1, min(int(limit), MAX_RETURNED_ROWS))

    conn = get_connection()
    try:
        total_count = conn.execute(f"SELECT COUNT(*) FROM claims {where}", params).fetchone()[0]
        rows = conn.execute(
            f"SELECT * FROM claims {where} ORDER BY service_date DESC LIMIT ?",
            [*params, capped_limit],
        ).fetchall()
    finally:
        conn.close()

    return {
        "total_matching_count": total_count,
        "returned_count": len(rows),
        "claims": [dict(row) for row in rows],
    }
