"""Synthesizes a remediation plan for a denial pattern - deliberately a plain,
deterministic aggregation over the claims data plus a policy-doc lookup rather
than a second nested LLM call, so it stays testable and admin-gated like any
other tool (see governance/permissions.py: only admins can call this)."""
from copilot.rag.retriever import get_retriever
from copilot.tools.claims_db import DENIAL_CODE_MEANINGS, REMEDIATION_PLAYBOOKS, get_connection


def create_remediation_plan(
    *, payer: str | None = None, denial_code: str | None = None, procedure_code: str | None = None,
) -> dict:
    clauses, params = ["denial_code IS NOT NULL"], []
    if payer:
        clauses.append("payer = ?")
        params.append(payer)
    if denial_code:
        clauses.append("denial_code = ?")
        params.append(denial_code)
    if procedure_code:
        clauses.append("procedure_code = ?")
        params.append(procedure_code)
    where = f"WHERE {' AND '.join(clauses)}"

    conn = get_connection()
    try:
        affected_count = conn.execute(f"SELECT COUNT(*) FROM claims {where}", params).fetchone()[0]

        code_breakdown = conn.execute(
            f"SELECT denial_code, COUNT(*) as c FROM claims {where} GROUP BY denial_code ORDER BY c DESC", params,
        ).fetchall()

        procedure_breakdown = conn.execute(
            f"SELECT procedure_code, COUNT(*) as c FROM claims {where} GROUP BY procedure_code ORDER BY c DESC LIMIT 3",
            params,
        ).fetchall()
    finally:
        conn.close()

    dominant_code = denial_code or (code_breakdown[0]["denial_code"] if code_breakdown else None)

    policy_query_parts = [DENIAL_CODE_MEANINGS.get(dominant_code, "")] if dominant_code else []
    if payer:
        policy_query_parts.append(payer)
    if procedure_code:
        policy_query_parts.append(procedure_code)
    policy_hits = get_retriever().search(" ".join(policy_query_parts) or "denial remediation", k=2)

    return {
        "pattern_summary": {
            "payer": payer, "denial_code": denial_code, "procedure_code": procedure_code,
            "affected_claim_count": affected_count,
        },
        "denial_code_breakdown": [
            {"denial_code": row["denial_code"], "count": row["c"],
             "meaning": DENIAL_CODE_MEANINGS.get(row["denial_code"], "unknown code")}
            for row in code_breakdown
        ],
        "top_procedure_codes": [{"procedure_code": row["procedure_code"], "count": row["c"]} for row in procedure_breakdown],
        "recommended_actions": REMEDIATION_PLAYBOOKS.get(dominant_code, []),
        "policy_references": [hit["source"] for hit in policy_hits],
    }
