from copilot.tools.claims_db import (
    APPEALABLE_DENIAL_CODES,
    ClaimNotFoundError,
    DENIAL_CODE_MEANINGS,
    REMEDIATION_PLAYBOOKS,
    get_connection,
)


def analyze_denial(claim_id: str) -> dict:
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM claims WHERE claim_id = ?", (claim_id,)).fetchone()
    finally:
        conn.close()

    if row is None:
        raise ClaimNotFoundError(f"no claim found with id {claim_id!r}")

    claim = dict(row)
    denial_code = claim["denial_code"]

    if denial_code is None:
        return {
            "claim_id": claim_id,
            "status": claim["status"],
            "denial_code": None,
            "message": "This claim was not denied - no denial analysis applies.",
        }

    return {
        "claim_id": claim_id,
        "payer": claim["payer"],
        "procedure_code": claim["procedure_code"],
        "service_date": claim["service_date"],
        "status": claim["status"],
        "denial_code": denial_code,
        "denial_code_meaning": DENIAL_CODE_MEANINGS.get(denial_code, "unknown code"),
        "is_appealable": denial_code in APPEALABLE_DENIAL_CODES,
        "appeal_filed": bool(claim["appeal_filed"]),
        "appeal_outcome": claim["appeal_outcome"],
        "recommended_actions": REMEDIATION_PLAYBOOKS.get(denial_code, []),
    }
