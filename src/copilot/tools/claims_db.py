"""Shared read-only access to the synthetic claims database, plus the small
reference tables (denial code meanings, remediation playbooks) every claims
tool draws from - kept in one place so `analyze_denial`, `calculate_denial_metrics`,
and `create_remediation_plan` never disagree with each other about what a code means.
"""
import sqlite3

from copilot.config import CLAIMS_DB_PATH
from copilot.sqlite_utils import connect as sqlite_connect

DENIAL_CODE_MEANINGS = {
    "CO-16": "Claim/service lacks information needed for adjudication",
    "CO-50": "Non-covered service - not deemed a medical necessity",
    "CO-97": "Benefit included in payment for another service already adjudicated",
    "PR-1": "Deductible amount - patient responsibility",
    "CO-197": "Precertification/authorization absent",
    "CO-29": "Timely filing limit expired",
}

# PR-1 is patient financial responsibility, not a payer error - not appealable by the provider.
APPEALABLE_DENIAL_CODES = frozenset(DENIAL_CODE_MEANINGS) - {"PR-1"}

VALID_PAYERS = frozenset({"Medicare", "Aetna", "BlueCross BlueShield", "UnitedHealthcare"})
VALID_STATUSES = frozenset({"paid", "denied", "appealed"})

REMEDIATION_PLAYBOOKS = {
    "CO-16": [
        "Audit intake forms for the missing field(s) most commonly absent on resubmission.",
        "Add a pre-submission validation check for required diagnosis/referring-NPI fields.",
    ],
    "CO-50": [
        "Require a letter of medical necessity and supporting documentation at time of scheduling, not after denial.",
        "Flag the affected procedure code for pre-submission clinical documentation review.",
    ],
    "CO-97": [
        "Review modifier usage and bundling rules for the affected procedure code.",
        "Add a claim-scrubber rule to catch the bundling conflict before submission.",
    ],
    "CO-197": [
        "Require prior-authorization verification at scheduling, before the visit occurs.",
        "For visits already rendered without authorization, pursue retroactive authorization within the payer's 30-day window and attach it to an appeal.",
    ],
    "CO-29": [
        "Shorten the internal claim-submission SLA to build buffer against the payer's timely filing window.",
        "Set up automated aging alerts for unsubmitted claims approaching the filing deadline.",
    ],
}


def get_connection() -> sqlite3.Connection:
    conn = sqlite_connect(CLAIMS_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


class ClaimNotFoundError(Exception):
    pass


class InvalidFilterError(Exception):
    pass
