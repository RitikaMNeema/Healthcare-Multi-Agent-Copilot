"""Deterministic generator for the synthetic healthcare claims database.

Re-running this script (same seed) always produces the same data, which is
what lets the eval harness assert exact SQL-accuracy ground truths against it.

Run: python data/generate_claims.py
"""
import os
import random
import sqlite3
from datetime import date, timedelta

SEED = 42
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "claims.db")
NUM_CLAIMS = 520

PAYERS = ["Medicare", "Aetna", "BlueCross BlueShield", "UnitedHealthcare"]

PROCEDURES = {
    "99213": "Office visit, established patient, low complexity",
    "99214": "Office visit, established patient, moderate complexity",
    "93000": "Electrocardiogram, routine",
    "80053": "Comprehensive metabolic panel",
    "71046": "Chest X-ray, 2 views",
    "29881": "Knee arthroscopy with meniscectomy",
    "97110": "Therapeutic exercise, 15 minutes",
    "J3490": "Unclassified drug",
}

DENIAL_CODES = {
    "CO-16": "Claim/service lacks information needed for adjudication",
    "CO-50": "Non-covered service - not deemed a medical necessity",
    "CO-97": "Benefit included in payment for another service already adjudicated",
    "PR-1": "Deductible amount - patient responsibility",
    "CO-197": "Precertification/authorization absent",
    "CO-29": "Timely filing limit expired",
}

# Appeal overturn probability by denial code - reflects that some denials are
# easy to cure on appeal (missing info) and some almost never are (timely filing).
OVERTURN_PROBABILITY = {
    "CO-16": 0.60,
    "CO-50": 0.30,
    "CO-97": 0.20,
    "CO-197": 0.40,
    "CO-29": 0.10,
    # PR-1 is patient financial responsibility, not a payer error - never appealed.
}

START_DATE = date(2024, 1, 1)
END_DATE = date(2024, 6, 30)
DATE_SPAN_DAYS = (END_DATE - START_DATE).days


def _random_date(rng: random.Random) -> date:
    return START_DATE + timedelta(days=rng.randint(0, DATE_SPAN_DAYS))


def _billed_amount(rng: random.Random, procedure_code: str) -> float:
    base = {
        "99213": 120, "99214": 175, "93000": 90, "80053": 60,
        "71046": 140, "29881": 4200, "97110": 55, "J3490": 310,
    }[procedure_code]
    return round(base * rng.uniform(0.85, 1.25), 2)


def _choose_denial_code(rng: random.Random, payer: str, procedure_code: str) -> str | None:
    """Engineered skew so a handful of clean, checkable facts exist in the data:

    - Aetna denies 97110 (therapeutic exercise) heavily for missing prior auth (CO-197).
    - UnitedHealthcare has an elevated timely-filing (CO-29) denial rate overall.
    - Medicare denies knee arthroscopy (29881) somewhat more for medical necessity (CO-50).
    - Baseline denial rate elsewhere is ~14%, spread across the remaining codes.
    """
    if payer == "Aetna" and procedure_code == "97110":
        return "CO-197" if rng.random() < 0.70 else None

    if payer == "UnitedHealthcare" and rng.random() < 0.22:
        return "CO-29"

    if payer == "Medicare" and procedure_code == "29881" and rng.random() < 0.35:
        return "CO-50"

    if rng.random() < 0.14:
        return rng.choice(["CO-16", "CO-50", "CO-97", "PR-1"])

    return None


def generate_claims(seed: int = SEED, num_claims: int = NUM_CLAIMS) -> list[dict]:
    rng = random.Random(seed)
    claims = []

    for i in range(1, num_claims + 1):
        claim_id = f"CLM-{i:06d}"
        patient_id = f"PT-{rng.randint(1, 180):05d}"
        payer = rng.choice(PAYERS)
        procedure_code = rng.choice(list(PROCEDURES))
        service_date = _random_date(rng)
        billed_amount = _billed_amount(rng, procedure_code)

        denial_code = _choose_denial_code(rng, payer, procedure_code)

        if denial_code is None:
            status = "paid"
            allowed_amount = round(billed_amount * rng.uniform(0.55, 0.85), 2)
            denial_reason = None
            appeal_filed = 0
            appeal_outcome = None
            days_to_resolution = rng.randint(10, 30)
        else:
            status = "denied"
            allowed_amount = 0.0
            denial_reason = DENIAL_CODES[denial_code]
            days_to_resolution = rng.randint(10, 30)

            overturn_p = OVERTURN_PROBABILITY.get(denial_code)
            if overturn_p is None:
                appeal_filed = 0
                appeal_outcome = None
            else:
                appeal_filed = 1 if rng.random() < 0.55 else 0
                if appeal_filed:
                    if rng.random() < overturn_p:
                        appeal_outcome = "overturned"
                        status = "appealed"
                        allowed_amount = round(billed_amount * rng.uniform(0.55, 0.85), 2)
                    else:
                        appeal_outcome = "upheld"
                        status = "appealed"
                    days_to_resolution += rng.randint(30, 90)
                else:
                    appeal_outcome = None

        claims.append({
            "claim_id": claim_id,
            "patient_id": patient_id,
            "payer": payer,
            "procedure_code": procedure_code,
            "procedure_description": PROCEDURES[procedure_code],
            "service_date": service_date.isoformat(),
            "billed_amount": billed_amount,
            "allowed_amount": allowed_amount,
            "status": status,
            "denial_code": denial_code,
            "denial_reason": denial_reason,
            "appeal_filed": appeal_filed,
            "appeal_outcome": appeal_outcome,
            "days_to_resolution": days_to_resolution,
        })

    return claims


SCHEMA = """
CREATE TABLE IF NOT EXISTS claims (
    claim_id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL,
    payer TEXT NOT NULL,
    procedure_code TEXT NOT NULL,
    procedure_description TEXT NOT NULL,
    service_date TEXT NOT NULL,
    billed_amount REAL NOT NULL,
    allowed_amount REAL NOT NULL,
    status TEXT NOT NULL,
    denial_code TEXT,
    denial_reason TEXT,
    appeal_filed INTEGER NOT NULL,
    appeal_outcome TEXT,
    days_to_resolution INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_claims_payer ON claims(payer);
CREATE INDEX IF NOT EXISTS idx_claims_denial_code ON claims(denial_code);
CREATE INDEX IF NOT EXISTS idx_claims_procedure ON claims(procedure_code);
CREATE INDEX IF NOT EXISTS idx_claims_status ON claims(status);
"""


def write_db(claims: list[dict], db_path: str = DB_PATH) -> None:
    if os.path.exists(db_path):
        os.remove(db_path)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    conn.executemany(
        "INSERT INTO claims VALUES (:claim_id, :patient_id, :payer, :procedure_code, "
        ":procedure_description, :service_date, :billed_amount, :allowed_amount, :status, "
        ":denial_code, :denial_reason, :appeal_filed, :appeal_outcome, :days_to_resolution)",
        claims,
    )
    conn.commit()
    conn.close()


if __name__ == "__main__":
    rows = generate_claims()
    write_db(rows)
    print(f"Wrote {len(rows)} claims to {DB_PATH}")
