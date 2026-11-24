"""
compliance_checker.py
---------------------
Hybrid decision:

  Layer 1  ML model   -> predicts compliant / non-compliant from label features
  Layer 2  Date rules -> explicit checks on the extracted dates (expired,
                         manufacturing date in the future, expiry before mfg)

  Final    If OCR read too little      -> UNABLE TO ANALYSE
           Else if a date rule fails   -> NON-COMPLIANT (date reason shown)
           Else                        -> the ML model's prediction

This is a demonstration of a configured set of checks, NOT a legal assessment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import joblib
import pandas as pd

from feature_extractor import FEATURE_COLUMNS, LabelFields, build_features

MODEL_PATH = Path(__file__).parent / "models" / "compliance_model.pkl"

MIN_WORDS = 8          # below this the OCR output is treated as unreadable
MIN_FIELDS = 2         # at least this many of the six declarations must be found

FIELD_NAMES = {
    "mrp_present": "MRP",
    "quantity_present": "Net quantity",
    "manufacturer_present": "Manufacturer",
    "address_present": "Address",
    "date_present": "Manufacturing / packing date",
    "customer_care_present": "Customer care details",
}


@dataclass
class Issue:
    kind: str      # "expired" | "missing" | "date" | "model" | "unable"
    title: str
    detail: str


@dataclass
class Verdict:
    status: str                      # "COMPLIANT" | "NON-COMPLIANT" | "UNABLE"
    reason: str
    features: dict
    ml_label: str | None = None
    ml_probability: float | None = None   # probability of NON-compliant
    model_name: str | None = None
    issues: list[Issue] = field(default_factory=list)
    mfg_date: date | None = None
    expiry_date: date | None = None
    date_rule_failed: bool = False


def load_model(path: Path = MODEL_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError("Model not found. Run:  python train_model.py")
    return joblib.load(path)


def check_label(fields: LabelFields, bundle: dict, today: date | None = None,
                mfg_override: date | None = None, expiry_override: date | None = None) -> Verdict:
    today = today or date.today()
    mfg = mfg_override or fields.dates.get("mfg_date")
    expiry = expiry_override or fields.dates.get("expiry_date")
    feats = build_features(fields, today, mfg_date=mfg, expiry_date=expiry)
    found = sum(feats[f] for f in FIELD_NAMES)

    # --- Gate: is there enough readable text to say anything at all? -------
    if fields.word_count < MIN_WORDS or found < MIN_FIELDS:
        return Verdict(
            status="UNABLE",
            reason="Unable to confidently analyse this label. Too little label information "
                   "was read - try a sharper, well-lit, straight-on photo.",
            features=feats, mfg_date=mfg, expiry_date=expiry,
            issues=[Issue("unable", "Low OCR coverage",
                          f"Read {fields.word_count} words and found {found} of 6 declarations.")],
        )

    # --- Gate: the label declares a date, but OCR could not read one -------
    fragments = fields.dates.get("fragments") or []
    if not mfg and not expiry and (fields.dates.get("date_declared") or fragments):
        seen = (" OCR only read fragments: " + ", ".join(f"'{f}'" for f in fragments[:4]) + "."
                if fragments else "")
        return Verdict(
            status="UNABLE",
            reason="The label declares a manufacturing / use-by date, but it could not be read "
                   "reliably, so no result is guessed. Enter the dates under manual correction.",
            features=feats, mfg_date=mfg, expiry_date=expiry,
            issues=[Issue("unable", "Declared date not readable",
                          "Date wording (e.g. 'Use By') was found but no complete date was read."
                          + seen + " No date is guessed.")],
        )

    # --- Layer 1: ML prediction -------------------------------------------
    model = bundle["model"]
    X = pd.DataFrame([feats])[FEATURE_COLUMNS]
    p_non = float(model.predict_proba(X)[0][1])
    ml_label = "NON-COMPLIANT" if p_non >= 0.5 else "COMPLIANT"

    # --- Layer 2: explicit date rules -------------------------------------
    issues: list[Issue] = []
    date_failed = False
    if expiry and expiry < today:
        date_failed = True
        issues.append(Issue("expired", "Past use-by / expiry date",
                            f"Declared use-by {expiry:%d %b %Y} is before today ({today:%d %b %Y})."))
    if mfg and mfg > today:
        date_failed = True
        issues.append(Issue("date", "Manufacturing date is in the future",
                            f"Manufacturing date {mfg:%b %Y} is after today."))
    if mfg and expiry and expiry < mfg:
        date_failed = True
        issues.append(Issue("date", "Expiry date is before manufacturing date",
                            f"{expiry:%b %Y} is earlier than {mfg:%b %Y}."))

    for key, name in FIELD_NAMES.items():
        if not feats[key]:
            issues.append(Issue("missing", f"{name} not detected",
                                "The OCR text did not contain a recognisable declaration."))

    # --- Final decision -----------------------------------------------------
    if date_failed:
        status = "NON-COMPLIANT"
        first = next(i for i in issues if i.kind in ("expired", "date"))
        reason = ("Product appears to be past its declared use-by/expiry date."
                  if first.kind == "expired" else f"Date check failed: {first.title.lower()}.")
    elif ml_label == "NON-COMPLIANT":
        status = "NON-COMPLIANT"
        reason = "The ML model classified the extracted label features as non-compliant."
        issues.insert(0, Issue("model", "Model prediction: non-compliant",
                               f"Probability {p_non:.0%} based on the detected declarations."))
    else:
        status = "COMPLIANT"
        missing = [i for i in issues if i.kind == "missing"]
        reason = ("No configured label issues were detected in this demonstration."
                  if not missing else
                  "The ML model classified the label as compliant; minor gaps are listed below.")

    return Verdict(status=status, reason=reason, features=feats, ml_label=ml_label,
                   ml_probability=p_non, model_name=bundle.get("model_name"), issues=issues,
                   mfg_date=mfg, expiry_date=expiry, date_rule_failed=date_failed)
