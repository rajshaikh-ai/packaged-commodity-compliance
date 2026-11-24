"""
test_pipeline.py
----------------
1. Unit checks for date parsing and the decision logic (no OCR needed).
2. End-to-end run on every image in samples/: OCR -> features -> model -> date rules.

    python test_pipeline.py

The expected results below are used ONLY to report pass/fail in this test.
The app itself never sees file names.
"""

from __future__ import annotations

import sys
import time
from datetime import date
from pathlib import Path

from compliance_checker import check_label, load_model
from feature_extractor import LabelFields, extract_dates, find_dates, parse_user_date

ROOT = Path(__file__).parent
TODAY = date.today()
results = []


def check(name: str, ok: bool, info: str = ""):
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({info})" if info else ""))


def unit_tests(bundle):
    print("\n== Date parsing ==")
    check("'31 JAN 2026'", find_dates("MFD. : 31 JAN 2026")[0].value == date(2026, 1, 31))
    check("'06/2020'", parse_user_date("06/2020") == (date(2020, 6, 1), "month"))
    check("'MAY 2026'", parse_user_date("May 2026") == (date(2026, 5, 1), "month"))
    check("'31/05/26'", parse_user_date("31/05/26") == (date(2026, 5, 31), "day"))
    d = extract_dates("MFG. DATE 06/2020 USE BY 05/2026")
    check("use-by month resolves to month end", d["expiry_date"] == date(2026, 5, 31), str(d["expiry_date"]))
    d = extract_dates("PKD ON: 03/2025 BEST BEFORE 24 MONTHS FROM PACKING")
    check("relative 'best before 24 months'", d["expiry_date"] == date(2027, 3, 31), str(d["expiry_date"]))
    check("plot numbers are not dates", find_dates("PLOT NO. SP 181-189 & 191") == [])
    d = extract_dates("RS 280\nBD0111\n06/20\n05/26\nMRP RS. BATCH NO., MFD. &\nUSE BY DATE")
    check("legend layout: 06/20 + 05/26 -> mfg + use-by",
          d["mfg_date"] == date(2020, 6, 1) and d["expiry_date"] == date(2026, 5, 31),
          f"{d['mfg_date']} / {d['expiry_date']}")
    d = extract_dates("06 0\n05_\nBATCH NO., MFD. & USE BY DATE")
    check("unreadable dates -> fragments, no guess",
          d["mfg_date"] is None and d["expiry_date"] is None and d["fragments"] == ["06 0", "05_"])

    print("\n== Decision logic (synthetic field sets) ==")
    full = dict(mrp="Rs. 10.00", net_quantity="50 G", manufacturer="X LTD", address="A, DELHI-110001",
                customer_care="1800 000 0000", word_count=40)
    expired = LabelFields(**full, dates={"mfg_date": date(2020, 6, 1), "expiry_date": date(2026, 5, 31)})
    fresh = LabelFields(**full, dates={"mfg_date": date(2026, 1, 31), "expiry_date": None})
    sparse = LabelFields(mrp="Rs. 10.00", word_count=4, dates={})
    no_core = LabelFields(manufacturer="X LTD", address="A, DELHI-110001", word_count=30,
                          dates={"mfg_date": date(2026, 1, 1)})
    today = date(2026, 11, 24)
    check("expired label -> NON-COMPLIANT", check_label(expired, bundle, today).status == "NON-COMPLIANT")
    check("complete, not expired -> COMPLIANT", check_label(fresh, bundle, today).status == "COMPLIANT")
    check("almost no text -> UNABLE", check_label(sparse, bundle, today).status == "UNABLE")
    v = check_label(no_core, bundle, today)
    check("MRP + quantity missing -> model says NON-COMPLIANT", v.status == "NON-COMPLIANT",
          f"P(non)={v.ml_probability:.2f}")
    v = check_label(fresh, bundle, today, expiry_override=date(2025, 12, 31))
    check("manual expiry correction is applied", v.status == "NON-COMPLIANT")


# Used for reporting only: (accepted automatic results, manual dates, result after correction)
EXPECTED = {
    "havells_led_lamp": ({"COMPLIANT"}, None, None),
    # Pen-modified dates: OCR may or may not read them. Either it reads them and
    # flags NON-COMPLIANT, or it must refuse to guess (UNABLE). After the manual
    # correction 06/2020 + 05/2026 it must be NON-COMPLIANT.
    "odonil_neem": ({"NON-COMPLIANT", "UNABLE"}, ("06/2020", "05/2026"), "NON-COMPLIANT"),
}


def end_to_end(bundle):
    from feature_extractor import extract_fields
    from ocr_engine import load_image, run_ocr

    print("\n== End-to-end on samples/ ==")
    images = sorted(p for p in (ROOT / "samples").iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".heic"})
    for path in images:
        t0 = time.time()
        ocr = run_ocr(load_image(path.read_bytes()))
        fields = extract_fields(ocr.text, ocr.largest_text)
        v = check_label(fields, bundle, TODAY)
        print(f"\n--- {path.name}  ({time.time() - t0:.1f}s, {ocr.word_count} words, "
              f"conf {ocr.mean_confidence:.0%}, pass={ocr.pass_used})")
        for k in ("product_name", "mrp", "net_quantity", "manufacturer", "address", "customer_care"):
            print(f"    {k:15s}: {getattr(fields, k)}")
        print(f"    mfg date       : {fields.dates.get('mfg_date')}  raw={fields.dates.get('mfg_raw')!r}")
        print(f"    expiry date    : {fields.dates.get('expiry_date')}  raw={fields.dates.get('expiry_raw')!r}")
        print(f"    features       : {v.features}")
        print(f"    ML             : {v.ml_label}  P(non)={v.ml_probability}")
        print(f"    date fragments : {fields.dates.get('fragments')}")
        print(f"    RESULT         : {v.status} - {v.reason}")
        if path.stem not in EXPECTED:
            continue
        auto_ok, manual, after = EXPECTED[path.stem]
        check(f"{path.name} automatic result in {sorted(auto_ok)}", v.status in auto_ok, v.status)
        if manual:
            from feature_extractor import month_end
            mfg = parse_user_date(manual[0])[0]
            exp = month_end(parse_user_date(manual[1])[0])
            v2 = check_label(fields, bundle, TODAY, mfg_override=mfg, expiry_override=exp)
            print(f"    + manual dates {manual}: {v2.status} - {v2.reason}")
            check(f"{path.name} with manual dates -> {after}", v2.status == after, v2.status)


if __name__ == "__main__":
    bundle = load_model()
    unit_tests(bundle)
    if "--no-ocr" not in sys.argv:
        end_to_end(bundle)
    print(f"\n{sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)
