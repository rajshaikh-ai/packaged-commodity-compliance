"""
feature_extractor.py
--------------------
Turns raw OCR text into structured label fields and ML features.

Only regular expressions and simple text rules are used here, so every
decision can be explained line by line. Nothing in this file looks at the
image file name or tries to recognise a specific brand.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass, field
from datetime import date

# The six label declarations this project checks for.
REQUIRED_FIELDS = [
    "mrp_present",
    "quantity_present",
    "manufacturer_present",
    "address_present",
    "date_present",
    "customer_care_present",
]

# The exact feature order the ML model was trained on.
FEATURE_COLUMNS = REQUIRED_FIELDS + ["expiry_detected", "is_expired", "label_completeness"]


# ---------------------------------------------------------------------------
# 1. Text cleaning
# ---------------------------------------------------------------------------
_DIGIT_LOOKALIKES = str.maketrans({"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "|": "1",
                                   "S": "5", "B": "8", "Z": "2"})


def _fix_numeric_token(match: re.Match) -> str:
    """'2O26' -> '2026', '3I' -> '31'. Only touches tokens that already contain a digit."""
    token = match.group(0)
    if not any(ch.isdigit() for ch in token):
        return token
    # Leave tokens with 2+ letters alone (batch / item codes like 'BD0111').
    if sum(ch.isalpha() for ch in token) > 1:
        return token
    return token.translate(_DIGIT_LOOKALIKES)


def clean_text(raw: str) -> str:
    """Upper-case, normalise punctuation and repair common OCR digit confusions."""
    text = raw.upper()
    text = text.replace("₹", " RS ")        # rupee sign
    text = re.sub(r"[‘’`´]", "'", text)
    text = re.sub(r"[–—]", "-", text)
    text = re.sub(r"\b[0-9OQDILSBZ|]{2,6}\b", _fix_numeric_token, text)
    text = re.sub(r"[ \t]+", " ", text)
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


# ---------------------------------------------------------------------------
# 2. Date parsing
# ---------------------------------------------------------------------------
MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6, "JUL": 7,
    "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}
_MON = r"(JAN(?:UARY)?|FEB(?:RUARY)?|MAR(?:CH)?|APR(?:IL)?|MAY|JUNE?|JULY?|AUG(?:UST)?|SEP(?:T(?:EMBER)?)?|OCT(?:OBER)?|NOV(?:EMBER)?|DEC(?:EMBER)?)"
_SEP = r"\s*[/\-.]\s*"

# (regex, kind) - tried in this order, longer / more specific formats first.
_DATE_PATTERNS = [
    (re.compile(rf"\b(\d{{1,2}})[\s\-./]*{_MON}[\s\-./,']*(\d{{4}}|\d{{2}})\b"), "d_mon_y"),
    (re.compile(rf"\b{_MON}[\s\-./,']*(\d{{4}}|\d{{2}})\b"), "mon_y"),
    (re.compile(rf"\b(\d{{1,2}}){_SEP}(\d{{1,2}}){_SEP}(\d{{4}}|\d{{2}})\b"), "d_m_y"),
    (re.compile(rf"\b(\d{{1,2}}){_SEP}(\d{{4}})\b"), "m_y"),
]
# Two-digit years with only month ("05/26") are ambiguous, so they are only
# accepted right after a date keyword such as "USE BY".
_SHORT_PATTERN = (re.compile(rf"\b(\d{{1,2}}){_SEP}(\d{{2}})\b"), "m_y")


@dataclass
class FoundDate:
    value: date          # first day for mfg, resolved later for expiry
    precision: str       # "day" or "month"
    raw: str
    start: int
    end: int


def _year(y: str) -> int | None:
    year = int(y)
    if year < 100:
        year += 2000
    return year if 1990 <= year <= 2069 else None


def _build(kind: str, groups: tuple) -> tuple[date, str] | None:
    try:
        if kind == "d_mon_y":
            d, mon, y = groups
            year = _year(y)
            return (date(year, MONTHS[mon[:3]], int(d)), "day") if year else None
        if kind == "mon_y":
            mon, y = groups
            year = _year(y)
            return (date(year, MONTHS[mon[:3]], 1), "month") if year else None
        if kind == "d_m_y":
            d, m, y = groups
            year = _year(y)
            return (date(year, int(m), int(d)), "day") if year else None
        if kind == "m_y":
            m, y = groups
            year = _year(y)
            return (date(year, int(m), 1), "month") if year and 1 <= int(m) <= 12 else None
    except (ValueError, KeyError):
        return None
    return None


def find_dates(text: str, allow_short: bool = False) -> list[FoundDate]:
    """Return every valid date in the text, without overlapping matches."""
    patterns = _DATE_PATTERNS + ([_SHORT_PATTERN] if allow_short else [])
    found: list[FoundDate] = []
    taken: list[tuple[int, int]] = []
    for regex, kind in patterns:
        for m in regex.finditer(text):
            if any(m.start() < e and s < m.end() for s, e in taken):
                continue
            built = _build(kind, m.groups())
            if built:
                found.append(FoundDate(built[0], built[1], m.group(0).strip(), m.start(), m.end()))
                taken.append((m.start(), m.end()))
    return sorted(found, key=lambda f: f.start)


def month_end(d: date) -> date:
    return date(d.year, d.month, calendar.monthrange(d.year, d.month)[1])


def add_months(d: date, months: int) -> date:
    total = d.month - 1 + months
    year, month = d.year + total // 12, total % 12 + 1
    return date(year, month, min(d.day, calendar.monthrange(year, month)[1]))


def parse_user_date(value: str) -> tuple[date, str] | None:
    """Parse a manually typed date such as '05/2026', 'MAY 2026' or '31/05/2026'."""
    value = clean_text(value or "")
    dates = find_dates(value, allow_short=True)
    return (dates[0].value, dates[0].precision) if dates else None


_MFG_KW = re.compile(
    r"(DATE\s*OF\s*(?:MFG|MANUFACTURE|MANUFACTURING|PACKING|PKG)|MFG\.?\s*DATE|MFD\.?\s*ON|"
    r"MANUFACTURED\s*ON|MANUFACTURING\s*DATE|PACKED\s*ON|PKD\.?\s*ON|\bMFG\b|\bMFD\b|\bPKD\b|\bPKG\b)"
)
_EXP_KW = re.compile(
    r"(USE\s*BY|USE\s*BEFORE|BEST\s*BEFORE|EXPIRY(?:\s*DATE)?|EXPIRES?\s*(?:ON)?|\bEXP\b\.?|BEST\s*USED\s*BEFORE)"
)
_RELATIVE_EXP = re.compile(
    r"(?:BEST\s*BEFORE|USE\s*(?:BY|BEFORE|WITHIN)|EXPIRY)\W{0,6}(\d{1,2})\s*MONTHS?"
)


def _date_after_keyword(text: str, kw: re.Pattern, stop: re.Pattern, window: int = 45):
    """First date that appears shortly after a keyword (and before an opposing keyword)."""
    for m in kw.finditer(text):
        segment = text[m.end(): m.end() + window]
        other = stop.search(segment)
        if other:
            segment = segment[: other.start()]
        dates = find_dates(segment, allow_short=True)
        if dates:
            return dates[0], m.group(0).strip()
    return None, None


# A line that is only the start of a date, e.g. '06/2', '06 0', '05_'
_FRAGMENT = re.compile(r"^\W*(0[1-9]|1[0-2])(?:\s*[/\-_. ]\s*\d{0,2})?\W*$")
# Wording that means "a date is printed on this label" (not "Mfd. by: company")
_DATE_DECLARED = re.compile(
    r"USE\s*BY|USE\s*BEFORE|BEST\s*BEFORE|EXPIRY|EXP\.?\s*DATE|MFG\.?\s*DATE|MFD\.?\s*DATE|"
    r"DATE\s*OF\s*(?:MFG|MANUFACTURE|PACKING)|PKD\.?\s*ON|PACKED\s*ON|MFD\.?\s*ON|MFG\.?\s*ON"
)


def extract_dates(text: str) -> dict:
    """Find manufacturing and expiry / use-by dates and explain how they were found."""
    flat = text.replace("\n", " ")
    lines = text.splitlines()
    # Dates printed on a line of their own. Common in "legend" layouts where the
    # label says "Mfd. & Use By Date (see above)" and the values sit elsewhere.
    standalone = []
    for line in lines:
        hit = find_dates(line, allow_short=True)
        if len(hit) == 1 and len(re.sub(r"\W", "", line)) <= len(re.sub(r"\W", "", hit[0].raw)) + 2:
            standalone.append(hit[0])
    fragments = [l.strip() for l in lines if _FRAGMENT.match(l)]

    mfg, mfg_kw = _date_after_keyword(flat, _MFG_KW, _EXP_KW)
    exp, exp_kw = _date_after_keyword(flat, _EXP_KW, _MFG_KW)
    all_dates = find_dates(flat)

    result = {
        "mfg_date": mfg.value if mfg else None,
        "mfg_raw": mfg.raw if mfg else None,
        "mfg_source": f"after keyword '{mfg_kw}'" if mfg else None,
        "mfg_precision": mfg.precision if mfg else None,
        "expiry_precision": None,
        "expiry_date": None,
        "expiry_raw": None,
        "expiry_source": None,
        "all_dates": [d.raw for d in all_dates],
        "fragments": fragments,
        "date_declared": bool(_DATE_DECLARED.search(flat)),
    }

    if exp:
        result["expiry_date"] = month_end(exp.value) if exp.precision == "month" else exp.value
        result["expiry_raw"] = exp.raw
        result["expiry_source"] = f"after keyword '{exp_kw}'"
        result["expiry_precision"] = exp.precision
    else:
        rel = _RELATIVE_EXP.search(flat)
        if rel and mfg:
            months = int(rel.group(1))
            result["expiry_date"] = month_end(add_months(mfg.value, months))
            result["expiry_raw"] = rel.group(0).strip()
            result["expiry_source"] = f"{months} months from manufacturing date"
            result["expiry_precision"] = "month"

    # Dates not next to any keyword. With exactly two distinct dates, or with a
    # "Mfd. & Use By" legend on the label: earlier = mfg, later = use-by.
    candidates = {d.value: d for d in all_dates + standalone}
    legend = bool(_MFG_KW.search(flat) and _EXP_KW.search(flat))
    if not mfg and not exp and (len(candidates) == 2 or (legend and len(candidates) > 2)):
        ordered = sorted(candidates.values(), key=lambda d: d.value)
        first, second = ordered[0], ordered[-1]
        how = "Mfd. & Use By legend" if legend else "two dates found"
        result.update(
            mfg_date=first.value, mfg_raw=first.raw, mfg_source=f"inferred: earlier date ({how})",
            expiry_date=month_end(second.value) if second.precision == "month" else second.value,
            expiry_raw=second.raw, expiry_source=f"inferred: later date ({how})",
            mfg_precision=first.precision, expiry_precision=second.precision,
        )
    return result


# ---------------------------------------------------------------------------
# 3. Other label fields
# ---------------------------------------------------------------------------
_MRP_KW = re.compile(r"(M\.?\s?R\.?\s?P\.?|MAX(?:IMUM)?\.?\s*RETAIL\s*PRICE|\bPRICE\b)")
# '1,999.00', '1 999.00', '1 999 00' (OCR dropped the point), '75.00', '280'
_PRICE = re.compile(r"\d{1,3}(?:[ ,]\d{3})+[ .]\d{2}\b|\d{1,3}(?:,\d{3})+|\d+\.\d{2}|\d{1,6}")

_QTY = re.compile(
    r"NET\s*(?:QUANTITY|QTY|WT|WEIGHT|CONTENTS?|VOL(?:UME)?|MASS)\.?\W{0,6}"
    r"(\d+(?:\.\d+)?)\s*(KG|GMS?|G|ML|LTR|L|N|U|PCS|NOS|UNITS?|PIECES?)\b"
)
_QTY_KW = re.compile(r"NET\s*(?:QUANTITY|QTY|WT|WEIGHT|CONTENTS?|VOL(?:UME)?|MASS)")
_QTY_LOOSE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(KG|GMS?|G|ML|LTR)\b")

_COMPANY = re.compile(
    r"([A-Z][A-Z0-9&.,'\- ]{2,50}?\b(?:PVT\.?\s*LTD|PRIVATE\s+LIMITED|LTD|LIMITED|LLP|"
    r"CORPORATION|CORP|INDUSTRIES|ENTERPRISES))\b\.?"
)
_MAKER_KW = re.compile(r"(MANUFACTURED\s*(?:&\s*MARKETED\s*)?BY|MFD\.?\s*(?:&\s*MKTD\.?\s*)?BY|MARKETED\s*BY|MKTD\.?\s*BY|PACKED\s*BY|IMPORTED\s*BY)\W*")

_PIN = re.compile(r"\b[1-8]\d{2}\s?\d{3}\b")
_ADDRESS_KW = re.compile(
    r"\b(PLOT|ROAD|RD|MARG|STREET|NAGAR|SECTOR|INDUSTRIAL|ESTATE|AREA|PHASE|DIST|DISTT|BUILDING|"
    r"FLOOR|VILLAGE|TEHSIL|HIGHWAY|OFFICE|COLONY|PARK|LANE|NEAR|BLOCK|"
    r"DELHI|MUMBAI|MAHARASHTRA|RAJASTHAN|GUJARAT|KARNATAKA|TAMIL|UTTAR|PRADESH|HARYANA|PUNJAB|"
    r"BENGAL|KERALA|TELANGANA|HIMACHAL|UTTARAKHAND|BIHAR|ODISHA|ASSAM|GOA)\b"
)
_CC_KW = re.compile(
    r"(CARES?\b|CALL\b|CUSTOMER\s*CARE|CONSUMER\s*CARE|CUSTOMER\s*SERVICE|CONSUMER\s*COMPLAINT|HELPLINE|HELP\s*LINE|"
    r"TOLL\s*FREE|FEEDBACK|CONTACT|E-?\s?MAIL|PH(?:ONE)?\.?\s*NO|TEL\b)"
)
_PHONE = re.compile(r"(?:\+?\s?91[\s\-]?)?\(?\d[\d\s\-()]{8,16}\d")
_EMAIL = re.compile(r"[A-Z0-9._%+\-]+\s?@\s?[A-Z0-9.\-]+\.(?:COM|IN|CO\.IN|NET|ORG)")
_NAME_KW = re.compile(
    r"(ITEM\s*NAME|PRODUCT\s*NAME|NAME\s*OF\s*(?:THE\s*)?(?:COMMODITY|PRODUCT)|GENERIC\s*NAME|COMMODITY)\s*[:;\-.]?\s*(.+)"
)
_DESC_KW = re.compile(r"(ITEM\s*DESCRIPTION|PRODUCT\s*DESCRIPTION|DESCRIPTION)\s*[:;\-.]?\s*(.+)")


def _first_line_match(lines: list[str], regex: re.Pattern, group: int = 2) -> str | None:
    for line in lines:
        m = regex.search(line)
        if m and m.group(group).strip(" :.-"):
            return m.group(group).strip(" :.-")
    return None


def extract_mrp(flat: str) -> str | None:
    for m in _MRP_KW.finditer(flat):
        window = flat[m.end(): m.end() + 40]
        # Drop 'Rs.' / '(incl. of all taxes)' after the keyword ...
        window = re.sub(r"^\W*(?:RS\.?|INR)?\W*(?:\(INC\w*[^)]{0,25}\)?)?\W*(?:RS\.?|INR)?", "", window)
        # ... and stop at the next word: the price must follow the MRP heading.
        window = re.split(r"[A-Z]{2,}", window, maxsplit=1)[0]
        # A lone single digit is far more likely OCR noise than a price.
        prices = [p.group(0) for p in _PRICE.finditer(window) if len(re.sub(r"\D", "", p.group(0))) >= 2]
        # Prefer a value with paise ('.00' or ' 00'), otherwise the first number.
        best = next((p for p in prices if re.search(r"[ .]\d{2}$", p)), prices[0] if prices else None)
        if best:
            value = re.sub(r"[ .](\d{2})$", r".\1", best).replace(" ", "").replace(",", "")
            try:
                if 0 < float(value) < 1_000_000:
                    return f"Rs. {float(value):,.2f}"
            except ValueError:
                pass
    # The MRP heading is printed but its value could not be read.
    return "Declared (value unclear)" if _MRP_KW.search(flat) else None


def extract_quantity(flat: str) -> str | None:
    matches = list(_QTY.finditer(flat))
    if matches:   # prefer 'Net Quantity: 192g' over 'Net Contents: 4N x 48g'
        m = next((m for m in matches if re.search(r"QUANTITY|QTY", m.group(0))), matches[0])
        return f"{m.group(1)} {m.group(2)}"
    if _QTY_KW.search(flat):
        loose = _QTY_LOOSE.search(flat[_QTY_KW.search(flat).end():][:40])
        return f"{loose.group(1)} {loose.group(2)}" if loose else "Declared (value unclear)"
    loose = _QTY_LOOSE.search(flat)
    return f"{loose.group(1)} {loose.group(2)}" if loose else None


def extract_manufacturer(lines: list[str]) -> str | None:
    usable = [l for l in lines if "TRADEMARK" not in l]
    # Prefer a company named right after "Manufactured by" / "Marketed by".
    for i, line in enumerate(usable):
        kw = _MAKER_KW.search(line)
        if kw:
            tail = (line[kw.end():] + " " + (usable[i + 1] if i + 1 < len(usable) else "")).strip()
            m = _COMPANY.search(tail)
            if m:
                return m.group(1).strip(" ,.:")
    for line in usable:
        m = _COMPANY.search(line)
        if m:
            return m.group(1).strip(" ,.:")
    return None


def extract_address(lines: list[str]) -> str | None:
    for i, line in enumerate(lines):
        if _PIN.search(line):
            block = lines[max(0, i - 2): i + 1]
            if any(_ADDRESS_KW.search(l) for l in block):
                # Start from the first line in the block that looks like an address.
                start = next(j for j, l in enumerate(block) if _ADDRESS_KW.search(l) or _PIN.search(l))
                address = " ".join(block[start:])
                address = re.sub(r"^.*?\b(?:MFD\.?\s*AT|ADDRESS|REGD\.?\s*OFFICE|OFFICE)\s*[:;\-]?\s*", "", address)
                address = re.sub(r"^.*?\b(?:LTD|LIMITED)\b\W*", "", address)   # drop company name
                return address.strip(" ,.:;&")
    return None


def extract_customer_care(lines: list[str]) -> str | None:
    flat = " ".join(lines)
    parts = []
    for i, line in enumerate(lines):
        if _CC_KW.search(line):
            scope = line + " " + (lines[i + 1] if i + 1 < len(lines) else "")
            for p in _PHONE.finditer(scope):
                digits = re.sub(r"\D", "", p.group(0))
                if 10 <= len(digits) <= 13 and p.group(0).strip() not in parts:
                    parts.append(p.group(0).strip())
                    break
        if parts:
            break
    email = _EMAIL.search(flat)
    if email:
        parts.append(email.group(0).replace(" ", "").lower())
    return " | ".join(parts) if parts else None


# ---------------------------------------------------------------------------
# 4. Putting it together
# ---------------------------------------------------------------------------
@dataclass
class LabelFields:
    product_name: str | None = None
    product_name_source: str | None = None
    description: str | None = None
    mrp: str | None = None
    net_quantity: str | None = None
    manufacturer: str | None = None
    address: str | None = None
    customer_care: str | None = None
    dates: dict = field(default_factory=dict)
    word_count: int = 0


def extract_fields(raw_text: str, largest_text: str | None = None) -> LabelFields:
    text = clean_text(raw_text)
    lines = text.splitlines()
    flat = " ".join(lines)

    name = _first_line_match(lines, _NAME_KW)
    name_source = "label field" if name else None
    if not name and largest_text:
        name, name_source = largest_text.title(), "largest text on label"

    return LabelFields(
        product_name=name.title() if name and name_source == "label field" else name,
        product_name_source=name_source,
        description=(_first_line_match(lines, _DESC_KW) or "").title() or None,
        mrp=extract_mrp(flat),
        net_quantity=extract_quantity(flat),
        manufacturer=extract_manufacturer(lines),
        address=extract_address(lines),
        customer_care=extract_customer_care(lines),
        dates=extract_dates(text),
        word_count=len(re.findall(r"[A-Z0-9]{2,}", flat)),
    )


def build_features(fields: LabelFields, today: date, mfg_date: date | None = None,
                   expiry_date: date | None = None) -> dict:
    """Convert extracted fields into the numeric feature vector used by the model.

    mfg_date / expiry_date let the caller pass manually corrected dates.
    """
    expiry = expiry_date if expiry_date is not None else fields.dates.get("expiry_date")
    mfg = mfg_date if mfg_date is not None else fields.dates.get("mfg_date")
    feats = {
        "mrp_present": int(fields.mrp is not None),
        "quantity_present": int(fields.net_quantity is not None),
        "manufacturer_present": int(fields.manufacturer is not None),
        "address_present": int(fields.address is not None),
        "date_present": int(mfg is not None or expiry is not None),
        "customer_care_present": int(fields.customer_care is not None),
        "expiry_detected": int(expiry is not None),
        "is_expired": int(expiry is not None and expiry < today),
    }
    feats["label_completeness"] = round(sum(feats[f] for f in REQUIRED_FIELDS) / len(REQUIRED_FIELDS), 3)
    return feats
