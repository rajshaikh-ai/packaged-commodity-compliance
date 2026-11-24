"""
app.py - Streamlit web app
Run with:  streamlit run app.py
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from compliance_checker import FIELD_NAMES, Verdict, check_label, load_model
from feature_extractor import extract_fields, month_end, parse_user_date
from ocr_engine import get_reader, load_image, run_ocr

ROOT = Path(__file__).parent
st.set_page_config(page_title="Label Compliance · SIH26034", page_icon="🔎", layout="wide")
st.markdown(f"<style>{(ROOT / 'assets' / 'style.css').read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True)

# Quick-demo photos. The buttons only load the photo and run the normal
# pipeline. For the Odonil photo, OCR cannot read the pen-written dates, so the
# demo also enters them as a MANUAL correction (shown as such in the UI).
DEMO_SAMPLES = {
    "odonil": {"file": "odonil_neem.jpg", "manual_dates": ("06/2020", "05/2026")},
    "havells": {"file": "havells_led_lamp.jpg", "manual_dates": None},
}

# ---------------------------------------------------------------------------
# Icons (inline SVG, stroke = currentColor)
# ---------------------------------------------------------------------------
_P = {
    "ok": '<circle cx="12" cy="12" r="9.5"/><path d="m8.5 12 2.5 2.5 4.5-5"/>',
    "miss": '<circle cx="12" cy="12" r="9.5"/><path d="M8 12h8"/>',
    "x": '<circle cx="12" cy="12" r="9.5"/><path d="m9.5 9.5 5 5"/><path d="m14.5 9.5-5 5"/>',
    "clock": '<circle cx="12" cy="12" r="9.5"/><path d="M12 7v5l3 2"/>',
    "warn": '<path d="M10.3 3.9 2.5 17.5A2 2 0 0 0 4.2 20.5h15.6a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
    "unk": '<circle cx="12" cy="12" r="9.5"/><path d="M9.1 9a3 3 0 0 1 5.8 1c0 2-3 3-3 3"/><path d="M12 17h.01"/>',
    "cube": '<path d="M21 7.5 12 2.5 3 7.5v9l9 5 9-5z"/><path d="M3 7.5l9 5 9-5"/><path d="M12 12.5v9"/>',
    "search": '<circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/>',
    "bars": '<path d="M5 20v-6"/><path d="M10 20V9"/><path d="M15 20v-9"/><path d="M20 20V4"/>',
    "info": '<circle cx="12" cy="12" r="9.5"/><path d="M12 11v5"/><path d="M12 7.5h.01"/>',
    "ocr": '<path d="M3 7V5a2 2 0 0 1 2-2h2"/><path d="M17 3h2a2 2 0 0 1 2 2v2"/><path d="M21 17v2a2 2 0 0 1-2 2h-2"/><path d="M7 21H5a2 2 0 0 1-2-2v-2"/><path d="M8 9h8M8 12h8M8 15h5"/>',
    "layers": '<path d="m12 3 9 5-9 5-9-5z"/><path d="m3 13 9 5 9-5"/>',
    "calendar": '<rect x="3.5" y="5" width="17" height="15.5" rx="2"/><path d="M3.5 10h17"/><path d="M8 3v4M16 3v4"/>',
    "cloud": '<path d="M7 18a4.5 4.5 0 0 1-.6-8.96A6 6 0 0 1 18 8.5a4.5 4.5 0 0 1-.5 9.5"/><path d="M12 12v8"/><path d="m8.5 15.5 3.5-3.5 3.5 3.5"/>',
    "image": '<rect x="3.5" y="3.5" width="17" height="17" rx="2"/><circle cx="9" cy="9" r="1.6"/><path d="m20.5 15-4.5-4.5L6 20.5"/>',
    "doc": '<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5"/><path d="M9 13h6M9 17h6"/>',
    "target": '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1.2"/>',
    "star": '<path d="m12 3 2.8 5.7 6.2.9-4.5 4.4 1.1 6.2L12 17.3 6.4 20.2l1.1-6.2L3 9.6l6.2-.9z"/>',
    "shield_ok": '<path d="M12 2.5 4 5.5v6c0 5 3.4 8.6 8 10 4.6-1.4 8-5 8-10v-6z"/><path d="m8.5 12 2.5 2.5 4.5-5"/>',
}


def icon(name: str, cls: str = "", size: int = 18) -> str:
    return (f'<svg class="ico {cls}" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{_P[name]}</svg>')


def html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_ocr_reader():
    """One EasyOCR model for the whole server, shared by all visitors.

    st.cache_resource survives Streamlit code reloads, so the ~500 MB model
    is never loaded twice (a second copy would exceed free-hosting memory).
    """
    return get_reader()


@st.cache_resource(show_spinner="Loading model…")
def get_model():
    try:
        return load_model()
    except Exception:
        # Missing model, or saved with a different scikit-learn version:
        # retrain (takes seconds, same seed -> same model) and load again.
        import train_model
        train_model.main()
        get_metrics.clear()
        return load_model()


@st.cache_data(show_spinner=False)
def get_metrics():
    path = ROOT / "evaluation" / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else None


# ---------------------------------------------------------------------------
# Session state
#   current   : the image being inspected {"data", "name", "digest", "origin"}
#   analysis  : OCR + extracted fields for `current`
#   overrides : manual date corrections {"mfg"/"exp": (date, precision, raw)}
#   pending   : demo sample waiting to be analysed
# ---------------------------------------------------------------------------
ss = st.session_state
for k, v in {"uploader_key": 0, "current": None, "analysis": None, "overrides": {}, "pending": None}.items():
    ss.setdefault(k, v)


def reset():
    ss.uploader_key += 1
    ss.current, ss.analysis, ss.overrides, ss.pending = None, None, {}, None


def load_demo(key: str):
    """Quick Demo button: load the sample photo; it is analysed on this run."""
    data = (ROOT / "samples" / DEMO_SAMPLES[key]["file"]).read_bytes()
    reset()
    ss.current = {"data": data, "name": DEMO_SAMPLES[key]["file"],
                  "digest": hashlib.md5(data).hexdigest(), "origin": "sample"}
    ss.pending = key


def parse_overrides(mfg_raw: str, exp_raw: str) -> tuple[dict, list[str]]:
    new, errors = {}, []
    for key, raw, name in (("mfg", mfg_raw, "Manufacturing date"), ("exp", exp_raw, "Expiry date")):
        if raw.strip():
            parsed = parse_user_date(raw)
            if parsed:
                value, precision = parsed
                if key == "exp" and precision == "month":   # use-by month = valid through month end
                    value = month_end(value)
                new[key] = (value, precision, raw.strip())
            else:
                errors.append(f"{name}: could not understand '{raw}'.")
    return new, errors


def analyse(status_parent) -> None:
    """Run the full pipeline on ss.current and store the result."""
    with status_parent.status("Analysing label…", expanded=True) as status:
        try:
            st.write("Preprocessing image and reading text (first run loads the OCR model)…")
            ocr = run_ocr(load_image(ss.current["data"]), reader=get_ocr_reader())
            st.write(f"OCR read {ocr.word_count} words · mean confidence {ocr.mean_confidence:.0%}")
            st.write("Extracting label fields, running ML model and date checks…")
            fields = extract_fields(ocr.text, ocr.largest_text)
            ss.analysis = {"digest": ss.current["digest"], "ocr": ocr, "fields": fields}
            ss.overrides = {}
            demo = DEMO_SAMPLES.get(ss.pending or "")
            if demo and demo["manual_dates"]:
                ss.overrides, _ = parse_overrides(*demo["manual_dates"])
            status.update(label="Analysis complete", state="complete", expanded=False)
        except Exception as exc:  # keep the app alive on any OCR failure
            ss.analysis = None
            status.update(label="Analysis failed", state="error")
            st.error(f"Could not analyse this image: {exc}")
        finally:
            ss.pending = None


# ---------------------------------------------------------------------------
# Header + hero
# ---------------------------------------------------------------------------
html(f"""
<header class="topbar">
  <div class="topbar-inner">
    <div class="tb-brand">
      <span class="tb-logo">{icon('cube', size=34)}</span>
      <span class="tb-title">Packaged Commodity Compliance<br>Classification from Product Labels</span>
      <span class="tb-sep"></span>
      <span class="tb-badge">SIH26034</span>
    </div>
    <nav class="tb-nav">
      <a class="active" href="#inspect" target="_self">{icon('search')} Inspect</a>
      <a href="#model-performance" target="_self">{icon('bars')} Model Performance</a>
      <a href="#about" target="_self">{icon('info')} About</a>
    </nav>
  </div>
</header>
<div id="inspect" class="anchor"></div>
""")

hero_l, hero_r = st.columns([5, 6], gap="large", vertical_alignment="center")
with hero_l:
    html("""<div class="hero">
      <h1 class="hero-title">Inspect Product Labels.<br><em class="hl">Identify Potential Violations.</em></h1>
      <p class="hero-sub">Upload a product label to extract key information and classify the product
      as Compliant or Non-Compliant.</p></div>""")
with hero_r:
    html(f"""<div class="features">
      <div class="feature">{icon('ocr', size=34)}<b>OCR Extraction</b><span>Read label text</span></div>
      <div class="feature">{icon('layers', size=34)}<b>ML Classification</b><span>Identify compliance</span></div>
      <div class="feature">{icon('calendar', size=34)}<b>Date Validation</b><span>Check expiry / use-by</span></div>
    </div>""")

try:
    bundle = get_model()
except FileNotFoundError as exc:
    st.error(f"{exc}")
    st.stop()

left, right = st.columns([1, 2.3], gap="medium")

# ---------------------------------------------------------------------------
# Left: upload card
# ---------------------------------------------------------------------------
with left:
    with st.container(key="upload_card"):
        html(f'<div class="card-head">{icon("cloud", size=26)}<span>Upload Product Label</span></div>')
        upload = st.file_uploader("Product label image", type=["jpg", "jpeg", "png", "heic", "heif"],
                                  key=f"uploader_{ss.uploader_key}", label_visibility="collapsed")

        # A new upload replaces whatever was being inspected (sample or earlier upload).
        if upload is not None:
            data = upload.getvalue()
            digest = hashlib.md5(data).hexdigest()
            if not ss.current or ss.current["digest"] != digest:
                ss.current = {"data": data, "name": upload.name, "digest": digest, "origin": "upload"}
                ss.analysis, ss.overrides, ss.pending = None, {}, None
        elif ss.current and ss.current["origin"] == "upload":   # uploaded file was removed
            ss.current, ss.analysis, ss.overrides = None, None, {}

        image_ok = True
        if ss.current:
            try:
                preview = load_image(ss.current["data"])
            except Exception:
                image_ok = False
                st.error("This file could not be opened as an image. Please upload a JPG or PNG photo.")

        analyse_clicked = st.button("Analyze Label", type="primary", icon=":material/search:",
                                    width="stretch", disabled=not (ss.current and image_ok))
        st.button("New Scan", icon=":material/refresh:", width="stretch", on_click=reset,
                  disabled=ss.current is None)

        html('<div class="demo-divider"><span>Quick Demo (Sample Products)</span></div>')
        d1, d2 = st.columns(2, gap="small")
        with d1:
            st.button("Test Odonil Sample", key="demo_odonil", icon=":material/inventory_2:",
                      width="stretch", on_click=load_demo, args=("odonil",))
        with d2:
            st.button("Test Havells Sample", key="demo_havells", icon=":material/lightbulb:",
                      width="stretch", on_click=load_demo, args=("havells",))

        if ss.current and image_ok and (analyse_clicked or ss.pending):
            analyse(st)

# ---------------------------------------------------------------------------
# Right: results
# ---------------------------------------------------------------------------
def fmt_date(d: date | None, precision_month: bool) -> str:
    if d is None:
        return ""
    return d.strftime("%b %Y") if precision_month else d.strftime("%d %b %Y")


def value_state(value: str | None) -> str:
    """Green = read, amber = heading found but value unclear, grey = missing."""
    if not value:
        return "miss"
    return "unk" if value.startswith("Declared") else "ok"


def info_row(label: str, value_html: str | None, note: str | None = None) -> str:
    body = value_html if value_html else '<span class="muted">Not detected</span>'
    if note:
        body += f'<span class="note">{escape(note)}</span>'
    return f'<tr><th>{escape(label)}</th><td>{body}</td></tr>'


def pill(text: str, tone: str, manual: bool = False) -> str:
    tag = '<span class="tag-manual">MANUAL</span>' if manual else ""
    return f'<span class="pill {tone}">{escape(text)}</span>{tag}'


def render_banner(v: Verdict | None, expiry_text: str = ""):
    if v is None:
        html(f"""<div class="result-banner idle"><div class="rb-main"><div class="rb-icon">{icon('search', size=26)}</div>
          <div><div class="rb-kicker">Classification result</div><div class="rb-title">Ready to analyze</div></div></div>
          <div class="rb-reason"><div class="rb-label">Next step</div>
          <div class="rb-text">Press <b>Analyze Label</b> to read the label and classify it.</div></div></div>""")
        return
    expired = any(i.kind == "expired" for i in v.issues)
    if v.status == "COMPLIANT":
        cls, ico, title = "ok", "shield_ok", "COMPLIANT"
    elif v.status == "NON-COMPLIANT":
        cls, ico, title = "bad", "x", "NON-COMPLIANT — EXPIRED PRODUCT" if expired else "NON-COMPLIANT"
    else:
        cls, ico, title = "unknown", "warn", "Unable to confidently analyze label"
    if expired:
        reason = (f'<div class="rb-text"><b>Expiry Date:</b> {escape(expiry_text)}</div>'
                  f'<div class="rb-text"><b>Reason:</b> {escape(v.reason)}</div>')
    else:
        reason = f'<div class="rb-label">Reason</div><div class="rb-text">{escape(v.reason)}</div>'
    meta = ""
    if v.ml_probability is not None:
        conf = max(v.ml_probability, 1 - v.ml_probability)
        meta = (f'<div class="rb-meta">{escape(v.model_name or "")} · ML: {v.ml_label.title()} ({conf:.0%}) · '
                f'Date check: {"failed" if v.date_rule_failed else "passed"} · '
                f'Completeness: {v.features["label_completeness"]:.0%}</div>')
    html(f"""<div class="result-banner {cls}"><div class="rb-main"><div class="rb-icon">{icon(ico, size=28)}</div>
      <div><div class="rb-kicker">Classification result</div><div class="rb-title">{title}</div>{meta}</div></div>
      <div class="rb-reason">{reason}</div></div>""")


def date_correction(expanded: bool):
    """Manual date entry for dates OCR could not read (clearly marked as manual)."""
    mo, eo = ss.overrides.get("mfg"), ss.overrides.get("exp")
    with st.expander("Correct a misread date (manual correction)", expanded=expanded):
        st.caption("Use this only if OCR misread a printed or handwritten date. Values entered here are "
                   "marked **MANUAL** and are not OCR output. Formats: 05/2026, MAY 2026, 31/05/2026.")
        with st.form("manual_dates"):
            f1, f2 = st.columns(2)
            mfg_in = f1.text_input("Manufacturing date", value=mo[2] if mo else "", placeholder="e.g. 06/2020")
            exp_in = f2.text_input("Use-by / expiry date", value=eo[2] if eo else "", placeholder="e.g. 05/2026")
            b1, b2 = st.columns(2)
            apply_btn = b1.form_submit_button("Validate", type="primary", width="stretch")
            clear_btn = b2.form_submit_button("Clear corrections", width="stretch")
        if apply_btn:
            new, errors = parse_overrides(mfg_in, exp_in)
            if errors:
                st.error(" ".join(errors))
            else:
                ss.overrides = new
                st.rerun()
        if clear_btn:
            ss.overrides = {}
            st.rerun()


def preview_card():
    with st.container(key="preview_card"):
        html(f'<div class="card-head small">{icon("image", size=20)}<span>Uploaded Label</span></div>')
        if ss.current:
            st.image(load_image(ss.current["data"])[:, :, ::-1], width="stretch")
            html(f'<p class="file-name">{escape(ss.current["name"])}</p>')


with right:
    analysis = ss.analysis if ss.analysis and ss.current and ss.analysis["digest"] == ss.current["digest"] else None

    if ss.current is None:
        html(f"""<div class="empty"><div class="icon">{icon('ocr', size=28)}</div>
        <h3>Results will appear here</h3>
        <p>Upload a clear photo of the label panel, or try one of the sample products,
        then press <b>Analyze Label</b>.</p></div>""")
    elif analysis is None:
        render_banner(None)
        c1, _ = st.columns([1, 1.9])
        with c1:
            preview_card()
    else:
        ocr, fields = analysis["ocr"], analysis["fields"]
        d = fields.dates
        today = date.today()
        mo, eo = ss.overrides.get("mfg"), ss.overrides.get("exp")
        verdict = check_label(fields, bundle, today=today,
                              mfg_override=mo[0] if mo else None,
                              expiry_override=eo[0] if eo else None)
        exp_val = verdict.expiry_date
        mfg_txt = (fmt_date(mo[0], mo[1] == "month") if mo else
                   fmt_date(d.get("mfg_date"), d.get("mfg_precision") == "month"))
        exp_txt = (fmt_date(eo[0], eo[1] == "month") if eo else
                   fmt_date(exp_val, d.get("expiry_precision") == "month"))
        unreadable = bool(d.get("date_declared") or d.get("fragments"))
        corrections_on_top = bool(ss.overrides) or (verdict.status == "UNABLE" and unreadable)

        render_banner(verdict, exp_txt)
        if corrections_on_top:   # open only while the dates are still unreadable
            date_correction(expanded=verdict.status == "UNABLE")

        c_img, c_info, c_iss = st.columns([1, 1.32, 1], gap="small")
        with c_img:
            preview_card()

        # ---- Product information --------------------------------------
        frag = ", ".join(f"'{f}'" for f in d.get("fragments") or [])
        unread_note = "Printed on label but not readable" + (f" (OCR read {frag})" if frag else "")
        expired = bool(exp_val and exp_val < today)
        if mfg_txt:
            mfg_html = pill(mfg_txt, "blue", manual=bool(mo))
            mfg_note = None if mo else (f'read "{d.get("mfg_raw")}"' if d.get("mfg_raw") else None)
        else:
            mfg_html, mfg_note = (f'<span class="muted">{escape(unread_note)}</span>', None) if unreadable else (None, None)
        if exp_txt:
            exp_html = pill(exp_txt + (" · expired" if expired else ""), "red" if expired else "blue", manual=bool(eo))
            exp_note = None if eo else (f'read "{d.get("expiry_raw")}"' if d.get("expiry_raw") else None)
        else:
            exp_html = (f'<span class="muted">{escape(unread_note)}</span>' if unreadable
                        else '<span class="muted">Not declared on label, or not read</span>')
            exp_note = None

        def val(v):
            return escape(v) if v else None

        rows = "".join([
            info_row("Product name", val(fields.product_name),
                     "largest text on label" if fields.product_name_source == "largest text on label" else None),
            info_row("Description", val(fields.description)),
            info_row("MRP", val(fields.mrp)),
            info_row("Net quantity", val(fields.net_quantity)),
            info_row("Mfg. date", mfg_html, mfg_note),
            info_row("Use-by / expiry", exp_html, exp_note),
            info_row("Manufacturer", val(fields.manufacturer)),
            info_row("Address", val(fields.address)),
            info_row("Customer care", val(fields.customer_care)),
        ])
        with c_info:
            html(f'<div class="card"><div class="card-head small">{icon("doc", size=20)}<span>Product Information</span></div>'
                 f'<table class="info-table">{rows}</table></div>')

        # ---- Detected issues --------------------------------------------
        items = []
        for i in verdict.issues:
            if i.kind == "missing":
                continue                     # shown in the field checklist below
            tone, ico = {"expired": ("bad", "x"), "date": ("bad", "warn"),
                         "model": ("warn", "warn"), "unable": ("warn", "unk")}[i.kind]
            items.append(f'<div class="issue-hero {tone}">{icon(ico, size=20)}<div><b>{escape(i.title)}</b>'
                         f'<small>{escape(i.detail)}</small></div></div>')
        field_values = {"mrp_present": fields.mrp, "quantity_present": fields.net_quantity}
        for key, name in FIELD_NAMES.items():
            short = name.split(" /")[0].replace("Manufacturing", "Date")
            if key == "date_present":
                state = "ok" if (mfg_txt or exp_txt) else ("unk" if unreadable else "miss")
                text = {"ok": "Dates entered manually" if (mo or eo) else "Date detected",
                        "unk": "Date printed but unreadable", "miss": "Date not detected"}[state]
            elif not verdict.features.get(key):
                state, text = "miss", f"{name} not detected"
            elif value_state(field_values.get(key, "x")) == "unk":
                state, text = "unk", f"{name} declared (value unclear)"
            else:
                state, text = "ok", f"{name} detected"
            ico = {"ok": icon("ok", "ok", 20), "unk": icon("unk", "warn", 20), "miss": icon("miss", "miss", 20)}[state]
            items.append(f'<div class="check-row {state}">{ico}<span>{escape(text)}</span></div>')
        with c_iss:
            html(f'<div class="card"><div class="card-head small">{icon("warn", size=20)}<span>Detected Issues</span></div>'
                 f'{"".join(items)}</div>')

        if not corrections_on_top:
            date_correction(expanded=False)

        # ---- OCR text + pipeline internals (for the viva) -----------------
        with st.expander("View Extracted Text"):
            st.caption(f"{ocr.word_count} words · mean OCR confidence {ocr.mean_confidence:.0%} · "
                       f"OCR pass used: {ocr.pass_used} image")
            st.code(ocr.text or "(no text read)", language=None)
        with st.expander("Preprocessing steps"):
            tabs = st.tabs(list(ocr.stages.keys()))
            for tab, (name, img) in zip(tabs, ocr.stages.items()):
                with tab:
                    st.image(img if img.ndim == 2 else img[:, :, ::-1], width="stretch")
        with st.expander("Model input features"):
            feat_df = pd.DataFrame([verdict.features]).T.rename(columns={0: "value"})
            st.dataframe(feat_df, width="stretch")
            if verdict.ml_probability is not None:
                st.caption(f"{verdict.model_name} · P(non-compliant) = {verdict.ml_probability:.3f}")

# ---------------------------------------------------------------------------
# Model performance
# ---------------------------------------------------------------------------
metrics = get_metrics()
if metrics:
    best_name = metrics["best_model"]
    best = metrics["models"][best_name]
    tiles = "".join(
        f'<div class="tile {tone}">{icon(ico, size=30)}<span>{label}</span><b>{best[key]:.1%}</b></div>'
        for tone, ico, label, key in (("blue", "target", "Accuracy", "accuracy"),
                                      ("green", "layers", "Precision", "precision"),
                                      ("purple", "bars", "Recall", "recall"),
                                      ("amber", "star", "F1 Score", "f1")))
    cm = best["confusion_matrix"]
    cm_html = f"""<div class="cm"><div class="cm-title">Confusion Matrix (Test Set)</div>
      <div class="cm-body"><div class="cm-y">Actual</div><div class="cm-grid">
        <span class="cm-lab">Compliant</span><div class="cm-cell hi">{cm[0][0]}</div><div class="cm-cell lo">{cm[0][1]}</div>
        <span class="cm-lab">Non-compliant</span><div class="cm-cell lo">{cm[1][0]}</div><div class="cm-cell hi">{cm[1][1]}</div>
        <span></span><span class="cm-xlab">Compliant</span><span class="cm-xlab">Non-compliant</span>
      </div></div><div class="cm-x">Predicted</div></div>"""
    rows = "".join(
        f'<tr class="{"best" if name == best_name else ""}"><td>{escape(name)}</td>'
        f'<td>{m["accuracy"]:.3f}</td><td>{m["precision"]:.3f}</td><td>{m["recall"]:.3f}</td><td>{m["f1"]:.3f}</td></tr>'
        for name, m in metrics["models"].items())
    html(f"""<div id="model-performance" class="anchor"></div>
    <section class="perf card">
      <div class="card-head">{icon('bars', size=26)}<span>Model Performance</span></div>
      <div class="perf-grid">
        <div><div class="tiles">{tiles}</div>
          <p class="disclaimer">{escape(best_name)} selected by {escape(metrics['selection_metric'])}; evaluated on
          {metrics['test_size']} held-out rows (positive class = non-compliant). Trained on a small <b>synthetic</b>
          dataset made for this class project, so these numbers are educational, not real-world accuracy.</p></div>
        {cm_html}
        <div class="compare"><div class="cm-title left">Model Comparison</div>
          <table class="compare-table"><thead><tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th>
          <th>F1 Score</th></tr></thead><tbody>{rows}</tbody></table></div>
      </div>
    </section>""")

# ---------------------------------------------------------------------------
# About
# ---------------------------------------------------------------------------
html(f"""<div id="about" class="anchor"></div>
<section class="about card">
  <div class="card-head">{icon('info', size=24)}<span>About this project</span></div>
  <div class="about-grid">
    <div><b>1 · OpenCV preprocessing</b><span>Resize, grayscale, denoise, contrast (CLAHE), threshold.</span></div>
    <div><b>2 · EasyOCR</b><span>Reads the label text; words are grouped into lines and columns.</span></div>
    <div><b>3 · Field extraction</b><span>Regex rules find MRP, quantity, dates, manufacturer, address, customer care.</span></div>
    <div><b>4 · ML classification</b><span>Random Forest on 9 label features (compared with Logistic Regression and Decision Tree).</span></div>
    <div><b>5 · Date validation</b><span>Expired or impossible dates make the label non-compliant. Unreadable dates are never guessed.</span></div>
  </div>
  <p class="footer-note">SIH26034 · Educational demonstration. Results reflect a configured set of label checks,
  not a legal compliance assessment.</p>
</section>""")
