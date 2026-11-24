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

from compliance_checker import Verdict, check_label, load_model
from feature_extractor import extract_fields, month_end, parse_user_date
from ocr_engine import get_reader, load_image, run_ocr

ROOT = Path(__file__).parent
st.set_page_config(page_title="Label Compliance · SIH26034", page_icon="🔎", layout="wide")
st.markdown(f"<style>{(ROOT / 'assets' / 'style.css').read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Icons (inline SVG, stroke = currentColor)
# ---------------------------------------------------------------------------
_P = {
    "ok": '<circle cx="12" cy="12" r="9.5"/><path d="m8.5 12 2.5 2.5 4.5-5"/>',
    "miss": '<circle cx="12" cy="12" r="9.5"/><path d="M8 12h8"/>',
    "bad": '<circle cx="12" cy="12" r="9.5"/><path d="M12 7v5l3 2"/>',
    "warn": '<path d="M10.3 3.9 2.5 17.5A2 2 0 0 0 4.2 20.5h15.6a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
    "unk": '<circle cx="12" cy="12" r="9.5"/><path d="M9.1 9a3 3 0 0 1 5.8 1c0 2-3 3-3 3"/><path d="M12 17h.01"/>',
    "shield_ok": '<path d="M12 2.5 4 5.5v6c0 5 3.4 8.6 8 10 4.6-1.4 8-5 8-10v-6z"/><path d="m8.5 12 2.5 2.5 4.5-5"/>',
    "shield_bad": '<path d="M12 2.5 4 5.5v6c0 5 3.4 8.6 8 10 4.6-1.4 8-5 8-10v-6z"/><path d="m9.5 9.5 5 5"/><path d="m14.5 9.5-5 5"/>',
    "scan": '<path d="M3 7V5a2 2 0 0 1 2-2h2"/><path d="M17 3h2a2 2 0 0 1 2 2v2"/><path d="M21 17v2a2 2 0 0 1-2 2h-2"/><path d="M7 21H5a2 2 0 0 1-2-2v-2"/><path d="M7 9h10M7 12h10M7 15h6"/>',
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
# ---------------------------------------------------------------------------
ss = st.session_state
ss.setdefault("uploader_key", 0)
ss.setdefault("analysis", None)       # OCR + extracted fields for the current image
ss.setdefault("overrides", {})        # manual date corrections


def reset():
    ss.uploader_key += 1
    ss.analysis = None
    ss.overrides = {}


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
html(f"""
<div class="app-header">
  <div class="brand">
    <div class="brand-mark">{icon('scan', size=24)}</div>
    <div>
      <h1>Packaged Commodity Compliance Classification</h1>
      <p>Intelligent Product Label Inspection</p>
    </div>
  </div>
  <div class="code-chip">PROJECT SIH26034</div>
</div>
""")

try:
    bundle = get_model()
except FileNotFoundError as exc:
    st.error(f"{exc}")
    st.stop()

left, right = st.columns([5, 7], gap="large")

# ---------------------------------------------------------------------------
# Left: upload + preview
# ---------------------------------------------------------------------------
with left:
    html('<p class="section-title">Upload a Product Label</p>'
         '<p class="section-sub">Drag and drop or browse · JPG, JPEG, PNG (HEIC also accepted)</p>')
    upload = st.file_uploader("Product label image", type=["jpg", "jpeg", "png", "heic", "heif"],
                              key=f"uploader_{ss.uploader_key}", label_visibility="collapsed")

    image = None
    if upload is not None:
        data = upload.getvalue()
        digest = hashlib.md5(data).hexdigest()
        if ss.analysis and ss.analysis["digest"] != digest:   # a different image was chosen
            ss.analysis, ss.overrides = None, {}
        try:
            image = load_image(data)
        except Exception:
            st.error("This file could not be opened as an image. Please upload a JPG or PNG photo.")

    c1, c2 = st.columns([3, 2])
    analyse = c1.button("Analyze Label", type="primary", width="stretch", disabled=image is None)
    c2.button("New scan", on_click=reset, width="stretch", disabled=upload is None)

    if analyse and image is not None:
        with st.status("Analysing label…", expanded=True) as status:
            try:
                st.write("Preprocessing image and reading text (first run loads the OCR model)…")
                ocr = run_ocr(image, reader=get_ocr_reader())
                st.write(f"OCR read {ocr.word_count} words · mean confidence {ocr.mean_confidence:.0%}")
                st.write("Extracting label fields and dates…")
                fields = extract_fields(ocr.text, ocr.largest_text)
                st.write("Running ML model and date checks…")
                ss.analysis = {"digest": digest, "ocr": ocr, "fields": fields}
                ss.overrides = {}
                status.update(label="Analysis complete", state="complete", expanded=False)
            except Exception as exc:  # keep the app alive on any OCR failure
                ss.analysis = None
                status.update(label="Analysis failed", state="error")
                st.error(f"Could not analyse this image: {exc}")

    if image is not None:
        html('<div class="preview-gap"></div>')
        st.image(image[:, :, ::-1], width="stretch")

    html("""<div class="pipeline"><span>OpenCV preprocessing</span><i>→</i><span>EasyOCR</span><i>→</i>
    <span>Field extraction</span><i>→</i><span>ML model</span><i>→</i><span>Date validation</span></div>""")

# ---------------------------------------------------------------------------
# Right: results
# ---------------------------------------------------------------------------
def fmt_date(d: date | None, precision_month: bool) -> str:
    if d is None:
        return ""
    return d.strftime("%b %Y") if precision_month else d.strftime("%d %b %Y")


def info_row(label: str, value: str | None, state: str, source: str | None = None, manual=False,
             missing_text: str | None = None) -> str:
    ico = {"ok": icon("ok", "ok"), "miss": icon("miss", "miss"), "bad": icon("bad", "bad"),
           "unk": icon("unk", "warn")}[state]
    if value:
        v = escape(value) + ('<span class="tag-manual">manual</span>' if manual else "")
        src = f'<span class="src">{escape(source)}</span>' if source else ""
        body = f'<div class="v">{v}{src}</div>'
    else:
        body = f'<div class="v muted">{escape(missing_text or "Not detected")}</div>'
    return f'<div class="info-row">{ico}<div class="k">{escape(label)}</div>{body}</div>'


def value_state(value: str | None) -> str:
    """Green = read, amber = heading found but value unclear, grey = missing."""
    if not value:
        return "miss"
    return "unk" if value.startswith("Declared") else "ok"


def render_verdict(v: Verdict, expiry_text: str = ""):
    expired = any(i.kind == "expired" for i in v.issues)
    reason = escape(v.reason)
    if v.status == "COMPLIANT":
        cls, ico, title = "ok", "shield_ok", "COMPLIANT"
    elif v.status == "NON-COMPLIANT" and expired:
        cls, ico, title = "bad", "shield_bad", "NON-COMPLIANT — EXPIRED PRODUCT"
        reason = f"<b>Expiry Date:</b> {escape(expiry_text)}<br><b>Reason:</b> {reason}"
    elif v.status == "NON-COMPLIANT":
        cls, ico, title = "bad", "shield_bad", "NON-COMPLIANT"
    else:
        cls, ico, title = "unknown", "warn", "Unable to confidently analyze label"
    meta = ""
    if v.ml_probability is not None:
        meta = (f'<div class="v-meta"><span>Model: {escape(v.model_name or "")}</span>'
                f'<span>ML prediction: {v.ml_label.title()} ({max(v.ml_probability, 1 - v.ml_probability):.0%})</span>'
                f'<span>Date check: {"failed" if v.date_rule_failed else "passed"}</span>'
                f'<span>Completeness: {v.features["label_completeness"]:.0%}</span></div>')
    html(f"""<div class="verdict {cls}"><div class="v-icon">{icon(ico, size=24)}</div><div>
      <p class="v-kicker">Classification result</p><p class="v-title">{title}</p>
      <p class="v-reason">{reason}</p>{meta}</div></div>""")


def date_correction(expanded: bool):
    """Manual date entry for dates OCR could not read (clearly marked as manual)."""
    mo, eo = ss.overrides.get("mfg"), ss.overrides.get("exp")
    with st.expander("Correct a misread date (manual correction)", expanded=expanded):
        st.caption("Use this only if OCR misread a printed or handwritten date. Values entered here are "
                   "marked **manual** and are not OCR output. Formats: 05/2026, MAY 2026, 31/05/2026.")
        with st.form("manual_dates"):
            f1, f2 = st.columns(2)
            mfg_in = f1.text_input("Manufacturing date", value=mo[2] if mo else "", placeholder="e.g. 06/2020")
            exp_in = f2.text_input("Use-by / expiry date", value=eo[2] if eo else "", placeholder="e.g. 05/2026")
            b1, b2 = st.columns(2)
            apply_btn = b1.form_submit_button("Validate", type="primary", width="stretch")
            clear_btn = b2.form_submit_button("Clear corrections", width="stretch")
        if apply_btn:
            new, errors = {}, []
            for key, raw, name in (("mfg", mfg_in, "Manufacturing date"), ("exp", exp_in, "Expiry date")):
                if raw.strip():
                    parsed = parse_user_date(raw)
                    if parsed:
                        value, precision = parsed
                        if key == "exp" and precision == "month":   # use-by month = valid through month end
                            value = month_end(value)
                        new[key] = (value, precision, raw.strip())
                    else:
                        errors.append(f"{name}: could not understand '{raw}'.")
            if errors:
                st.error(" ".join(errors))
            else:
                ss.overrides = new
                st.rerun()
        if clear_btn:
            ss.overrides = {}
            st.rerun()


with right:
    if ss.analysis is None:
        html(f"""<div class="empty"><div class="icon">{icon('scan', size=26)}</div>
        <h3>Results will appear here</h3>
        <p>Upload a clear photo of the label panel that shows MRP, net quantity, dates and manufacturer details, then press <b>Analyze Label</b>.</p></div>""")
    else:
        ocr, fields = ss.analysis["ocr"], ss.analysis["fields"]
        today = date.today()
        mo, eo = ss.overrides.get("mfg"), ss.overrides.get("exp")
        verdict = check_label(fields, bundle, today=today,
                              mfg_override=mo[0] if mo else None,
                              expiry_override=eo[0] if eo else None)
        dd = fields.dates
        corrections_on_top = bool(ss.overrides) or (
            verdict.status == "UNABLE" and bool(dd.get("date_declared") or dd.get("fragments")))
        exp_display = (fmt_date(eo[0], eo[1] == "month") if eo else
                       fmt_date(verdict.expiry_date, dd.get("expiry_precision") == "month"))
        render_verdict(verdict, exp_display)
        if corrections_on_top:
            date_correction(expanded=True)

        # ---- Product information --------------------------------------
        d = fields.dates
        mfg_txt = (fmt_date(mo[0], mo[1] == "month") if mo else
                   fmt_date(d.get("mfg_date"), d.get("mfg_precision") == "month"))
        exp_val = verdict.expiry_date
        exp_txt = (fmt_date(eo[0], eo[1] == "month") if eo else
                   fmt_date(exp_val, d.get("expiry_precision") == "month"))
        exp_state = "bad" if exp_val and exp_val < today else ("ok" if exp_val else "miss")
        # Label declares a date but OCR could not read it -> amber "unable to detect".
        unreadable = d.get("date_declared") or d.get("fragments")
        frag = ", ".join(f"'{f}'" for f in d.get("fragments") or [])
        unreadable_note = "printed on label but not readable" + (f" · OCR fragments: {frag}" if frag else "")
        mfg_state = "ok" if mfg_txt else ("unk" if unreadable else "miss")
        if not exp_val and unreadable:
            exp_state = "unk"
        rows = [
            info_row("Product name", fields.product_name, "ok" if fields.product_name else "miss",
                     f"from {fields.product_name_source}" if fields.product_name_source else None),
            info_row("Description", fields.description, "ok" if fields.description else "miss"),
            info_row("MRP", fields.mrp, value_state(fields.mrp)),
            info_row("Net quantity", fields.net_quantity, value_state(fields.net_quantity)),
            info_row("Mfg. date", mfg_txt or None, mfg_state,
                     "manual correction — not OCR output" if mo else
                     (f'read "{d.get("mfg_raw")}" · {d.get("mfg_source")}' if d.get("mfg_raw") else None),
                     manual=bool(mo), missing_text=unreadable_note if mfg_state == "unk" else None),
            info_row("Use-by / expiry", (exp_txt + ("  · expired" if exp_state == "bad" else "")) or None, exp_state,
                     "manual correction — not OCR output" if eo else
                     (f'read "{d.get("expiry_raw")}" · {d.get("expiry_source")}' if d.get("expiry_raw") else None),
                     manual=bool(eo),
                     missing_text=unreadable_note if exp_state == "unk" else "Not declared on label, or not read"),
            info_row("Manufacturer", fields.manufacturer, "ok" if fields.manufacturer else "miss"),
            info_row("Address", fields.address, "ok" if fields.address else "miss"),
            info_row("Customer care", fields.customer_care, "ok" if fields.customer_care else "miss"),
        ]
        html(f'<div class="panel"><h4>Product information</h4>{"".join(rows)}</div>')

        # ---- Detected issues --------------------------------------------
        kind_icon = {"expired": icon("bad", "bad"), "date": icon("warn", "bad"), "missing": icon("miss", "miss"),
                     "model": icon("warn", "warn"), "unable": icon("unk", "warn")}
        if verdict.issues:
            items = "".join(f'<div class="issue">{kind_icon[i.kind]}<div><b>{escape(i.title)}</b>'
                            f'<small>{escape(i.detail)}</small></div></div>' for i in verdict.issues)
        else:
            items = f'<p class="no-issues">{icon("ok", "ok")} No configured issues detected.</p>'
        html(f'<div class="panel"><h4>Detected issues</h4>{items}</div>')

        if not corrections_on_top:
            date_correction(expanded=False)

        # ---- OCR text + pipeline internals ------------------------------
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
# Model performance (small, below the fold)
# ---------------------------------------------------------------------------
metrics = get_metrics()
if metrics:
    st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
    html('<p class="section-label">Model performance</p>')
    best = metrics["models"][metrics["best_model"]]
    m_left, m_right = st.columns([7, 5], gap="large")
    with m_left:
        html(f"""<p class="section-title">{escape(metrics['best_model'])}</p>
        <p class="section-sub">Selected by {escape(metrics['selection_metric'])} · evaluated on {metrics['test_size']} held-out test rows · positive class = non-compliant</p>
        <div class="metric-grid">
          <div class="metric"><div class="m-label">Accuracy</div><div class="m-value">{best['accuracy']:.1%}</div></div>
          <div class="metric"><div class="m-label">Precision</div><div class="m-value">{best['precision']:.1%}</div></div>
          <div class="metric"><div class="m-label">Recall</div><div class="m-value">{best['recall']:.1%}</div></div>
          <div class="metric"><div class="m-label">F1 score</div><div class="m-value">{best['f1']:.1%}</div></div>
        </div>
        <p class="disclaimer">Trained on a small <b>synthetic</b> dataset generated for this class project.
        These numbers are educational and do not represent real-world regulatory accuracy.</p>""")
        comp = pd.DataFrame(metrics["models"]).T[["accuracy", "precision", "recall", "f1", "cv_f1_mean"]]
        comp.columns = ["Accuracy", "Precision", "Recall", "F1", "CV F1 (mean)"]
        with st.expander("Compare all three models"):
            st.dataframe(comp.style.format("{:.3f}"), width="stretch")
    with m_right:
        cm_path = ROOT / "evaluation" / "confusion_matrix.png"
        if cm_path.exists():
            st.image(str(cm_path), caption="Confusion matrix (test set)", width="stretch")

html('<p class="footer-note">SIH26034 · Educational demonstration. Results reflect a configured set of label checks, '
     'not a legal compliance assessment.</p>')
