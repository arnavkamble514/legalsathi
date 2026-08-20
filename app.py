import os
import io
import json
import base64
import tempfile
import re
from datetime import date
from pathlib import Path

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import ollama
import fitz  # PyMuPDF
from fpdf import FPDF
from PIL import Image

VISION_MODEL  = "qwen2.5vl:7b"
REASON_MODEL  = "deepseek-r1:14b"
OUTPUT_DIR    = Path(tempfile.gettempdir()) / "legalsathi_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def strip_markdown(text: str) -> str:
    """Remove markdown formatting that LLMs emit — bold, italic, headers, bullets."""
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'__(.*?)__',     r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'_(.*?)_',   r'\1', text)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[-*]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[-*]{3,}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'`+', '', text)
    return text.strip()


app = FastAPI(title="LegalSathi", version="3.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def extract_pdf_text(file_bytes: bytes) -> str:
    """Extract text from a digital PDF using PyMuPDF (CPU, ~100% accuracy)."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        if text:
            pages.append(f"[Page {i+1}]\n{text}")
    doc.close()
    return "\n\n".join(pages) if pages else "[PDF appears to be image-based or empty]"


def extract_image_text(file_bytes: bytes, filename: str) -> str:
    """Use Qwen2.5-VL to intelligently read an image/screenshot."""
    # Convert to PNG for consistency
    img = Image.open(io.BytesIO(file_bytes))
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    prompt = (
        "You are reading a document image for a legal case. "
        "Extract ALL text you see — preserve amounts, dates, order numbers, names, and any important details. "
        "If it is a chat/email screenshot, note sender/receiver. "
        "If it is a bill/receipt, list all line items with amounts. "
        "Output clean, structured plain text. Do not add commentary."
    )

    response = ollama.chat(
        model=VISION_MODEL,
        options={"temperature": 0},
        messages=[{
            "role": "user",
            "content": prompt,
            "images": [b64]
        }]
    )
    return response["message"]["content"].strip()


def route_file(file_bytes: bytes, filename: str, content_type: str) -> str:
    """Route file to appropriate extractor based on type."""
    fname = filename.lower()
    if fname.endswith(".pdf") or content_type == "application/pdf":
        return extract_pdf_text(file_bytes)
    else:
        return extract_image_text(file_bytes, filename)


AGENT1_SYSTEM = """You are a senior consumer rights lawyer in India with 20 years of experience.
You specialize in the Consumer Protection Act 2019.
Analyze the complaint and evidence provided. Respond ONLY in valid JSON — no preamble, no markdown, no explanation.
The JSON must contain exactly these fields:
{
  "is_valid": true/false,
  "evidence_strength": "Low / Medium / High",
  "legal_validity": "Low / Medium / High",
  "recovery_likelihood": "Low / Medium / High",
  "overall_strength": "Weak / Moderate / Strong",
  "case_type": "E-commerce / Banking / Telecom / Insurance / Real Estate / Airlines / Other",
  "applicable_laws": ["Section X of Consumer Protection Act 2019 - Description", ...],
  "user_rights": ["Right 1", "Right 2", "Right 3"],
  "estimated_compensation_min": integer (INR),
  "estimated_compensation_max": integer (INR),
  "case_summary": "2-3 sentence plain language summary",
  "recommended_action": "What the user should do next"
}
Definitions for the three metric parameters:
- evidence_strength: Quality and quantity of proof the user has (bills, screenshots, recordings, emails). Low = no proof, Medium = some proof, High = strong documentary evidence.
- legal_validity: How clearly the company violated Consumer Protection Act 2019. Low = unclear violation, Medium = likely violation, High = clear and obvious violation.
- recovery_likelihood: Probability the company will pay/settle after receiving notice. Low = unlikely, Medium = possible, High = very likely within 72 hours.
- overall_strength: Derived from all three — Weak if most are Low, Moderate if mixed, Strong if most are High."""


def run_agent1(complaint: str, evidence_text: str) -> dict:
    """Agent 1: Evaluate case validity, laws, and compensation."""
    user_msg = f"""CONSUMER COMPLAINT:
{complaint}

EVIDENCE EXTRACTED FROM DOCUMENTS/IMAGES:
{evidence_text if evidence_text.strip() else "No files uploaded — proceeding on complaint text alone."}

Analyze this case and respond with the JSON evaluation."""

    response = ollama.chat(
        model=REASON_MODEL,
        options={"temperature": 0},
        messages=[
            {"role": "system", "content": AGENT1_SYSTEM},
            {"role": "user",   "content": user_msg}
        ]
    )
    raw = response["message"]["content"].strip()

    # Strip <think>...</think> blocks that DeepSeek R1 emits
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    # Extract JSON from response
    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not json_match:
        raise ValueError(f"Agent 1 did not return valid JSON. Raw: {raw[:300]}")
    return json.loads(json_match.group())


AGENT2_SYSTEM = """You are a senior advocate with 20 years of consumer law experience in India.
Draft a formal legal notice under the Consumer Protection Act 2019.
The notice must be legally intimidating, factually precise, and professionally formatted.
Output ONLY the legal notice text — no preamble, no explanation, no markdown."""


def run_agent2(case_eval: dict, user_name: str, user_address: str,
               company_name: str, evidence_text: str) -> str:
    """Agent 2: Draft the complete formal legal notice."""
    today = date.today().strftime("%d %B %Y")

    user_msg = f"""CASE EVALUATION (from Agent 1):
{json.dumps(case_eval, indent=2)}

USER DETAILS:
- Full Name: {user_name}
- Address: {user_address}

DEFENDANT:
- Company Name: {company_name}

EVIDENCE SUMMARY:
{evidence_text[:2000] if evidence_text.strip() else "Complaint text only — no documentary evidence uploaded."}

TODAY'S DATE: {today}

Draft a complete, formal legal notice. Structure it as follows:
1. Addressee block (To: The Grievance Officer / Registered Office of {company_name})
2. From block ({user_name}, {user_address})
3. Date ({today})
4. Subject: LEGAL NOTICE UNDER CONSUMER PROTECTION ACT 2019
5. Opening paragraph establishing sender's identity and purpose
6. Numbered FACTS section — what happened, dates, amounts, company's failures
7. LEGAL VIOLATIONS section — cite specific Consumer Protection Act 2019 sections
8. DEMAND section — exact compensation amount, refund, and legal costs
9. RELIEF SOUGHT section
10. 15-day deadline with warning of Consumer Disputes Redressal Commission filing
11. Signature block: {user_name}, Through: LegalSathi AI Agent

Write this as a real lawyer would — firm, professional, legally precise."""

    response = ollama.chat(
        model=REASON_MODEL,
        options={"temperature": 0},
        messages=[
            {"role": "system", "content": AGENT2_SYSTEM},
            {"role": "user",   "content": user_msg}
        ]
    )
    raw = response["message"]["content"].strip()
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    raw = strip_markdown(raw)
    return raw


AGENT2_HINDI_SYSTEM = """आप भारत में 20 वर्षों के अनुभव वाले एक वरिष्ठ अधिवक्ता हैं।
उपभोक्ता संरक्षण अधिनियम 2019 के तहत एक औपचारिक कानूनी नोटिस तैयार करें।
नोटिस कानूनी रूप से प्रभावशाली, तथ्यात्मक रूप से सटीक और पेशेवर रूप से प्रारूपित होना चाहिए।
केवल कानूनी नोटिस का पाठ आउटपुट करें — कोई प्रस्तावना नहीं, कोई स्पष्टीकरण नहीं, कोई markdown नहीं।"""


def run_agent2_hindi(case_eval: dict, user_name: str, user_address: str,
                     company_name: str, evidence_text: str) -> str:
    """Agent 2 Hindi: Draft the complete formal legal notice in Hindi."""
    today = date.today().strftime("%d %B %Y")

    user_msg = f"""केस मूल्यांकन (Agent 1 से):
{json.dumps(case_eval, indent=2)}

उपयोगकर्ता विवरण:
- पूरा नाम: {user_name}
- पता: {user_address}

प्रतिवादी:
- कंपनी का नाम: {company_name}

साक्ष्य सारांश:
{evidence_text[:2000] if evidence_text.strip() else "केवल शिकायत पाठ — कोई दस्तावेज़ी साक्ष्य अपलोड नहीं किया गया।"}

आज की तारीख: {today}

एक पूर्ण, औपचारिक कानूनी नोटिस हिंदी में तैयार करें। संरचना इस प्रकार हो:
1. प्राप्तकर्ता ब्लॉक (सेवा में: शिकायत अधिकारी / {company_name} का पंजीकृत कार्यालय)
2. प्रेषक ब्लॉक ({user_name}, {user_address})
3. दिनांक ({today})
4. विषय: उपभोक्ता संरक्षण अधिनियम 2019 के तहत कानूनी नोटिस
5. प्रारंभिक अनुच्छेद
6. क्रमांकित तथ्य अनुभाग — क्या हुआ, तारीखें, राशि, कंपनी की विफलताएं
7. कानूनी उल्लंघन अनुभाग — उपभोक्ता संरक्षण अधिनियम 2019 की विशिष्ट धाराएं
8. मांग अनुभाग — सटीक मुआवजा राशि, धनवापसी और कानूनी लागत
9. राहत की मांग
10. 15 दिन की समय सीमा के साथ उपभोक्ता विवाद निवारण आयोग में दाखिल करने की चेतावनी
11. हस्ताक्षर ब्लॉक: {user_name}, के माध्यम से: LegalSathi AI Agent

एक वास्तविक वकील की तरह लिखें — दृढ़, पेशेवर, कानूनी रूप से सटीक।"""

    response = ollama.chat(
        model=REASON_MODEL,
        options={"temperature": 0},
        messages=[
            {"role": "system", "content": AGENT2_HINDI_SYSTEM},
            {"role": "user",   "content": user_msg}
        ]
    )
    raw = response["message"]["content"].strip()
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    raw = strip_markdown(raw)
    return raw


def _get_unicode_font() -> tuple[str, str]:
    """
    Return (font_name, font_path) for a Unicode-capable TTF font.
    Tries system fonts first, then downloads DejaVu as a fallback.
    """
    candidates = [
        # Common Linux paths
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        # macOS
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        # Windows
        "C:/Windows/Fonts/arial.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            return ("UniFont", p)

    # Download DejaVuSans into the output dir as a last resort
    import urllib.request
    url = "https://github.com/dejavu-fonts/dejavu-fonts/raw/master/ttf/DejaVuSans.ttf"
    dest = OUTPUT_DIR / "DejaVuSans.ttf"
    if not dest.exists():
        urllib.request.urlretrieve(url, str(dest))
    return ("UniFont", str(dest))


def generate_pdf(notice_text: str, user_name: str, company_name: str, lang: str = "en") -> Path:
    """Convert legal notice text to a formatted PDF using FPDF2 with Unicode support."""
    font_name, font_path = _get_unicode_font()

    pdf = FPDF()
    pdf.add_font(font_name, style="",  fname=font_path)
    pdf.add_font(font_name, style="B", fname=font_path)  # FPDF2 uses same TTF for bold variant
    pdf.set_margins(25, 25, 25)
    pdf.add_page()

    def set_body(bold=False):
        pdf.set_font(font_name, style="B" if bold else "", size=11)

    def set_small(italic=False):
        pdf.set_font(font_name, style="", size=9 if italic else 8)

    # Header
    pdf.set_font(font_name, style="B", size=16)
    pdf.set_text_color(15, 40, 80)
    pdf.cell(0, 10, "LEGAL NOTICE", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font(font_name, style="", size=10)
    pdf.set_text_color(100, 100, 100)
    subtitle = "Under Consumer Protection Act 2019 | Generated by LegalSathi v3.0" if lang == "en" else "उपभोक्ता संरक्षण अधिनियम 2019 के तहत | LegalSathi v3.0 द्वारा निर्मित"
    pdf.cell(0, 6, subtitle,
             align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # Divider
    pdf.set_draw_color(15, 40, 80)
    pdf.set_line_width(0.8)
    pdf.line(25, pdf.get_y(), 185, pdf.get_y())
    pdf.ln(8)

    # Notice body
    set_body(bold=False)
    pdf.set_text_color(20, 20, 20)

    for line in notice_text.split("\n"):
        line = line.strip()
        if not line:
            pdf.ln(4)
            continue

        # Bold headings (ALL CAPS lines or special prefixes)
        is_heading = (line.isupper() and len(line) > 3) or \
                     re.match(r'^(To:|From:|Subject:|Date:|RE:)', line)
        set_body(bold=bool(is_heading))
        pdf.multi_cell(0, 6, line, new_x="LMARGIN", new_y="NEXT")

    # Footer
    pdf.ln(6)
    pdf.set_line_width(0.4)
    pdf.set_draw_color(180, 180, 180)
    pdf.line(25, pdf.get_y(), 185, pdf.get_y())
    pdf.ln(3)
    set_small()
    pdf.set_text_color(140, 140, 140)
    pdf.cell(0, 5,
             f"Generated by LegalSathi AI | {date.today().strftime('%d %B %Y')} | "
             "Send via Registered Post with Acknowledgement Due",
             align="C")

    safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", company_name)[:30]
    lang_suffix = "_Hindi" if lang == "hi" else "_English"
    out_path = OUTPUT_DIR / f"LegalNotice_{safe_name}{lang_suffix}_{date.today().strftime('%Y%m%d')}.pdf"
    pdf.output(str(out_path))
    return out_path


@app.post("/api/extract")
async def api_extract(files: list[UploadFile] = File(default=[])):
    """Extract text from uploaded files."""
    if not files or all(f.filename == "" for f in files):
        return JSONResponse({"evidence_text": "", "file_summaries": []})

    summaries = []
    combined = []

    for f in files:
        if not f.filename:
            continue
        raw = await f.read()
        try:
            text = route_file(raw, f.filename, f.content_type or "")
            combined.append(f"[File: {f.filename}]\n{text}")
            summaries.append({"filename": f.filename, "chars": len(text), "preview": text[:200]})
        except Exception as e:
            summaries.append({"filename": f.filename, "error": str(e)})

    return JSONResponse({
        "evidence_text": "\n\n---\n\n".join(combined),
        "file_summaries": summaries
    })


@app.post("/api/evaluate")
async def api_evaluate(
    complaint: str = Form(...),
    evidence_text: str = Form(default="")
):
    """Agent 1: Evaluate the case."""
    try:
        result = run_agent1(complaint, evidence_text)
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(500, f"Agent 1 error: {str(e)}")


@app.post("/api/generate")
async def api_generate(
    case_eval_json: str = Form(...),
    user_name: str = Form(...),
    user_address: str = Form(...),
    company_name: str = Form(...),
    evidence_text: str = Form(default="")
):
    """Agent 2: Draft legal notice and generate PDF."""
    try:
        case_eval = json.loads(case_eval_json)
        # English notice
        notice_en = run_agent2(case_eval, user_name, user_address, company_name, evidence_text)
        pdf_en = generate_pdf(notice_en, user_name, company_name, lang="en")
        # Hindi notice
        notice_hi = run_agent2_hindi(case_eval, user_name, user_address, company_name, evidence_text)
        pdf_hi = generate_pdf(notice_hi, user_name, company_name, lang="hi")
        return JSONResponse({
            "notice_en": notice_en,
            "notice_hi": notice_hi,
            "pdf_filename_en": pdf_en.name,
            "pdf_filename_hi": pdf_hi.name,
        })
    except Exception as e:
        raise HTTPException(500, f"Agent 2 / PDF error: {str(e)}")


@app.get("/api/download/{filename}")
async def api_download(filename: str):
    """Download the generated PDF."""
    path = OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(
        str(path),
        media_type="application/pdf",
        filename=filename
    )


@app.get("/api/health")
async def health():
    return {"status": "ok", "models": {"vision": VISION_MODEL, "reason": REASON_MODEL}}


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LegalSathi — Justice for Every Indian</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;0,900;1,700&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root {
  --ink:     #0f1a2e;
  --gold:    #c8973a;
  --gold2:   #e8b84b;
  --cream:   #faf7f2;
  --paper:   #f2ede4;
  --red:     #b83232;
  --green:   #1e6e45;
  --muted:   #8a8070;
  --border:  #d8cfc0;
  --shadow:  0 4px 24px rgba(15,26,46,0.10);
  --r:       10px;
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
  background: var(--cream);
  color: var(--ink);
  font-family: 'DM Sans', sans-serif;
  min-height: 100vh;
  overflow-x: hidden;
}

/* ── HEADER ── */
header {
  background: var(--ink);
  padding: 20px 40px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  position: sticky; top: 0; z-index: 100;
  border-bottom: 2px solid var(--gold);
}
.logo {
  display: flex; align-items: center; gap: 14px;
}
.logo-icon {
  width: 42px; height: 42px;
  background: var(--gold);
  border-radius: 8px;
  display: flex; align-items: center; justify-content: center;
  font-size: 22px;
}
.logo-text h1 {
  font-family: 'Playfair Display', serif;
  font-size: 1.5rem;
  color: var(--cream);
  letter-spacing: 0.01em;
}
.logo-text p {
  font-size: 0.72rem;
  color: var(--gold);
  letter-spacing: 0.08em;
  text-transform: uppercase;
  font-weight: 500;
}
.header-badge {
  background: rgba(200,151,58,0.15);
  border: 1px solid var(--gold);
  border-radius: 20px;
  padding: 5px 14px;
  font-size: 0.75rem;
  color: var(--gold2);
  font-weight: 500;
  letter-spacing: 0.06em;
}

/* ── HERO ── */
.hero {
  background: var(--ink);
  color: var(--cream);
  text-align: center;
  padding: 60px 40px 50px;
  border-bottom: 1px solid rgba(200,151,58,0.3);
  position: relative;
  overflow: hidden;
}
.hero::before {
  content: '';
  position: absolute; inset: 0;
  background: radial-gradient(ellipse at 50% 100%, rgba(200,151,58,0.08) 0%, transparent 70%);
}
.hero h2 {
  font-family: 'Playfair Display', serif;
  font-size: clamp(2rem, 4vw, 3.2rem);
  line-height: 1.15;
  max-width: 700px; margin: 0 auto 16px;
  position: relative;
}
.hero h2 em { color: var(--gold2); font-style: italic; }
.hero p {
  color: rgba(250,247,242,0.65);
  font-size: 1rem;
  max-width: 540px; margin: 0 auto;
  line-height: 1.6;
  position: relative;
}
.hero-stats {
  display: flex; justify-content: center; gap: 40px;
  margin-top: 32px; position: relative;
}
.stat { text-align: center; }
.stat .num {
  font-family: 'Playfair Display', serif;
  font-size: 1.8rem; font-weight: 900;
  color: var(--gold2);
}
.stat .lbl {
  font-size: 0.72rem; text-transform: uppercase;
  letter-spacing: 0.08em; color: rgba(250,247,242,0.5);
}

/* ── STEPS INDICATOR ── */
.steps-bar {
  background: var(--paper);
  border-bottom: 1px solid var(--border);
  padding: 0 40px;
  display: flex; align-items: stretch;
  overflow-x: auto;
}
.step-item {
  display: flex; align-items: center; gap: 10px;
  padding: 16px 20px;
  cursor: pointer;
  border-bottom: 3px solid transparent;
  transition: all 0.2s;
  white-space: nowrap;
  font-size: 0.85rem;
  font-weight: 500;
  color: var(--muted);
  user-select: none;
}
.step-item.active {
  color: var(--ink);
  border-bottom-color: var(--gold);
}
.step-item.done { color: var(--green); }
.step-num {
  width: 24px; height: 24px; border-radius: 50%;
  background: var(--border);
  display: flex; align-items: center; justify-content: center;
  font-size: 0.72rem; font-weight: 700;
  transition: all 0.2s;
}
.step-item.active .step-num { background: var(--gold); color: white; }
.step-item.done .step-num { background: var(--green); color: white; }
.step-sep { color: var(--border); padding: 16px 0; font-size: 0.8rem; }

/* ── MAIN LAYOUT ── */
main { max-width: 900px; margin: 0 auto; padding: 36px 24px 80px; }

/* ── PANELS ── */
.panel {
  background: white;
  border: 1px solid var(--border);
  border-radius: var(--r);
  box-shadow: var(--shadow);
  margin-bottom: 24px;
  overflow: hidden;
  animation: fadeUp 0.35s ease both;
}
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(16px); }
  to   { opacity: 1; transform: translateY(0); }
}
.panel-header {
  background: var(--paper);
  border-bottom: 1px solid var(--border);
  padding: 16px 24px;
  display: flex; align-items: center; gap: 12px;
}
.panel-header .icon { font-size: 1.2rem; }
.panel-header h3 {
  font-family: 'Playfair Display', serif;
  font-size: 1.05rem;
  font-weight: 700;
}
.panel-header .sub {
  font-size: 0.78rem; color: var(--muted);
  margin-top: 1px;
}
.panel-body { padding: 24px; }

/* ── FORM ELEMENTS ── */
label { display: block; font-size: 0.82rem; font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.06em;
  color: var(--muted); margin-bottom: 6px; }
textarea, input[type=text] {
  width: 100%;
  background: var(--cream);
  border: 1.5px solid var(--border);
  border-radius: 7px;
  padding: 12px 14px;
  font-family: 'DM Sans', sans-serif;
  font-size: 0.95rem;
  color: var(--ink);
  transition: border-color 0.2s, box-shadow 0.2s;
  resize: vertical;
}
textarea:focus, input[type=text]:focus {
  outline: none;
  border-color: var(--gold);
  box-shadow: 0 0 0 3px rgba(200,151,58,0.12);
}
textarea { min-height: 120px; }
.field-group { margin-bottom: 20px; }
.field-hint { font-size: 0.77rem; color: var(--muted); margin-top: 5px; }
.row-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }

/* ── FILE UPLOAD ── */
.upload-zone {
  border: 2px dashed var(--border);
  border-radius: var(--r);
  padding: 32px 24px;
  text-align: center;
  cursor: pointer;
  transition: all 0.2s;
  background: var(--cream);
}
.upload-zone:hover, .upload-zone.dragover {
  border-color: var(--gold);
  background: rgba(200,151,58,0.04);
}
.upload-zone .icon { font-size: 2.2rem; margin-bottom: 10px; }
.upload-zone h4 { font-size: 0.95rem; font-weight: 600; margin-bottom: 4px; }
.upload-zone p { font-size: 0.8rem; color: var(--muted); }
#fileInput { display: none; }
.file-chips {
  display: flex; flex-wrap: wrap; gap: 8px; margin-top: 16px;
}
.chip {
  background: var(--paper); border: 1px solid var(--border);
  border-radius: 20px; padding: 4px 12px;
  font-size: 0.78rem; display: flex; align-items: center; gap: 6px;
}
.chip button {
  background: none; border: none; cursor: pointer;
  color: var(--muted); font-size: 1rem; line-height: 1;
  padding: 0; transition: color 0.15s;
}
.chip button:hover { color: var(--red); }

/* ── BUTTONS ── */
.btn {
  display: inline-flex; align-items: center; gap: 8px;
  padding: 12px 28px;
  border-radius: 7px;
  font-family: 'DM Sans', sans-serif;
  font-size: 0.9rem; font-weight: 600;
  cursor: pointer; border: none;
  transition: all 0.18s;
}
.btn-primary {
  background: var(--ink); color: var(--cream);
}
.btn-primary:hover { background: #1a2d4a; transform: translateY(-1px); box-shadow: 0 4px 16px rgba(15,26,46,0.2); }
.btn-gold {
  background: var(--gold); color: white;
}
.btn-gold:hover { background: var(--gold2); transform: translateY(-1px); box-shadow: 0 4px 16px rgba(200,151,58,0.35); }
.btn-outline {
  background: transparent; color: var(--ink);
  border: 1.5px solid var(--border);
}
.btn-outline:hover { border-color: var(--ink); }
.btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none !important; }
.btn-row { display: flex; gap: 12px; align-items: center; justify-content: flex-end; margin-top: 24px; }

/* ── LOADING ── */
.spinner {
  width: 18px; height: 18px;
  border: 2px solid rgba(255,255,255,0.3);
  border-top-color: white;
  border-radius: 50%;
  animation: spin 0.7s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
.loading-bar {
  height: 4px; background: var(--border); border-radius: 2px;
  overflow: hidden; margin: 12px 0;
}
.loading-bar-fill {
  height: 100%; background: linear-gradient(90deg, var(--gold), var(--gold2));
  border-radius: 2px;
  animation: loadPulse 1.5s ease-in-out infinite;
}
@keyframes loadPulse {
  0%   { width: 0%; margin-left: 0; }
  50%  { width: 60%; margin-left: 20%; }
  100% { width: 0%; margin-left: 100%; }
}
.status-msg {
  font-size: 0.82rem; color: var(--muted);
  font-style: italic; text-align: center;
}

/* ── CASE EVALUATION RESULT ── */
.eval-grid {
  display: grid; grid-template-columns: 1fr 1fr;
  gap: 16px; margin-bottom: 20px;
}
.eval-card {
  background: var(--cream);
  border: 1px solid var(--border);
  border-radius: 8px; padding: 16px;
}
.eval-card .label {
  font-size: 0.72rem; text-transform: uppercase;
  letter-spacing: 0.07em; color: var(--muted); margin-bottom: 6px;
}
.eval-card .value {
  font-family: 'Playfair Display', serif;
  font-size: 1.25rem; font-weight: 700;
}
.score-bar {
  height: 8px; background: var(--border);
  border-radius: 4px; margin-top: 8px; overflow: hidden;
}
.score-fill {
  height: 100%; border-radius: 4px;
  background: linear-gradient(90deg, var(--gold), var(--gold2));
  transition: width 1s ease;
}
.badge {
  display: inline-block;
  padding: 3px 10px; border-radius: 12px;
  font-size: 0.75rem; font-weight: 700;
  letter-spacing: 0.05em;
}
.badge-strong { background: #d4edda; color: var(--green); }
.badge-moderate { background: #fff3cd; color: #856404; }
.badge-weak { background: #f8d7da; color: var(--red); }
.metric-pill {
  display: inline-block;
  padding: 6px 18px;
  border-radius: 20px;
  font-size: 0.88rem;
  font-weight: 700;
  margin-top: 6px;
  letter-spacing: 0.04em;
}
.metric-high   { background: #d4edda; color: var(--green); }
.metric-medium { background: #fff3cd; color: #856404; }
.metric-low    { background: #f8d7da; color: var(--red); }
.laws-list {
  list-style: none;
}
.laws-list li {
  padding: 8px 12px; background: var(--cream);
  border-left: 3px solid var(--gold);
  margin-bottom: 6px; border-radius: 0 6px 6px 0;
  font-size: 0.87rem;
}
.rights-list li {
  padding: 7px 12px; margin-bottom: 5px;
  font-size: 0.87rem;
  display: flex; align-items: flex-start; gap: 8px;
}
.rights-list li::before { content: "✓"; color: var(--green); font-weight: 700; flex-shrink: 0; }
.compensation-box {
  background: linear-gradient(135deg, #0f1a2e, #1a2d4a);
  color: white; border-radius: 8px; padding: 20px 24px;
  text-align: center; margin: 16px 0;
}
.compensation-box .cmp-label {
  font-size: 0.75rem; text-transform: uppercase;
  letter-spacing: 0.1em; color: rgba(255,255,255,0.6);
  margin-bottom: 8px;
}
.compensation-box .cmp-amount {
  font-family: 'Playfair Display', serif;
  font-size: 2rem; font-weight: 900; color: var(--gold2);
}
.compensation-box .cmp-sub {
  font-size: 0.78rem; color: rgba(255,255,255,0.5); margin-top: 4px;
}

/* ── NOTICE PREVIEW ── */
.notice-preview {
  background: var(--cream);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 28px 32px;
  font-family: 'DM Mono', monospace;
  font-size: 0.82rem;
  line-height: 1.7;
  white-space: pre-wrap;
  max-height: 480px;
  overflow-y: auto;
  color: var(--ink);
}
.notice-preview::-webkit-scrollbar { width: 6px; }
.notice-preview::-webkit-scrollbar-track { background: transparent; }
.notice-preview::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

/* ── DOWNLOAD SECTION ── */
.download-box {
  background: linear-gradient(135deg, var(--green), #2a8a5a);
  border-radius: var(--r); padding: 32px;
  text-align: center; color: white;
}
.download-box h3 {
  font-family: 'Playfair Display', serif;
  font-size: 1.5rem; margin-bottom: 8px;
}
.download-box p { opacity: 0.8; margin-bottom: 20px; font-size: 0.9rem; }
.download-box .btn {
  background: white; color: var(--green); font-size: 1rem;
  padding: 14px 36px;
}
.download-box .btn:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,0,0,0.2); }
.tip-box {
  background: rgba(255,255,255,0.15);
  border-radius: 8px; padding: 14px 18px;
  margin-top: 20px; font-size: 0.82rem; text-align: left;
  line-height: 1.6;
}

/* ── ALERT ── */
.alert {
  padding: 12px 16px; border-radius: 7px;
  font-size: 0.87rem; margin: 12px 0;
  display: flex; align-items: flex-start; gap: 10px;
}
.alert-error { background: #fff5f5; border: 1px solid #fcc; color: var(--red); }
.alert-info  { background: #f0f7ff; border: 1px solid #b8d4f0; color: #1a4a7a; }

/* ── HIDDEN ── */
.hidden { display: none !important; }

/* ── RESPONSIVE ── */
@media (max-width: 640px) {
  header { padding: 14px 20px; }
  .hero { padding: 40px 20px; }
  .hero-stats { gap: 24px; }
  main { padding: 20px 16px 60px; }
  .eval-grid { grid-template-columns: 1fr; }
  .row-2 { grid-template-columns: 1fr; }
  .steps-bar { padding: 0 16px; }
  .panel-body { padding: 16px; }
}
</style>
</head>
<body>

<header>
  <div class="logo">
    <div class="logo-icon">⚖️</div>
    <div class="logo-text">
      <h1>LegalSathi</h1>
      <p>AI Consumer Rights Agent</p>
    </div>
  </div>
  <div class="header-badge">100% Offline · Zero Cost</div>
</header>

<div class="hero">
  <h2>You Were Cheated.<br><em>Now Fight Back.</em></h2>
  <p>From complaint to court-ready legal notice in under 10 minutes — powered by local AI, completely free.</p>
  <div class="hero-stats">
    <div class="stat"><div class="num">₹0</div><div class="lbl">Cost</div></div>
    <div class="stat"><div class="num">&lt;10</div><div class="lbl">Minutes</div></div>
    <div class="stat"><div class="num">72hr</div><div class="lbl">Avg. Settlement</div></div>
    <div class="stat"><div class="num">100%</div><div class="lbl">Offline</div></div>
  </div>
</div>

<div class="steps-bar">
  <div class="step-item active" id="stp1"><div class="step-num">1</div>Your Complaint</div>
  <div class="step-sep">›</div>
  <div class="step-item" id="stp2"><div class="step-num">2</div>Case Analysis</div>
  <div class="step-sep">›</div>
  <div class="step-item" id="stp3"><div class="step-num">3</div>Your Details</div>
  <div class="step-sep">›</div>
  <div class="step-item" id="stp4"><div class="step-num">4</div>Legal Notice</div>
</div>

<main>

  <!-- ── STEP 1: COMPLAINT + FILES ── -->
  <div id="section1">
    <div class="panel">
      <div class="panel-header">
        <span class="icon">📝</span>
        <div>
          <h3>Describe Your Complaint</h3>
          <div class="sub">Plain Hindi or English — just tell us what happened</div>
        </div>
      </div>
      <div class="panel-body">
        <div class="field-group">
          <label>What happened to you? *</label>
          <textarea id="complaint" placeholder="Example: I ordered a laptop from Amazon on 10 March 2025 worth ₹65,000. It arrived damaged — the screen was cracked. I raised a return request immediately. Amazon support kept asking me to wait. After 3 weeks, they rejected my return saying the damage was 'user-induced'. I have photos of the unboxing and all chat transcripts..."></textarea>
          <div class="field-hint">Be specific — include dates, amounts, order numbers, and what the company did (or didn't do).</div>
        </div>

        <div class="field-group">
          <label>Upload Evidence (Optional)</label>
          <div class="upload-zone" id="dropZone">
            <div class="icon">📎</div>
            <h4>Drop files here or click to browse</h4>
            <p>Screenshots · Bills · PDFs · Order confirmations · Chat exports<br>PNG, JPG, PDF accepted</p>
          </div>
          <input type="file" id="fileInput" multiple accept=".pdf,.png,.jpg,.jpeg">
          <div class="file-chips" id="fileChips"></div>
        </div>

        <div id="extractStatus" class="hidden">
          <div class="loading-bar"><div class="loading-bar-fill"></div></div>
          <div class="status-msg" id="extractMsg">Reading files with Qwen2.5-VL…</div>
        </div>
        <div id="extractError" class="alert alert-error hidden"></div>

        <div class="btn-row">
          <button class="btn btn-primary" id="btnAnalyze" onclick="startAnalysis()">
            Analyze My Case →
          </button>
        </div>
      </div>
    </div>
  </div>

  <!-- ── STEP 2: EVALUATION RESULT ── -->
  <div id="section2" class="hidden">
    <div class="panel">
      <div class="panel-header">
        <span class="icon">⚡</span>
        <div>
          <h3>Case Evaluation</h3>
          <div class="sub">Powered by DeepSeek R1:14b Legal Reasoning</div>
        </div>
      </div>
      <div class="panel-body">

        <div id="evalLoading">
          <div class="loading-bar"><div class="loading-bar-fill"></div></div>
          <div class="status-msg">DeepSeek R1 is analyzing your case — this takes 30–90 seconds…</div>
        </div>

        <div id="evalResult" class="hidden">

          <!-- Three Metric Parameters -->
          <div class="eval-grid" style="grid-template-columns:1fr 1fr 1fr;margin-bottom:12px">
            <div class="eval-card" style="text-align:center">
              <div class="label">Evidence Strength</div>
              <div id="metricEvidence" class="metric-pill">—</div>
              <div style="font-size:0.72rem;color:var(--muted);margin-top:6px">Quality of your proof</div>
            </div>
            <div class="eval-card" style="text-align:center">
              <div class="label">Legal Validity</div>
              <div id="metricLegal" class="metric-pill">—</div>
              <div style="font-size:0.72rem;color:var(--muted);margin-top:6px">Strength of law violation</div>
            </div>
            <div class="eval-card" style="text-align:center">
              <div class="label">Recovery Likelihood</div>
              <div id="metricRecovery" class="metric-pill">—</div>
              <div style="font-size:0.72rem;color:var(--muted);margin-top:6px">Chance company settles</div>
            </div>
          </div>
          <!-- Overall + Case Type + Action -->
          <div class="eval-grid">
            <div class="eval-card">
              <div class="label">Overall Case Strength</div>
              <div class="value" id="evalStrength">—</div>
              <div id="strengthBadge"></div>
            </div>
            <div class="eval-card">
              <div class="label">Case Type</div>
              <div class="value" style="font-size:1rem" id="evalType">—</div>
            </div>
            <div class="eval-card" style="grid-column:span 2">
              <div class="label">Recommended Action</div>
              <div class="value" style="font-size:0.88rem;line-height:1.4" id="evalAction">—</div>
            </div>
          </div>

          <div class="compensation-box">
            <div class="cmp-label">Estimated Claimable Compensation</div>
            <div class="cmp-amount" id="evalComp">—</div>
            <div class="cmp-sub">Range based on Consumer Protection Act 2019 precedents</div>
          </div>

          <div class="field-group">
            <label>Case Summary</label>
            <div id="evalSummary" style="font-size:0.92rem;line-height:1.6;color:var(--ink);padding:14px;background:var(--cream);border-radius:7px;border:1px solid var(--border)"></div>
          </div>

          <div class="row-2">
            <div class="field-group">
              <label>Applicable Laws</label>
              <ul class="laws-list" id="evalLaws"></ul>
            </div>
            <div class="field-group">
              <label>Your Rights</label>
              <ul class="laws-list rights-list" id="evalRights"></ul>
            </div>
          </div>

          <div class="btn-row">
            <button class="btn btn-outline" onclick="goBack(1)">← Edit Complaint</button>
            <button class="btn btn-gold" onclick="showSection(3)">Proceed to Notice →</button>
          </div>
        </div>

        <div id="evalError" class="alert alert-error hidden"></div>
      </div>
    </div>
  </div>

  <!-- ── STEP 3: USER DETAILS ── -->
  <div id="section3" class="hidden">
    <div class="panel">
      <div class="panel-header">
        <span class="icon">👤</span>
        <div>
          <h3>Your Details & Defendant</h3>
          <div class="sub">This information will appear on the legal notice</div>
        </div>
      </div>
      <div class="panel-body">
        <div class="row-2">
          <div class="field-group">
            <label>Your Full Name *</label>
            <input type="text" id="userName" placeholder="Rahul Kumar Sharma">
          </div>
          <div class="field-group">
            <label>Company / Defendant Name *</label>
            <input type="text" id="companyName" placeholder="Amazon Seller Services Pvt. Ltd.">
          </div>
        </div>
        <div class="field-group">
          <label>Your Full Address *</label>
          <textarea id="userAddress" style="min-height:80px" placeholder="Flat 4B, Sunrise Apartments, MG Road, Pune – 411001, Maharashtra"></textarea>
        </div>
        <div class="alert alert-info">
          📮 After downloading, send the notice via <strong>Registered Post with Acknowledgement Due (RPAD)</strong> to the company's registered office and grievance officer.
        </div>
        <div class="btn-row">
          <button class="btn btn-outline" onclick="goBack(2)">← Back</button>
          <button class="btn btn-gold" id="btnGenerate" onclick="generateNotice()">Generate Legal Notice →</button>
        </div>
      </div>
    </div>
  </div>

  <!-- ── STEP 4: NOTICE + DOWNLOAD ── -->
  <div id="section4" class="hidden">
    <div class="panel">
      <div class="panel-header">
        <span class="icon">📄</span>
        <div>
          <h3>Legal Notice — Ready</h3>
          <div class="sub">Drafted by DeepSeek R1 under Consumer Protection Act 2019</div>
        </div>
      </div>
      <div class="panel-body">

        <div id="genLoading">
          <div class="loading-bar"><div class="loading-bar-fill"></div></div>
          <div class="status-msg">Agent 2 is drafting your legal notice — please wait…</div>
        </div>

        <div id="genResult" class="hidden">
          <div class="download-box" id="downloadBox">
            <h3>🎉 Your Legal Notice is Ready</h3>
            <p>A professionally formatted PDF has been generated. Download and send via Registered Post.</p>
            <button class="btn" id="btnDownload" onclick="downloadPDF()">⬇ Download PDF</button>
            <div class="tip-box">
              <strong>Next Steps:</strong><br>
              1. Print two copies of this notice<br>
              2. Send one via <strong>Registered Post with Acknowledgement Due (RPAD)</strong><br>
              3. Keep the other copy with postal receipt<br>
              4. Company has <strong>15 days</strong> to respond — most settle within 72 hours<br>
              5. If ignored → file at Consumer Disputes Redressal Commission (free under ₹50L)
            </div>
          </div>

          <div class="field-group" style="margin-top:24px">
            <label>Notice Preview</label>
            <div class="notice-preview" id="noticePreview"></div>
          </div>

          <div class="btn-row">
            <button class="btn btn-outline" onclick="resetAll()">Start New Case</button>
            <button class="btn btn-primary" onclick="downloadPDF()">⬇ Download PDF</button>
          </div>
        </div>

        <div id="genError" class="alert alert-error hidden"></div>
      </div>
    </div>
  </div>

</main>

<script>
// ─── STATE ───
let state = {
  evidenceText: '',
  caseEval: null,
  pdfFilename: null,
  files: []
};

// ─── STEP NAV ───
function showSection(n) {
  [1,2,3,4].forEach(i => {
    document.getElementById('section' + i).classList.toggle('hidden', i !== n);
    const s = document.getElementById('stp' + i);
    s.classList.remove('active','done');
    if (i < n) s.classList.add('done');
    else if (i === n) s.classList.add('active');
  });
  window.scrollTo({ top: 0, behavior: 'smooth' });
}
function goBack(n) { showSection(n); }

// ─── FILE HANDLING ───
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');

dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('dragover');
  addFiles([...e.dataTransfer.files]);
});
fileInput.addEventListener('change', () => addFiles([...fileInput.files]));

function addFiles(newFiles) {
  newFiles.forEach(f => {
    if (!state.files.find(x => x.name === f.name && x.size === f.size)) state.files.push(f);
  });
  renderChips();
}
function removeFile(i) {
  state.files.splice(i, 1);
  renderChips();
}
function renderChips() {
  const container = document.getElementById('fileChips');
  container.innerHTML = state.files.map((f, i) =>
    `<div class="chip">
      <span>${f.name.length > 28 ? f.name.slice(0,25)+'…' : f.name}</span>
      <button onclick="removeFile(${i})" title="Remove">×</button>
    </div>`
  ).join('');
}

// ─── STEP 1 → 2: EXTRACT + ANALYZE ───
async function startAnalysis() {
  const complaint = document.getElementById('complaint').value.trim();
  if (!complaint) { alert('Please describe your complaint.'); return; }

  document.getElementById('btnAnalyze').disabled = true;
  showSection(2);

  // Extract files
  if (state.files.length > 0) {
    document.getElementById('extractStatus').classList.remove('hidden');
    document.getElementById('extractMsg').textContent =
      `Reading ${state.files.length} file(s) with AI vision…`;

    const fd = new FormData();
    state.files.forEach(f => fd.append('files', f));

    try {
      const r = await fetch('/api/extract', { method: 'POST', body: fd });
      const d = await r.json();
      state.evidenceText = d.evidence_text || '';
      document.getElementById('extractStatus').classList.add('hidden');
    } catch(e) {
      document.getElementById('extractError').textContent = 'File reading failed: ' + e.message;
      document.getElementById('extractError').classList.remove('hidden');
      document.getElementById('extractStatus').classList.add('hidden');
    }
  }

  // Agent 1
  document.getElementById('evalLoading').classList.remove('hidden');
  const fd2 = new FormData();
  fd2.append('complaint', complaint);
  fd2.append('evidence_text', state.evidenceText);

  try {
    const r = await fetch('/api/evaluate', { method: 'POST', body: fd2 });
    if (!r.ok) throw new Error(await r.text());
    const d = await r.json();
    state.caseEval = d;
    renderEval(d);
    document.getElementById('evalLoading').classList.add('hidden');
    document.getElementById('evalResult').classList.remove('hidden');
  } catch(e) {
    document.getElementById('evalLoading').classList.add('hidden');
    document.getElementById('evalError').textContent = 'Analysis error: ' + e.message;
    document.getElementById('evalError').classList.remove('hidden');
  }

  document.getElementById('btnAnalyze').disabled = false;
}

function renderMetricPill(elementId, value) {
  const el = document.getElementById(elementId);
  const v = (value || '').toLowerCase();
  const cls = v === 'high' ? 'metric-high' : v === 'medium' ? 'metric-medium' : 'metric-low';
  const label = v === 'high' ? 'High' : v === 'medium' ? 'Medium' : v === 'low' ? 'Low' : '—';
  el.textContent = label;
  el.className = 'metric-pill ' + cls;
}

function renderEval(d) {
  // Three metric parameters
  renderMetricPill('metricEvidence', d.evidence_strength);
  renderMetricPill('metricLegal',    d.legal_validity);
  renderMetricPill('metricRecovery', d.recovery_likelihood);

  // Overall strength
  const strength = d.overall_strength || d.strength || '—';
  document.getElementById('evalStrength').textContent = strength;
  const badgeMap = { Strong: 'badge-strong', Moderate: 'badge-moderate', Weak: 'badge-weak' };
  document.getElementById('strengthBadge').innerHTML =
    `<span class="badge ${badgeMap[strength] || ''}" style="margin-top:6px">${d.is_valid ? 'Valid Case' : 'Weak / Invalid'}</span>`;

  document.getElementById('evalType').textContent = d.case_type || '—';
  document.getElementById('evalAction').textContent = d.recommended_action || '—';
  document.getElementById('evalSummary').textContent = d.case_summary || '—';

  const min = (d.estimated_compensation_min || 0).toLocaleString('en-IN');
  const max = (d.estimated_compensation_max || 0).toLocaleString('en-IN');
  document.getElementById('evalComp').textContent = `\u20b9${min} \u2013 \u20b9${max}`;

  const lawsList = document.getElementById('evalLaws');
  lawsList.innerHTML = (d.applicable_laws || []).map(l => `<li>${l}</li>`).join('');

  const rightsList = document.getElementById('evalRights');
  rightsList.innerHTML = (d.user_rights || []).map(r => `<li>${r}</li>`).join('');
}

// ─── STEP 3 → 4: GENERATE ───
async function generateNotice() {
  const name    = document.getElementById('userName').value.trim();
  const address = document.getElementById('userAddress').value.trim();
  const company = document.getElementById('companyName').value.trim();
  if (!name || !address || !company) { alert('Please fill in all fields.'); return; }

  document.getElementById('btnGenerate').disabled = true;
  showSection(4);
  document.getElementById('genLoading').classList.remove('hidden');
  document.getElementById('genResult').classList.add('hidden');
  document.getElementById('genError').classList.add('hidden');

  const fd = new FormData();
  fd.append('case_eval_json', JSON.stringify(state.caseEval));
  fd.append('user_name', name);
  fd.append('user_address', address);
  fd.append('company_name', company);
  fd.append('evidence_text', state.evidenceText);

  try {
    const r = await fetch('/api/generate', { method: 'POST', body: fd });
    if (!r.ok) throw new Error(await r.text());
    const d = await r.json();
    state.pdfFilename = d.pdf_filename;
    document.getElementById('noticePreview').textContent = d.notice_text;
    document.getElementById('genLoading').classList.add('hidden');
    document.getElementById('genResult').classList.remove('hidden');
  } catch(e) {
    document.getElementById('genLoading').classList.add('hidden');
    document.getElementById('genError').textContent = 'Generation error: ' + e.message;
    document.getElementById('genError').classList.remove('hidden');
  }

  document.getElementById('btnGenerate').disabled = false;
}

function downloadPDF() {
  if (!state.pdfFilename) { alert('PDF not ready.'); return; }
  window.open('/api/download/' + state.pdfFilename, '_blank');
}

function resetAll() {
  state = { evidenceText: '', caseEval: null, pdfFilename: null, files: [] };
  document.getElementById('complaint').value = '';
  document.getElementById('userName').value = '';
  document.getElementById('userAddress').value = '';
  document.getElementById('companyName').value = '';
  renderChips();
  showSection(1);
}
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(HTML)


if __name__ == "__main__":
    import webbrowser, threading, time

    def open_browser():
        time.sleep(1.5)
        webbrowser.open("http://localhost:8000")

    threading.Thread(target=open_browser, daemon=True).start()

    print("\n" + "="*55)
    print("  ⚖️  LegalSathi v3.0  —  Justice for Every Indian")
    print("="*55)
    print(f"  Vision Model : {VISION_MODEL}")
    print(f"  Reason Model : {REASON_MODEL}")
    print(f"  Output Dir   : {OUTPUT_DIR}")
    print("="*55)
    print("  → http://localhost:8000")
    print("="*55 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
