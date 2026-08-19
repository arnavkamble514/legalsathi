# LegalSathi v3.0

**An offline AI agent that turns a consumer complaint into a court-ready legal notice.**

India has over 500 million internet consumers. A lot of them get wronged by companies (damaged deliveries, wrongful charges, refused refunds) and never fight back, because they assume the legal process needs a lawyer and money they don't have.

It doesn't. The Consumer Protection Act 2019 lets any citizen send a legal notice directly. The barrier isn't the law, it's knowing how to write one.

LegalSathi closes that gap. You describe what happened in plain Hindi or English, upload whatever proof you have, and in under ten minutes you get a properly formatted legal notice PDF in both languages, ready to print and post. It costs nothing to run and never sends your data anywhere.

---

## How it works

```
Browser (localhost:8000)
      |  HTTP
FastAPI backend  ──  PyMuPDF        (digital PDFs -> text, CPU)
      |             Qwen2.5-VL:7b   (screenshots/bills -> text, vision model)
Ollama (localhost:11434)
      |  CUDA
Local GPU
```

**Four steps, two AI agents:**

1. **You describe the complaint** and upload evidence: screenshots, bills, order confirmations, PDFs.
2. **Evidence gets read.** Digital PDFs go through PyMuPDF for exact text extraction. Images go through Qwen2.5-VL, a vision-language model that reads screenshots the way a person would, picking up tables, layout, and Hindi text instead of just pixels.
3. **Agent 1 evaluates the case.** DeepSeek R1:14b assesses it across three parameters and identifies which sections of the Consumer Protection Act 2019 apply, plus a compensation estimate.
4. **Agent 2 drafts the notice.** It takes Agent 1's findings as context and writes the full formal notice in English and Hindi, which FPDF2 turns into a downloadable PDF.

---

## Design decisions worth explaining

**Why two agents instead of one prompt.** Asking a single model to both analyze a case and write a legal document gives worse output on both fronts. Agent 1 does pure analysis and emits structured JSON. Agent 2 does pure legal drafting, using that JSON as input. It's basically how a law firm splits research from brief-writing.

**Why fully offline.** Complaints contain names, addresses, financial details, grievances. Sending that to a third-party API is the wrong default for the people this tool is built for. Running locally also means zero cost per use, which matters when your users are the ones who couldn't afford a lawyer in the first place.

**Why three categorical parameters instead of a score out of 100.** The first version returned a validity score, and it was unstable: the same complaint would score 75 one run and 82 the next. The cause wasn't the model, it was the output schema. A hundred possible values gives the model too much room to wobble. Swapping that for three fields with three values each (Evidence Strength, Legal Validity, Recovery Likelihood) cut the output space from 100 combinations down to 27 and made results consistent. It's also more useful. "Evidence Strength: Low" tells a user exactly what to fix. "62/100" tells them nothing.

**Why temperature=0.** Both agents run deterministically, so the same complaint always produces the same evaluation and the same notice.

**Why a Unicode TTF font pipeline.** FPDF2's built-in fonts are Latin-only. The rupee symbol (U+20B9) crashes them outright, and Devanagari won't render at all. The app checks for a system Unicode font and falls back to downloading DejaVuSans if it doesn't find one. Anything built for Indian users needs this.

---

## Are the notices legally valid?

Yes. The Consumer Protection Act 2019 doesn't require an advocate to draft a notice; any aggrieved consumer can send one directly. A valid notice identifies both parties, states the facts, cites the relevant statute, makes a specific demand, and sets a deadline. LegalSathi produces all of that, and tells the user to send it via Registered Post with Acknowledgement Due, so there's a legally admissible paper trail.

The honest limitation: on niche or unusual cases, the model can cite a section number wrong. I limit this by restricting citations to CPA 2019 statute sections only, never to specific court judgments, which are much harder to verify. For high-value disputes, get a lawyer to review before sending.

---

## Running it

**Prerequisites:** Python 3.10+, [Ollama](https://ollama.com), and a GPU with about 10GB VRAM free.

```bash
# One-time setup
ollama pull deepseek-r1:14b     # ~9GB, legal reasoning
ollama pull qwen2.5vl:7b        # ~5GB, vision/OCR

pip install -r requirements.txt

# Run
python app.py                   # opens http://localhost:8000
```

Ollama loads one model at a time, so peak VRAM stays under 10GB. Built and tested on an RTX 5060 Ti 16GB.

---

## API

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | Serves the frontend |
| `POST` | `/api/extract` | Extracts text from uploaded files |
| `POST` | `/api/evaluate` | Agent 1, case evaluation |
| `POST` | `/api/generate` | Agent 2, drafts notice, generates PDFs |
| `GET` | `/api/download/{filename}` | Returns the generated PDF |
| `GET` | `/api/health` | Health check |

---

## Stack

Python, FastAPI, Uvicorn, Ollama, DeepSeek R1:14b, Qwen2.5-VL:7b, PyMuPDF, FPDF2, Pillow, vanilla HTML/CSS/JS.

Everything lives in a single `app.py`: backend, AI pipeline, and frontend. One file, one command.

---

## Roadmap

- SQLite case history with 15-day response deadline tracking
- Pre-filled eDaakhil complaint form generator for Consumer Court filing
- Registered office address lookup for major Indian companies
- Support for more statutes: IT Act 2000, RERA, Banking Ombudsman Scheme

---

Built by [Arnav Kamble](https://github.com/arnavkamble514).
