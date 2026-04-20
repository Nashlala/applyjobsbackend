"""
ApplyOS Backend — FastAPI + python-jobspy + Claude AI + SQLite
Run with: python backend.py
API runs on http://localhost:8000
"""

import os, json, sqlite3, time, threading, uuid, io
from datetime import datetime
from typing import Optional, List

import anthropic
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# PDF + DOCX extraction
try:
    import pdfplumber
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

try:
    from docx import Document as DocxDocument
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

# PDF generation
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib import colors
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

try:
    import jobspy
    HAS_JOBSPY = True
except ImportError:
    HAS_JOBSPY = False

# ─── Config ───────────────────────────────────────────────────────────────────
DB_PATH       = "applyjobs.db"
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

app = FastAPI(title="ApplyOS API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Database ─────────────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS jobs (
                id            TEXT PRIMARY KEY,
                title         TEXT,
                company       TEXT,
                location      TEXT,
                description   TEXT,
                url           TEXT,
                source        TEXT DEFAULT 'manual',
                status        TEXT DEFAULT 'NEW',
                fit_score     INTEGER,
                verdict       TEXT,
                analysis      TEXT,
                salary        TEXT,
                job_type      TEXT,
                date_posted   TEXT,
                added_at      REAL,
                analyzed_at   REAL,
                tailored_resume TEXT
            );

            CREATE TABLE IF NOT EXISTS resume (
                id          INTEGER PRIMARY KEY CHECK (id = 1),
                text        TEXT,
                filename    TEXT,
                file_bytes  BLOB,
                file_type   TEXT,
                updated_at  REAL
            );

            CREATE TABLE IF NOT EXISTS scan_config (
                id                   INTEGER PRIMARY KEY CHECK (id = 1),
                search_term          TEXT DEFAULT 'Software Engineer',
                location             TEXT DEFAULT 'United States',
                remote_only          INTEGER DEFAULT 0,
                sites                TEXT DEFAULT 'linkedin,indeed,glassdoor',
                results_per_site     INTEGER DEFAULT 20,
                last_scan            REAL,
                auto_scan            INTEGER DEFAULT 0,
                scan_interval_mins   INTEGER DEFAULT 60
            );

            INSERT OR IGNORE INTO scan_config(id) VALUES(1);
            INSERT OR IGNORE INTO resume(id, text, updated_at) VALUES(1, '', 0);
        """)
        # Add tailored_resume column if upgrading from old DB
        try:
            conn.execute("ALTER TABLE jobs ADD COLUMN tailored_resume TEXT")
        except:
            pass
        # Add file columns if upgrading
        for col in ["filename TEXT", "file_bytes BLOB", "file_type TEXT"]:
            try:
                conn.execute(f"ALTER TABLE resume ADD COLUMN {col}")
            except:
                pass

init_db()

# ─── File text extraction ──────────────────────────────────────────────────────
def extract_pdf_text(file_bytes: bytes) -> str:
    if not HAS_PDF:
        return ""
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        return text.strip()
    except Exception as e:
        return f"[PDF extraction failed: {e}]"

def extract_docx_text(file_bytes: bytes) -> str:
    if not HAS_DOCX:
        return ""
    try:
        doc = DocxDocument(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as e:
        return f"[DOCX extraction failed: {e}]"

# ─── PDF generation ───────────────────────────────────────────────────────────
def generate_resume_pdf(resume_text: str, job_title: str = "", company: str = "") -> bytes:
    if not HAS_REPORTLAB:
        # Fallback: return plain text as bytes
        return resume_text.encode("utf-8")

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch,
    )

    styles = getSampleStyleSheet()
    normal = ParagraphStyle(
        "NormalCustom",
        parent=styles["Normal"],
        fontSize=10.5,
        leading=15,
        spaceAfter=4,
    )
    heading = ParagraphStyle(
        "HeadingCustom",
        parent=styles["Heading2"],
        fontSize=12,
        spaceBefore=12,
        spaceAfter=4,
        textColor=colors.HexColor("#1a1a2e"),
    )

    story = []

    if job_title or company:
        story.append(Paragraph(
            f"<font color='#888888' size='9'>Tailored for: {job_title} @ {company}</font>",
            normal
        ))
        story.append(Spacer(1, 8))

    for line in resume_text.split("\n"):
        line = line.strip()
        if not line:
            story.append(Spacer(1, 4))
            continue
        # Detect section headers (all caps or ends with colon)
        if line.isupper() or (line.endswith(":") and len(line) < 40):
            story.append(Paragraph(line, heading))
        else:
            # Escape XML special chars
            safe = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            story.append(Paragraph(safe, normal))

    doc.build(story)
    return buffer.getvalue()

# ─── AI Analysis ──────────────────────────────────────────────────────────────
def analyze_with_claude(resume_text: str, job_desc: str) -> dict:
    if not ANTHROPIC_KEY:
        return {
            "fitScore": 0, "verdict": "SKIP",
            "jobTitle": "Unknown", "company": "Unknown", "location": "Unknown",
            "salary": None, "jobType": "Full-time", "topSkills": [],
            "scoreExplanation": "Set ANTHROPIC_API_KEY to enable AI analysis.",
            "strengths": [], "weaknesses": [], "resumeBullets": [],
            "keywords": [], "missingSignals": [], "gapClosers": [],
            "networkingTargets": [], "outreachTemplate": ""
        }

    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    prompt = f"""You are a senior career coach. Analyze fit between this resume and job description.

RESUME:
{resume_text or "(No resume provided)"}

JOB DESCRIPTION:
{job_desc[:3000]}

Respond ONLY with valid JSON, no markdown:
{{
  "fitScore": <0-100>,
  "verdict": <"APPLY" or "SKIP">,
  "jobTitle": "<job title>",
  "company": "<company or Unknown>",
  "location": "<location or Remote>",
  "salary": "<salary range or null>",
  "jobType": "<Full-time/Part-time/Contract/Internship>",
  "topSkills": ["<s1>","<s2>","<s3>"],
  "scoreExplanation": "<2-3 sentence explanation>",
  "strengths": ["<s1>","<s2>","<s3>"],
  "weaknesses": ["<w1>","<w2>"],
  "resumeBullets": ["<bullet1>","<bullet2>","<bullet3>","<bullet4>"],
  "keywords": ["<k1>","<k2>","<k3>","<k4>","<k5>"],
  "missingSignals": ["<gap1>","<gap2>","<gap3>"],
  "gapClosers": ["<fix1>","<fix2>","<fix3>"],
  "networkingTargets": ["<who1>","<who2>","<who3>"],
  "outreachTemplate": "<3-4 line LinkedIn message>"
}}"""

    msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    text = msg.content[0].text.strip()
    return json.loads(text.replace("```json","").replace("```","").strip())


def tailor_resume_with_claude(resume_text: str, job_desc: str, analysis: dict) -> str:
    """Rewrite the resume to be optimally tailored for a specific job."""
    if not ANTHROPIC_KEY:
        return resume_text

    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

    bullets = "\n".join(f"- {b}" for b in analysis.get("resumeBullets", []))
    keywords = ", ".join(analysis.get("keywords", []))
    gaps = "\n".join(f"- {g}" for g in analysis.get("gapClosers", []))

    prompt = f"""You are an expert resume writer. Rewrite the candidate's resume to be optimally tailored for this specific job.

ORIGINAL RESUME:
{resume_text}

JOB DESCRIPTION:
{job_desc[:2500]}

AI ANALYSIS INSIGHTS:
Suggested bullets to incorporate:
{bullets}

Keywords to include: {keywords}

Gaps to address:
{gaps}

INSTRUCTIONS:
1. Keep ALL real experience, education, and skills — never fabricate anything
2. Reorder and rewrite bullet points to emphasize what this job values most
3. Naturally weave in the keywords listed above
4. Incorporate the suggested bullets where they fit authentically
5. Use strong action verbs and quantify achievements where possible
6. Keep the same general structure (contact info, experience, education, skills)
7. Do NOT add fake experience or credentials
8. Output ONLY the rewritten resume text — no commentary, no markdown headers like ```

Output the complete tailored resume as plain text, ready to copy or save as PDF."""

    msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )
    return msg.content[0].text.strip()


# ─── Job Scanning ─────────────────────────────────────────────────────────────
def run_scan(search_term, location, remote_only, sites, results_per_site):
    if not HAS_JOBSPY:
        return {"scraped": 0, "new": 0, "error": "python-jobspy not installed"}
    try:
        df = jobspy.scrape_jobs(
            site_name=sites,
            search_term=search_term,
            location=location,
            results_wanted=results_per_site,
            hours_old=72,
            country_indeed="USA",
            is_remote=remote_only,
            linkedin_fetch_description=True,
        )
    except Exception as e:
        return {"scraped": 0, "new": 0, "error": str(e)}

    resume_text = ""
    with get_db() as conn:
        row = conn.execute("SELECT text FROM resume WHERE id=1").fetchone()
        if row:
            resume_text = row["text"] or ""

    new_count = 0
    for _, row in df.iterrows():
        url     = str(row.get("job_url") or "")
        title   = str(row.get("title") or "")
        company = str(row.get("company") or "")
        desc    = str(row.get("description") or "")
        loc     = str(row.get("location") or "")
        salary  = str(row.get("min_amount") or "") or None
        source  = str(row.get("site") or "jobspy")
        date_p  = str(row.get("date_posted") or "")

        if not title or not desc or len(desc) < 100:
            continue

        with get_db() as conn:
            exists = conn.execute(
                "SELECT id FROM jobs WHERE url=? OR (title=? AND company=?)",
                (url, title, company)
            ).fetchone()
            if exists:
                continue

        job_id = str(uuid.uuid4())
        analysis = None
        fit_score = None
        verdict = None

        try:
            analysis  = analyze_with_claude(resume_text, desc)
            fit_score = analysis.get("fitScore")
            verdict   = analysis.get("verdict")
            analysis["jobTitle"] = title or analysis.get("jobTitle")
            analysis["company"]  = company or analysis.get("company")
            analysis["location"] = loc or analysis.get("location")
        except Exception:
            pass

        initial_status = "SKIPPED" if verdict == "SKIP" else "NEW"

        with get_db() as conn:
            conn.execute("""
                INSERT INTO jobs
                (id,title,company,location,description,url,source,status,
                 fit_score,verdict,analysis,salary,job_type,date_posted,added_at,analyzed_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                job_id, title, company, loc, desc, url, source,
                initial_status, fit_score, verdict,
                json.dumps(analysis) if analysis else None,
                salary, None, date_p, time.time(), time.time()
            ))
        new_count += 1

    with get_db() as conn:
        conn.execute("UPDATE scan_config SET last_scan=? WHERE id=1", (time.time(),))

    return {"scraped": len(df), "new": new_count}


# ─── Auto-scan loop ───────────────────────────────────────────────────────────
def auto_scan_loop():
    while True:
        time.sleep(60)
        with get_db() as conn:
            cfg = conn.execute("SELECT * FROM scan_config WHERE id=1").fetchone()
        if not cfg or not cfg["auto_scan"]:
            continue
        interval = (cfg["scan_interval_mins"] or 60) * 60
        last = cfg["last_scan"] or 0
        if time.time() - last >= interval:
            sites = (cfg["sites"] or "linkedin,indeed").split(",")
            run_scan(cfg["search_term"], cfg["location"],
                     bool(cfg["remote_only"]), sites, cfg["results_per_site"])

threading.Thread(target=auto_scan_loop, daemon=True).start()


# ─── Pydantic models ──────────────────────────────────────────────────────────
class ResumeUpdate(BaseModel):
    text: str

class StatusUpdate(BaseModel):
    status: str

class ManualJob(BaseModel):
    description: str
    url: Optional[str] = ""

class ScanRequest(BaseModel):
    search_term: str = "Software Engineer"
    location: str = "United States"
    remote_only: bool = False
    sites: List[str] = ["linkedin", "indeed"]
    results_per_site: int = 15

class ScanConfigUpdate(BaseModel):
    search_term: Optional[str] = None
    location: Optional[str] = None
    remote_only: Optional[bool] = None
    sites: Optional[List[str]] = None
    results_per_site: Optional[int] = None
    auto_scan: Optional[bool] = None
    scan_interval_mins: Optional[int] = None

STATUS_LABELS = {"NEW","REVIEWING","APPLIED","INTERVIEW","OFFER","REJECTED","SKIPPED"}

# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status": "ApplyOS API v2 running",
        "docs": "/docs",
        "features": ["job-scanning", "ai-scoring", "resume-upload", "resume-tailoring"]
    }


# ── Resume ────────────────────────────────────────────────────────────────────
@app.get("/resume")
def get_resume():
    with get_db() as conn:
        row = conn.execute("SELECT text, filename, file_type, updated_at FROM resume WHERE id=1").fetchone()
    return {
        "text": row["text"] or "",
        "filename": row["filename"],
        "file_type": row["file_type"],
        "updated_at": row["updated_at"]
    }

@app.post("/resume")
def save_resume_text(body: ResumeUpdate):
    with get_db() as conn:
        conn.execute("UPDATE resume SET text=?, updated_at=? WHERE id=1",
                     (body.text, time.time()))
    return {"ok": True}

@app.post("/resume/upload")
async def upload_resume(file: UploadFile = File(...)):
    """Upload a PDF or DOCX resume file. Text is extracted automatically."""
    content = await file.read()
    filename = file.filename or "resume"
    file_type = ""
    extracted = ""

    if filename.lower().endswith(".pdf"):
        file_type = "pdf"
        extracted = extract_pdf_text(content)
    elif filename.lower().endswith(".docx"):
        file_type = "docx"
        extracted = extract_docx_text(content)
    elif filename.lower().endswith(".txt"):
        file_type = "txt"
        extracted = content.decode("utf-8", errors="ignore")
    else:
        raise HTTPException(400, "Unsupported file type. Upload PDF, DOCX, or TXT.")

    if not extracted or len(extracted) < 50:
        raise HTTPException(400, "Could not extract text from file. Try copying the text manually.")

    with get_db() as conn:
        conn.execute(
            "UPDATE resume SET text=?, filename=?, file_bytes=?, file_type=?, updated_at=? WHERE id=1",
            (extracted, filename, content, file_type, time.time())
        )

    return {
        "ok": True,
        "filename": filename,
        "file_type": file_type,
        "chars_extracted": len(extracted),
        "preview": extracted[:300] + "..." if len(extracted) > 300 else extracted
    }

@app.get("/resume/download")
def download_resume():
    """Download the stored resume file."""
    with get_db() as conn:
        row = conn.execute("SELECT file_bytes, filename, file_type FROM resume WHERE id=1").fetchone()

    if not row or not row["file_bytes"]:
        raise HTTPException(404, "No resume file uploaded yet.")

    ext = row["file_type"] or "pdf"
    media = "application/pdf" if ext == "pdf" else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    return StreamingResponse(
        io.BytesIO(row["file_bytes"]),
        media_type=media,
        headers={"Content-Disposition": f'attachment; filename="{row["filename"]}"'}
    )


# ── Jobs ──────────────────────────────────────────────────────────────────────
@app.get("/jobs")
def get_jobs(status: Optional[str] = None, sort: str = "newest", search: Optional[str] = None):
    q = "SELECT * FROM jobs WHERE 1=1"
    params = []
    if status and status != "ALL":
        q += " AND status=?"; params.append(status)
    if search:
        q += " AND (title LIKE ? OR company LIKE ? OR description LIKE ?)"
        params += [f"%{search}%", f"%{search}%", f"%{search}%"]
    q += " ORDER BY fit_score DESC NULLS LAST" if sort == "score" else " ORDER BY added_at DESC"

    with get_db() as conn:
        rows = conn.execute(q, params).fetchall()

    jobs = []
    for r in rows:
        j = dict(r)
        j["analysis"] = json.loads(j["analysis"]) if j["analysis"] else None
        j.pop("tailored_resume", None)  # don't send large text in list
        jobs.append(j)
    return jobs


@app.get("/jobs/stats")
def get_stats():
    with get_db() as conn:
        total      = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        by_status  = conn.execute("SELECT status, COUNT(*) as cnt FROM jobs GROUP BY status").fetchall()
        by_verdict = conn.execute("SELECT verdict, COUNT(*) as cnt FROM jobs WHERE verdict IS NOT NULL GROUP BY verdict").fetchall()
        avg_score  = conn.execute("SELECT AVG(fit_score) FROM jobs WHERE fit_score IS NOT NULL").fetchone()[0]
        cfg        = conn.execute("SELECT * FROM scan_config WHERE id=1").fetchone()
    return {
        "total": total,
        "avg_score": round(avg_score or 0),
        "by_status":  {r["status"]:  r["cnt"] for r in by_status},
        "by_verdict": {r["verdict"]: r["cnt"] for r in by_verdict},
        "last_scan": cfg["last_scan"] if cfg else None,
        "auto_scan": bool(cfg["auto_scan"]) if cfg else False,
    }


@app.post("/jobs/manual")
def add_manual_job(body: ManualJob, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())

    with sqlite3.connect(DB_PATH) as conn:
        resume_row = conn.execute("SELECT text FROM resume WHERE id=1").fetchone()
        resume_text = resume_row[0] or "" if resume_row else ""
        conn.execute(
            "INSERT INTO jobs (id, description, url, source, status, added_at) VALUES (?,?,?,'manual','NEW',?)",
            (job_id, body.description, body.url or "", time.time())
        )

    def analyze_async():
        try:
            a = analyze_with_claude(resume_text, body.description)
            status = "SKIPPED" if a.get("verdict") == "SKIP" else "NEW"
            with sqlite3.connect(DB_PATH) as c:
                c.execute("""
                    UPDATE jobs SET title=?,company=?,location=?,salary=?,
                    fit_score=?,verdict=?,analysis=?,status=?,analyzed_at=? WHERE id=?
                """, (a.get("jobTitle"), a.get("company"), a.get("location"),
                      a.get("salary"), a.get("fitScore"), a.get("verdict"),
                      json.dumps(a), status, time.time(), job_id))
        except Exception:
            with sqlite3.connect(DB_PATH) as c:
                c.execute("UPDATE jobs SET title='Analysis failed' WHERE id=?", (job_id,))

    background_tasks.add_task(analyze_async)
    return {"id": job_id, "status": "analyzing"}


@app.patch("/jobs/{job_id}/status")
def update_status(job_id: str, body: StatusUpdate):
    if body.status not in STATUS_LABELS:
        raise HTTPException(400, f"Invalid status")
    with get_db() as conn:
        conn.execute("UPDATE jobs SET status=? WHERE id=?", (body.status, job_id))
    return {"ok": True}


@app.delete("/jobs/{job_id}")
def delete_job(job_id: str):
    with get_db() as conn:
        conn.execute("DELETE FROM jobs WHERE id=?", (job_id,))
    return {"ok": True}


@app.delete("/jobs")
def clear_all_jobs():
    with get_db() as conn:
        conn.execute("DELETE FROM jobs")
    return {"ok": True}


# ── Resume Tailoring ──────────────────────────────────────────────────────────
@app.post("/jobs/{job_id}/tailor")
def tailor_resume_for_job(job_id: str, background_tasks: BackgroundTasks):
    """
    Generate a tailored version of the resume for a specific job.
    Runs async — poll GET /jobs/{job_id}/tailored to check when ready.
    """
    with get_db() as conn:
        job = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        resume_row = conn.execute("SELECT text FROM resume WHERE id=1").fetchone()

    if not job:
        raise HTTPException(404, "Job not found")

    resume_text = resume_row[0] if resume_row else ""
    if not resume_text or len(resume_text) < 50:
        raise HTTPException(400, "No resume found. Upload or paste your resume first.")

    analysis = json.loads(job["analysis"]) if job["analysis"] else {}

    def tailor_async():
        try:
            tailored = tailor_resume_with_claude(resume_text, job["description"] or "", analysis)
            with sqlite3.connect(DB_PATH) as c:
                c.execute("UPDATE jobs SET tailored_resume=? WHERE id=?", (tailored, job_id))
        except Exception as e:
            with sqlite3.connect(DB_PATH) as c:
                c.execute("UPDATE jobs SET tailored_resume=? WHERE id=?",
                          (f"[Tailoring failed: {e}]", job_id))

    background_tasks.add_task(tailor_async)
    return {"status": "tailoring_started", "job_id": job_id}


@app.get("/jobs/{job_id}/tailored")
def get_tailored_resume(job_id: str):
    """Check if tailoring is done and return the tailored text."""
    with get_db() as conn:
        row = conn.execute("SELECT tailored_resume, title, company FROM jobs WHERE id=?", (job_id,)).fetchone()

    if not row:
        raise HTTPException(404, "Job not found")

    return {
        "job_id": job_id,
        "title": row["title"],
        "company": row["company"],
        "tailored_resume": row["tailored_resume"],
        "ready": bool(row["tailored_resume"])
    }


@app.get("/jobs/{job_id}/tailored/pdf")
def download_tailored_pdf(job_id: str):
    """Download the tailored resume as a PDF."""
    with get_db() as conn:
        row = conn.execute("SELECT tailored_resume, title, company FROM jobs WHERE id=?", (job_id,)).fetchone()

    if not row or not row["tailored_resume"]:
        raise HTTPException(404, "Tailored resume not ready yet. Call POST /jobs/{id}/tailor first.")

    pdf_bytes = generate_resume_pdf(row["tailored_resume"], row["title"] or "", row["company"] or "")
    filename = f"resume_tailored_{(row['company'] or 'job').replace(' ','_')}.pdf"

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


# ── Scanning ──────────────────────────────────────────────────────────────────
@app.post("/scan")
def trigger_scan(body: ScanRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(
        run_scan, body.search_term, body.location,
        body.remote_only, body.sites, body.results_per_site
    )
    return {"status": "scan started", "config": body.dict()}

@app.get("/scan/config")
def get_scan_config():
    with get_db() as conn:
        row = conn.execute("SELECT * FROM scan_config WHERE id=1").fetchone()
    cfg = dict(row)
    cfg["sites"] = cfg["sites"].split(",") if cfg["sites"] else []
    return cfg

@app.post("/scan/config")
def update_scan_config(body: ScanConfigUpdate):
    updates = {k: v for k, v in body.dict().items() if v is not None}
    if "sites" in updates:
        updates["sites"] = ",".join(updates["sites"])
    if not updates:
        return {"ok": True}
    set_clause = ", ".join(f"{k}=?" for k in updates)
    with get_db() as conn:
        conn.execute(f"UPDATE scan_config SET {set_clause} WHERE id=1", list(updates.values()))
    return {"ok": True}


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("\n╔══════════════════════════════════════╗")
    print("║   ApplyOS Backend v2 starting...      ║")
    print("║   API:  http://localhost:8000          ║")
    print("║   Docs: http://localhost:8000/docs     ║")
    print("╚══════════════════════════════════════╝\n")
    if not ANTHROPIC_KEY:
        print("⚠️  ANTHROPIC_API_KEY not set — AI analysis disabled")
    if not HAS_PDF:
        print("⚠️  pdfplumber not installed — PDF upload disabled")
    if not HAS_DOCX:
        print("⚠️  python-docx not installed — DOCX upload disabled")
    if not HAS_REPORTLAB:
        print("⚠️  reportlab not installed — PDF download will be plain text")
    print()
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
