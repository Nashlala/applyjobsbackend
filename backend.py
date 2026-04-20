"""
ApplyOS Backend — FastAPI + python-jobspy + Claude AI + SQLite
Run with: python backend.py
API runs on http://localhost:8000
"""

import os, json, sqlite3, time, threading, uuid
from datetime import datetime
from typing import Optional, List

import anthropic
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import jobspy

# ─── Config ───────────────────────────────────────────────────────────────────
DB_PATH         = "applyjobs.db"
ANTHROPIC_KEY   = os.environ.get("ANTHROPIC_API_KEY", "")
SCAN_INTERVAL   = 60 * 60  # auto-scan every 60 min (0 to disable)

app = FastAPI(title="ApplyOS API", version="1.0.0")

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
                id          TEXT PRIMARY KEY,
                title       TEXT,
                company     TEXT,
                location    TEXT,
                description TEXT,
                url         TEXT,
                source      TEXT DEFAULT 'manual',
                status      TEXT DEFAULT 'NEW',
                fit_score   INTEGER,
                verdict     TEXT,
                analysis    TEXT,
                salary      TEXT,
                job_type    TEXT,
                date_posted TEXT,
                added_at    REAL,
                analyzed_at REAL
            );

            CREATE TABLE IF NOT EXISTS resume (
                id   INTEGER PRIMARY KEY CHECK (id = 1),
                text TEXT,
                updated_at REAL
            );

            CREATE TABLE IF NOT EXISTS scan_config (
                id          INTEGER PRIMARY KEY CHECK (id = 1),
                search_term TEXT DEFAULT 'Software Engineer',
                location    TEXT DEFAULT 'United States',
                remote_only INTEGER DEFAULT 0,
                sites       TEXT DEFAULT 'linkedin,indeed,glassdoor',
                results_per_site INTEGER DEFAULT 20,
                last_scan   REAL,
                auto_scan   INTEGER DEFAULT 0,
                scan_interval_mins INTEGER DEFAULT 60
            );

            INSERT OR IGNORE INTO scan_config(id) VALUES(1);
            INSERT OR IGNORE INTO resume(id, text, updated_at) VALUES(1, '', 0);
        """)

init_db()

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

Respond ONLY with valid JSON, no markdown, no backticks:
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


# ─── Job Scanning ─────────────────────────────────────────────────────────────
def run_scan(search_term: str, location: str, remote_only: bool,
             sites: List[str], results_per_site: int) -> dict:
    """Scrape jobs from multiple boards using python-jobspy."""
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
            analysis = analyze_with_claude(resume_text, desc)
            fit_score = analysis.get("fitScore")
            verdict   = analysis.get("verdict")
            # Override parsed fields with scraped ones (more reliable)
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
            run_scan(
                cfg["search_term"], cfg["location"],
                bool(cfg["remote_only"]), sites, cfg["results_per_site"]
            )

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


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ApplyOS API running", "docs": "/docs"}


# ── Resume ────────────────────────────────────────────────────────────────────
@app.get("/resume")
def get_resume():
    with get_db() as conn:
        row = conn.execute("SELECT text, updated_at FROM resume WHERE id=1").fetchone()
    return {"text": row["text"] or "", "updated_at": row["updated_at"]}

@app.post("/resume")
def save_resume(body: ResumeUpdate):
    with get_db() as conn:
        conn.execute("UPDATE resume SET text=?, updated_at=? WHERE id=1",
                     (body.text, time.time()))
    return {"ok": True}


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
    if sort == "score":
        q += " ORDER BY fit_score DESC NULLS LAST"
    else:
        q += " ORDER BY added_at DESC"

    with get_db() as conn:
        rows = conn.execute(q, params).fetchall()

    jobs = []
    for r in rows:
        j = dict(r)
        j["analysis"] = json.loads(j["analysis"]) if j["analysis"] else None
        jobs.append(j)
    return jobs


@app.get("/jobs/stats")
def get_stats():
    with get_db() as conn:
        total     = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        by_status = conn.execute("SELECT status, COUNT(*) as cnt FROM jobs GROUP BY status").fetchall()
        by_verdict= conn.execute("SELECT verdict, COUNT(*) as cnt FROM jobs WHERE verdict IS NOT NULL GROUP BY verdict").fetchall()
        avg_score = conn.execute("SELECT AVG(fit_score) FROM jobs WHERE fit_score IS NOT NULL").fetchone()[0]
        cfg       = conn.execute("SELECT * FROM scan_config WHERE id=1").fetchone()

    return {
        "total": total,
        "avg_score": round(avg_score or 0),
        "by_status": {r["status"]: r["cnt"] for r in by_status},
        "by_verdict": {r["verdict"]: r["cnt"] for r in by_verdict},
        "last_scan": cfg["last_scan"] if cfg else None,
        "auto_scan": bool(cfg["auto_scan"]) if cfg else False,
    }


@app.post("/jobs/manual")
def add_manual_job(body: ManualJob, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())

    with get_db() as conn:
        resume_row = conn.execute("SELECT text FROM resume WHERE id=1").fetchone()
        resume_text = resume_row["text"] or "" if resume_row else ""

    conn2 = sqlite3.connect(DB_PATH)
    conn2.execute("""
        INSERT INTO jobs (id, description, url, source, status, added_at)
        VALUES (?, ?, ?, 'manual', 'NEW', ?)
    """, (job_id, body.description, body.url or "", time.time()))
    conn2.commit()
    conn2.close()

    def analyze_async():
        try:
            a = analyze_with_claude(resume_text, body.description)
            status = "SKIPPED" if a.get("verdict") == "SKIP" else "NEW"
            c = sqlite3.connect(DB_PATH)
            c.execute("""
                UPDATE jobs SET title=?,company=?,location=?,salary=?,
                fit_score=?,verdict=?,analysis=?,status=?,analyzed_at=? WHERE id=?
            """, (
                a.get("jobTitle"), a.get("company"), a.get("location"),
                a.get("salary"), a.get("fitScore"), a.get("verdict"),
                json.dumps(a), status, time.time(), job_id
            ))
            c.commit(); c.close()
        except Exception as e:
            c = sqlite3.connect(DB_PATH)
            c.execute("UPDATE jobs SET title='Analysis failed' WHERE id=?", (job_id,))
            c.commit(); c.close()

    background_tasks.add_task(analyze_async)
    return {"id": job_id, "status": "analyzing"}


@app.patch("/jobs/{job_id}/status")
def update_status(job_id: str, body: StatusUpdate):
    valid = list(STATUS_LABELS)
    if body.status not in valid:
        raise HTTPException(400, f"Status must be one of {valid}")
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


STATUS_LABELS = {"NEW","REVIEWING","APPLIED","INTERVIEW","OFFER","REJECTED","SKIPPED"}


# ── Scanning ──────────────────────────────────────────────────────────────────
@app.post("/scan")
def trigger_scan(body: ScanRequest, background_tasks: BackgroundTasks):
    def do_scan():
        run_scan(body.search_term, body.location, body.remote_only,
                 body.sites, body.results_per_site)

    background_tasks.add_task(do_scan)
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
        conn.execute(f"UPDATE scan_config SET {set_clause} WHERE id=1",
                     list(updates.values()))
    return {"ok": True}


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("\n╔══════════════════════════════════════╗")
    print("║   ApplyOS Backend starting...        ║")
    print("║   API:  http://localhost:8000         ║")
    print("║   Docs: http://localhost:8000/docs    ║")
    print("╚══════════════════════════════════════╝\n")
    if not ANTHROPIC_KEY:
        print("⚠️  ANTHROPIC_API_KEY not set — AI analysis disabled")
        print("   Set it with: export ANTHROPIC_API_KEY=sk-ant-...\n")
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
