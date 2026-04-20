"""
ApplyOS Backend v3 — FastAPI + PostgreSQL + python-jobspy + Claude AI
Run with: python backend.py
"""

import os, json, time, threading, uuid, io
from typing import Optional, List

import anthropic
import psycopg2
import psycopg2.extras
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

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
DATABASE_URL  = os.environ.get("DATABASE_URL", "")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

app = FastAPI(title="ApplyOS API", version="3.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Database ─────────────────────────────────────────────────────────────────
def get_db():
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = False
    return conn

def init_db():
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
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
                    added_at      DOUBLE PRECISION,
                    analyzed_at   DOUBLE PRECISION,
                    tailored_resume TEXT
                );

                CREATE TABLE IF NOT EXISTS resume (
                    id          INTEGER PRIMARY KEY,
                    text        TEXT,
                    filename    TEXT,
                    file_bytes  BYTEA,
                    file_type   TEXT,
                    updated_at  DOUBLE PRECISION
                );

                CREATE TABLE IF NOT EXISTS scan_config (
                    id                   INTEGER PRIMARY KEY,
                    search_term          TEXT DEFAULT 'Software Engineer',
                    location             TEXT DEFAULT 'United States',
                    remote_only          BOOLEAN DEFAULT FALSE,
                    sites                TEXT DEFAULT 'linkedin,indeed,glassdoor',
                    results_per_site     INTEGER DEFAULT 20,
                    last_scan            DOUBLE PRECISION,
                    auto_scan            BOOLEAN DEFAULT FALSE,
                    scan_interval_mins   INTEGER DEFAULT 60
                );

                INSERT INTO scan_config(id) VALUES(1) ON CONFLICT DO NOTHING;
                INSERT INTO resume(id, text, updated_at) VALUES(1, '', 0) ON CONFLICT DO NOTHING;
            """)
        conn.commit()

init_db()

# ─── DB helpers ───────────────────────────────────────────────────────────────
def db_fetchone(query, params=()):
    conn = get_db()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, params)
            return cur.fetchone()
    finally:
        conn.close()

def db_fetchall(query, params=()):
    conn = get_db()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, params)
            return cur.fetchall()
    finally:
        conn.close()

def db_execute(query, params=()):
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute(query, params)
        conn.commit()
    finally:
        conn.close()

def db_executemany(query, params_list):
    conn = get_db()
    try:
        with conn.cursor() as cur:
            for params in params_list:
                cur.execute(query, params)
        conn.commit()
    finally:
        conn.close()

# ─── File extraction ──────────────────────────────────────────────────────────
def extract_pdf_text(file_bytes):
    if not HAS_PDF:
        return ""
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            return "\n".join(p.extract_text() or "" for p in pdf.pages).strip()
    except Exception as e:
        return f"[PDF extraction failed: {e}]"

def extract_docx_text(file_bytes):
    if not HAS_DOCX:
        return ""
    try:
        doc = DocxDocument(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as e:
        return f"[DOCX extraction failed: {e}]"

# ─── PDF generation ───────────────────────────────────────────────────────────
def generate_resume_pdf(resume_text, job_title="", company=""):
    if not HAS_REPORTLAB:
        return resume_text.encode("utf-8")
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            rightMargin=0.75*inch, leftMargin=0.75*inch,
                            topMargin=0.75*inch, bottomMargin=0.75*inch)
    styles = getSampleStyleSheet()
    normal = ParagraphStyle("N", parent=styles["Normal"], fontSize=10.5, leading=15, spaceAfter=4)
    heading = ParagraphStyle("H", parent=styles["Heading2"], fontSize=12, spaceBefore=12, spaceAfter=4, textColor=colors.HexColor("#1a1a2e"))
    story = []
    if job_title or company:
        story.append(Paragraph(f"<font color='#888888' size='9'>Tailored for: {job_title} @ {company}</font>", normal))
        story.append(Spacer(1, 8))
    for line in resume_text.split("\n"):
        line = line.strip()
        if not line:
            story.append(Spacer(1, 4))
        elif line.isupper() or (line.endswith(":") and len(line) < 40):
            story.append(Paragraph(line, heading))
        else:
            safe = line.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
            story.append(Paragraph(safe, normal))
    doc.build(story)
    return buffer.getvalue()

# ─── AI ───────────────────────────────────────────────────────────────────────
def analyze_with_claude(resume_text, job_desc):
    if not ANTHROPIC_KEY:
        return {"fitScore":0,"verdict":"SKIP","jobTitle":"Unknown","company":"Unknown","location":"Unknown","salary":None,"jobType":"Full-time","topSkills":[],"scoreExplanation":"Set ANTHROPIC_API_KEY to enable AI analysis.","strengths":[],"weaknesses":[],"resumeBullets":[],"keywords":[],"missingSignals":[],"gapClosers":[],"networkingTargets":[],"outreachTemplate":""}

    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    prompt = f"""You are a senior career coach. Analyze fit between this resume and job description.

RESUME:
{resume_text or "(No resume provided)"}

JOB DESCRIPTION:
{job_desc[:3000]}

Respond ONLY with valid JSON, no markdown:
{{"fitScore":<0-100>,"verdict":"APPLY or SKIP","jobTitle":"<title>","company":"<company>","location":"<location>","salary":"<range or null>","jobType":"<type>","topSkills":["s1","s2","s3"],"scoreExplanation":"<2-3 sentences>","strengths":["s1","s2","s3"],"weaknesses":["w1","w2"],"resumeBullets":["b1","b2","b3","b4"],"keywords":["k1","k2","k3","k4","k5"],"missingSignals":["g1","g2","g3"],"gapClosers":["f1","f2","f3"],"networkingTargets":["w1","w2","w3"],"outreachTemplate":"<message>"}}"""

    msg = client.messages.create(model="claude-sonnet-4-20250514", max_tokens=1000,
                                  messages=[{"role":"user","content":prompt}])
    return json.loads(msg.content[0].text.strip().replace("```json","").replace("```","").strip())

def tailor_resume_with_claude(resume_text, job_desc, analysis):
    if not ANTHROPIC_KEY:
        return resume_text
    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    bullets  = "\n".join(f"- {b}" for b in analysis.get("resumeBullets",[]))
    keywords = ", ".join(analysis.get("keywords",[]))
    gaps     = "\n".join(f"- {g}" for g in analysis.get("gapClosers",[]))
    prompt = f"""You are an expert resume writer. Rewrite this resume tailored for the specific job.

ORIGINAL RESUME:
{resume_text}

JOB DESCRIPTION:
{job_desc[:2500]}

INSIGHTS:
Suggested bullets:
{bullets}
Keywords: {keywords}
Gaps to address:
{gaps}

RULES:
- Never fabricate experience or credentials
- Reorder and rewrite bullets to emphasize what this job values
- Weave in keywords naturally
- Use strong action verbs and quantify achievements
- Keep same structure (contact, experience, education, skills)
- Output ONLY the resume text, no commentary or markdown"""

    msg = client.messages.create(model="claude-sonnet-4-20250514", max_tokens=2000,
                                  messages=[{"role":"user","content":prompt}])
    return msg.content[0].text.strip()

# ─── Scanning ─────────────────────────────────────────────────────────────────
def run_scan(search_term, location, remote_only, sites, results_per_site):
    if not HAS_JOBSPY:
        return {"scraped":0,"new":0,"error":"python-jobspy not installed"}
    try:
        df = jobspy.scrape_jobs(site_name=sites, search_term=search_term, location=location,
                                results_wanted=results_per_site, hours_old=72, country_indeed="USA",
                                is_remote=remote_only, linkedin_fetch_description=True)
    except Exception as e:
        return {"scraped":0,"new":0,"error":str(e)}

    row = db_fetchone("SELECT text FROM resume WHERE id=1")
    resume_text = row["text"] or "" if row else ""

    new_count = 0
    for _, r in df.iterrows():
        url=str(r.get("job_url") or ""); title=str(r.get("title") or "")
        company=str(r.get("company") or ""); desc=str(r.get("description") or "")
        loc=str(r.get("location") or ""); salary=str(r.get("min_amount") or "") or None
        source=str(r.get("site") or "jobspy"); date_p=str(r.get("date_posted") or "")
        if not title or not desc or len(desc)<100:
            continue
        exists = db_fetchone("SELECT id FROM jobs WHERE url=%s OR (title=%s AND company=%s)",(url,title,company))
        if exists:
            continue
        job_id=str(uuid.uuid4()); analysis=None; fit_score=None; verdict=None
        try:
            analysis=analyze_with_claude(resume_text,desc)
            fit_score=analysis.get("fitScore"); verdict=analysis.get("verdict")
            analysis["jobTitle"]=title or analysis.get("jobTitle")
            analysis["company"]=company or analysis.get("company")
            analysis["location"]=loc or analysis.get("location")
        except:
            pass
        status="SKIPPED" if verdict=="SKIP" else "NEW"
        db_execute("""INSERT INTO jobs (id,title,company,location,description,url,source,status,
                   fit_score,verdict,analysis,salary,job_type,date_posted,added_at,analyzed_at)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                   (job_id,title,company,loc,desc,url,source,status,fit_score,verdict,
                    json.dumps(analysis) if analysis else None,salary,None,date_p,time.time(),time.time()))
        new_count+=1

    db_execute("UPDATE scan_config SET last_scan=%s WHERE id=1",(time.time(),))
    return {"scraped":len(df),"new":new_count}

def auto_scan_loop():
    while True:
        time.sleep(60)
        cfg = db_fetchone("SELECT * FROM scan_config WHERE id=1")
        if not cfg or not cfg["auto_scan"]:
            continue
        interval=(cfg["scan_interval_mins"] or 60)*60
        if time.time()-(cfg["last_scan"] or 0)>=interval:
            sites=(cfg["sites"] or "linkedin,indeed").split(",")
            run_scan(cfg["search_term"],cfg["location"],bool(cfg["remote_only"]),sites,cfg["results_per_site"])

threading.Thread(target=auto_scan_loop,daemon=True).start()

# ─── Models ───────────────────────────────────────────────────────────────────
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
    sites: List[str] = ["linkedin","indeed"]
    results_per_site: int = 15
class ScanConfigUpdate(BaseModel):
    search_term: Optional[str] = None
    location: Optional[str] = None
    remote_only: Optional[bool] = None
    sites: Optional[List[str]] = None
    results_per_site: Optional[int] = None
    auto_scan: Optional[bool] = None
    scan_interval_mins: Optional[int] = None

STATUS_LABELS={"NEW","REVIEWING","APPLIED","INTERVIEW","OFFER","REJECTED","SKIPPED"}

# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status":"ApplyOS API v3 running","db":"postgres","docs":"/docs"}

@app.get("/resume")
def get_resume():
    row=db_fetchone("SELECT text,filename,file_type,updated_at FROM resume WHERE id=1")
    return {"text":row["text"] or "","filename":row["filename"],"file_type":row["file_type"],"updated_at":row["updated_at"]}

@app.post("/resume")
def save_resume_text(body: ResumeUpdate):
    db_execute("UPDATE resume SET text=%s,updated_at=%s WHERE id=1",(body.text,time.time()))
    return {"ok":True}

@app.post("/resume/upload")
async def upload_resume(file: UploadFile = File(...)):
    content=await file.read()
    filename=file.filename or "resume"
    if filename.lower().endswith(".pdf"):
        file_type="pdf"; extracted=extract_pdf_text(content)
    elif filename.lower().endswith(".docx"):
        file_type="docx"; extracted=extract_docx_text(content)
    elif filename.lower().endswith(".txt"):
        file_type="txt"; extracted=content.decode("utf-8",errors="ignore")
    else:
        raise HTTPException(400,"Unsupported file type. Upload PDF, DOCX, or TXT.")
    if not extracted or len(extracted)<50:
        raise HTTPException(400,"Could not extract text from file. Try pasting manually.")
    db_execute("UPDATE resume SET text=%s,filename=%s,file_bytes=%s,file_type=%s,updated_at=%s WHERE id=1",
               (extracted,filename,psycopg2.Binary(content),file_type,time.time()))
    return {"ok":True,"filename":filename,"file_type":file_type,"chars_extracted":len(extracted),"preview":extracted[:300]+"..." if len(extracted)>300 else extracted}

@app.get("/resume/download")
def download_resume():
    row=db_fetchone("SELECT file_bytes,filename,file_type FROM resume WHERE id=1")
    if not row or not row["file_bytes"]:
        raise HTTPException(404,"No resume file uploaded yet.")
    ext=row["file_type"] or "pdf"
    media="application/pdf" if ext=="pdf" else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    return StreamingResponse(io.BytesIO(bytes(row["file_bytes"])),media_type=media,
                             headers={"Content-Disposition":f'attachment; filename="{row["filename"]}"'})

@app.get("/jobs")
def get_jobs(status: Optional[str]=None, sort: str="newest", search: Optional[str]=None):
    q="SELECT id,title,company,location,url,source,status,fit_score,verdict,analysis,salary,job_type,date_posted,added_at,analyzed_at FROM jobs WHERE 1=1"
    params=[]
    if status and status!="ALL":
        q+=" AND status=%s"; params.append(status)
    if search:
        q+=" AND (title ILIKE %s OR company ILIKE %s OR description ILIKE %s)"
        params+=[f"%{search}%",f"%{search}%",f"%{search}%"]
    q+=" ORDER BY fit_score DESC NULLS LAST" if sort=="score" else " ORDER BY added_at DESC"
    rows=db_fetchall(q,params)
    jobs=[]
    for r in rows:
        j=dict(r)
        j["analysis"]=json.loads(j["analysis"]) if j["analysis"] else None
        jobs.append(j)
    return jobs

@app.get("/jobs/stats")
def get_stats():
    total=db_fetchone("SELECT COUNT(*) as c FROM jobs")["c"]
    by_status={r["status"]:r["cnt"] for r in db_fetchall("SELECT status,COUNT(*) as cnt FROM jobs GROUP BY status")}
    by_verdict={r["verdict"]:r["cnt"] for r in db_fetchall("SELECT verdict,COUNT(*) as cnt FROM jobs WHERE verdict IS NOT NULL GROUP BY verdict")}
    avg=db_fetchone("SELECT AVG(fit_score) as a FROM jobs WHERE fit_score IS NOT NULL")["a"]
    cfg=db_fetchone("SELECT last_scan,auto_scan FROM scan_config WHERE id=1")
    return {"total":total,"avg_score":round(avg or 0),"by_status":by_status,"by_verdict":by_verdict,
            "last_scan":cfg["last_scan"] if cfg else None,"auto_scan":bool(cfg["auto_scan"]) if cfg else False}

@app.post("/jobs/manual")
def add_manual_job(body: ManualJob, background_tasks: BackgroundTasks):
    job_id=str(uuid.uuid4())
    row=db_fetchone("SELECT text FROM resume WHERE id=1")
    resume_text=row["text"] or "" if row else ""
    db_execute("INSERT INTO jobs (id,description,url,source,status,added_at) VALUES (%s,%s,%s,'manual','NEW',%s)",
               (job_id,body.description,body.url or "",time.time()))
    def analyze_async():
        try:
            a=analyze_with_claude(resume_text,body.description)
            status="SKIPPED" if a.get("verdict")=="SKIP" else "NEW"
            db_execute("UPDATE jobs SET title=%s,company=%s,location=%s,salary=%s,fit_score=%s,verdict=%s,analysis=%s,status=%s,analyzed_at=%s WHERE id=%s",
                       (a.get("jobTitle"),a.get("company"),a.get("location"),a.get("salary"),a.get("fitScore"),a.get("verdict"),json.dumps(a),status,time.time(),job_id))
        except:
            db_execute("UPDATE jobs SET title='Analysis failed' WHERE id=%s",(job_id,))
    background_tasks.add_task(analyze_async)
    return {"id":job_id,"status":"analyzing"}

@app.patch("/jobs/{job_id}/status")
def update_status(job_id: str, body: StatusUpdate):
    if body.status not in STATUS_LABELS:
        raise HTTPException(400,"Invalid status")
    db_execute("UPDATE jobs SET status=%s WHERE id=%s",(body.status,job_id))
    return {"ok":True}

@app.delete("/jobs/{job_id}")
def delete_job(job_id: str):
    db_execute("DELETE FROM jobs WHERE id=%s",(job_id,))
    return {"ok":True}

@app.delete("/jobs")
def clear_all_jobs():
    db_execute("DELETE FROM jobs")
    return {"ok":True}

@app.post("/jobs/{job_id}/tailor")
def tailor_resume_for_job(job_id: str, background_tasks: BackgroundTasks):
    job=db_fetchone("SELECT * FROM jobs WHERE id=%s",(job_id,))
    if not job:
        raise HTTPException(404,"Job not found")
    row=db_fetchone("SELECT text FROM resume WHERE id=1")
    resume_text=row["text"] or "" if row else ""
    if not resume_text or len(resume_text)<50:
        raise HTTPException(400,"No resume found. Upload or paste your resume first.")
    analysis=json.loads(job["analysis"]) if job["analysis"] else {}
    def tailor_async():
        try:
            tailored=tailor_resume_with_claude(resume_text,job["description"] or "",analysis)
            db_execute("UPDATE jobs SET tailored_resume=%s WHERE id=%s",(tailored,job_id))
        except Exception as e:
            db_execute("UPDATE jobs SET tailored_resume=%s WHERE id=%s",(f"[Tailoring failed: {e}]",job_id))
    background_tasks.add_task(tailor_async)
    return {"status":"tailoring_started","job_id":job_id}

@app.get("/jobs/{job_id}/tailored")
def get_tailored_resume(job_id: str):
    row=db_fetchone("SELECT tailored_resume,title,company FROM jobs WHERE id=%s",(job_id,))
    if not row:
        raise HTTPException(404,"Job not found")
    return {"job_id":job_id,"title":row["title"],"company":row["company"],"tailored_resume":row["tailored_resume"],"ready":bool(row["tailored_resume"])}

@app.get("/jobs/{job_id}/tailored/pdf")
def download_tailored_pdf(job_id: str):
    row=db_fetchone("SELECT tailored_resume,title,company FROM jobs WHERE id=%s",(job_id,))
    if not row or not row["tailored_resume"]:
        raise HTTPException(404,"Tailored resume not ready yet.")
    pdf=generate_resume_pdf(row["tailored_resume"],row["title"] or "",row["company"] or "")
    fname=f"resume_tailored_{(row['company'] or 'job').replace(' ','_')}.pdf"
    return StreamingResponse(io.BytesIO(pdf),media_type="application/pdf",
                             headers={"Content-Disposition":f'attachment; filename="{fname}"'})

@app.post("/scan")
def trigger_scan(body: ScanRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(run_scan,body.search_term,body.location,body.remote_only,body.sites,body.results_per_site)
    return {"status":"scan started"}

@app.get("/scan/config")
def get_scan_config():
    row=db_fetchone("SELECT * FROM scan_config WHERE id=1")
    cfg=dict(row)
    cfg["sites"]=cfg["sites"].split(",") if cfg["sites"] else []
    return cfg

@app.post("/scan/config")
def update_scan_config(body: ScanConfigUpdate):
    updates={k:v for k,v in body.dict().items() if v is not None}
    if "sites" in updates:
        updates["sites"]=",".join(updates["sites"])
    if not updates:
        return {"ok":True}
    set_clause=", ".join(f"{k}=%s" for k in updates)
    db_execute(f"UPDATE scan_config SET {set_clause} WHERE id=1",list(updates.values()))
    return {"ok":True}

# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("\n╔══════════════════════════════════════╗")
    print("║   ApplyOS Backend v3 (Postgres)       ║")
    print("║   API:  http://localhost:8000          ║")
    print("║   Docs: http://localhost:8000/docs     ║")
    print("╚══════════════════════════════════════╝\n")
    if not DATABASE_URL:
        print("⚠️  DATABASE_URL not set")
    if not ANTHROPIC_KEY:
        print("⚠️  ANTHROPIC_API_KEY not set")
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
