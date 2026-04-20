# ApplyOS — Full Stack Job Tracker

Auto-scrapes LinkedIn, Indeed, Glassdoor & ZipRecruiter. Scores every job 
against your resume with Claude AI. Tracks your full application pipeline.

---

## Setup (2 steps)

### 1. Backend

```bash
# Install dependencies
pip install -r requirements.txt

# Set your Anthropic API key (get one at console.anthropic.com)
export ANTHROPIC_API_KEY=sk-ant-...         # Mac/Linux
set ANTHROPIC_API_KEY=sk-ant-...            # Windows CMD
$env:ANTHROPIC_API_KEY="sk-ant-..."         # Windows PowerShell

# Run the server
python backend.py
```

Server starts at **http://localhost:8000**  
Interactive API docs at **http://localhost:8000/docs**

### 2. Frontend

Open the `frontend.jsx` artifact in Claude.ai — it auto-connects to localhost:8000.

Or run it locally with Vite:
```bash
npm create vite@latest applyjobs -- --template react
cd applyjobs
cp ../frontend.jsx src/App.jsx
npm install && npm run dev
```

---

## Features

| Feature | Description |
|---|---|
| **Auto-scan** | Scrapes LinkedIn, Indeed, Glassdoor, ZipRecruiter |
| **AI scoring** | Claude scores every job 0–100 against your resume |
| **Fit analysis** | Strengths, weaknesses, gap analysis per job |
| **Resume bullets** | Tailored bullets + keywords for each application |
| **Networking** | Who to contact + outreach message template |
| **Pipeline** | Track status: New → Reviewing → Applied → Interview → Offer |
| **Auto-scan** | Set it and forget — scans every N hours automatically |
| **Persistent DB** | SQLite database, survives restarts |

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | Your Anthropic API key for Claude AI scoring |

---

## API Endpoints

```
GET    /jobs              List all jobs (filterable, sortable)
GET    /jobs/stats        Dashboard stats
POST   /jobs/manual       Add a job manually (triggers async AI analysis)
PATCH  /jobs/{id}/status  Update job status
DELETE /jobs/{id}         Delete a job
DELETE /jobs              Clear all jobs

GET    /resume            Get saved resume
POST   /resume            Save resume

POST   /scan              Trigger a job scan now
GET    /scan/config       Get auto-scan configuration
POST   /scan/config       Update auto-scan configuration
```

---

## Notes

- **LinkedIn scraping**: May require a logged-in session. If blocked, use Indeed/Glassdoor.
- **Rate limits**: python-jobspy respects rate limits. Don't scan too frequently.
- **Auto-scan**: Enable in the "Scan Jobs" tab. Runs in background every N hours.
- **Database**: Stored in `applyjobs.db` in the same folder as `backend.py`.
