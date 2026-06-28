from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from orchestrator.adapters.member1_adapter import RAW_CANDIDATE_PATH
from orchestrator.models import JobInput
from orchestrator.services.export_service import export_run_to_csv
from orchestrator.services.orchestrator import run_pipeline
from orchestrator.storage import run_store

DEFAULT_JD = (
    "Senior AI/ML Engineer — Founding Team. Build an intelligent candidate ranking "
    "system using Python, FAISS, LightGBM, FastAPI and Docker. 4+ years experience "
    "in search, NLP, retrieval, or semantic matching required. Bangalore preferred."
)

app = FastAPI(title="India Runs Member 4 Demo", version="0.1.0")
templates = Jinja2Templates(directory="orchestrator/templates")
app.mount("/static", StaticFiles(directory="orchestrator/static"), name="static")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "recent_runs": run_store.list_recent(),
            "default_title": "Senior AI/ML Engineer",
            "default_jd": DEFAULT_JD,
            "dataset_ready": RAW_CANDIDATE_PATH.exists(),
        },
    )


@app.post("/runs")
def create_run(
    title: str = Form(default="Senior AI Engineer"),
    description: str = Form(...),
    top_k: int = Form(default=100),
) -> RedirectResponse:
    job = JobInput(title=title.strip() or "AI Engineer", description=description, top_k=top_k)
    run = run_pipeline(job)
    run_store.save(run)
    return RedirectResponse(url=f"/runs/{run.run_id}", status_code=303)


@app.get("/runs/{run_id}", response_class=HTMLResponse)
def run_details(request: Request, run_id: str) -> HTMLResponse:
    run = run_store.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    export_path = Path("submission/outputs") / f"{run_id}.csv"
    return templates.TemplateResponse(
        request,
        "run.html",
        {
            "run": run,
            "export_exists": export_path.exists(),
            "latest_submission_path": Path("data/output/submission.csv").as_posix(),
        },
    )


@app.get("/runs/{run_id}/export")
def export_results(run_id: str) -> FileResponse:
    run = run_store.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    output_path = export_run_to_csv(run)
    return FileResponse(
        output_path,
        media_type="text/csv",
        filename=output_path.name,
    )
