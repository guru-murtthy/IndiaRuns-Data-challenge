from fastapi.testclient import TestClient

from orchestrator.main import app


client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_run_creation_and_export() -> None:
    response = client.post(
        "/runs",
        data={
            "title": "AI Engineer",
            "description": (
                "Need Python, FAISS, FastAPI, Docker, embeddings, and ranking skills "
                "with 3 years experience in retrieval systems."
            ),
            "top_k": 5,
        },
        follow_redirects=False,
    )
    assert response.status_code == 303

    run_url = response.headers["location"]
    details_response = client.get(run_url)
    assert details_response.status_code == 200
    assert "Ranked Shortlist" in details_response.text

    export_response = client.get(f"{run_url}/export")
    assert export_response.status_code == 200
    assert export_response.headers["content-type"].startswith("text/csv")
