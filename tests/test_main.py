from fastapi.testclient import TestClient
from spirit.main import app

client = TestClient(app)

def test_ping():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
