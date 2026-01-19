import pytest
from datetime import date
from fastapi.testclient import TestClient

pytestmark = pytest.mark.asyncio

async def test_log_execution(client: TestClient, goal, token):
    r = client.post(
        "/api/trajectory/execute",
        json={"objective_text": "Write 500 words", "executed": True, "day": str(date.today())},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert r.status_code == 200
    data = r.json()
    assert data["objective_text"] == "Write 500 words"
    assert data["executed"] is True

async def test_history(client: TestClient, goal, token):
    # log two days
    for i in range(2):
        day = date.today().replace(day=date.today().day - i)
        client.post(
            "/api/trajectory/execute",
            json={"objective_text": f"Day {i}", "executed": True, "day": str(day)},
            headers={"Authorization": f"Bearer {token}"}
        )
    r = client.get("/api/trajectory/history", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    assert len(r.json()) == 2
