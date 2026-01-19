import pytest
from fastapi.testclient import TestClient
from spirit.models import GoalState

pytestmark = pytest.mark.asyncio

async def test_declare_goal(client: TestClient, user, token):
    r = client.post(
        "/api/goals",
        json={"text": "Write every day"},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert r.status_code == 200
    data = r.json()
    assert data["text"] == "Write every day"
    assert data["state"] == GoalState.active

async def test_only_one_active_goal(client: TestClient, user, token):
    # first goal
    client.post("/api/goals", json={"text": "First"}, headers={"Authorization": f"Bearer {token}"})
    # second should fail
    r = client.post("/api/goals", json={"text": "Second"}, headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 409
