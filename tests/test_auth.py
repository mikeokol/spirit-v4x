import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.asyncio

async def test_register(client: TestClient):
    r = client.post("/api/auth/register", json={"email": "new@spirit.local", "password": "str0ng!"})
    assert r.status_code == 200
    assert "access_token" in r.json()

async def test_login(client: TestClient, user):
    r = client.post("/api/auth/login", data={"username": user.email, "password": "str0ng!"})
    assert r.status_code == 200
    assert "access_token" in r.json()
