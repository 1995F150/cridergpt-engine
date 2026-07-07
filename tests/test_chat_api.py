import pytest
from fastapi.testclient import TestClient
from app import app
import uuid

client = TestClient(app)

def test_chat_optional_ids():
    # POST {"message": "hey"} with no user_id and no conversation_id
    response = client.post("/chat", json={"message": "hey"})
    
    # Assert it returns a 200 HTTP response
    assert response.status_code == 200
    
    data = response.json()
    # Assert that a valid UUID conversation_id is echoed back in the response JSON
    conv_id = data.get("conversation_id")
    assert conv_id is not None
    # Check if it's a valid UUID
    uuid.UUID(conv_id)
