import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.server.app import app

client = app.test_client()

def post_json(path, payload):
    res = client.post(path, json=payload)
    try:
        return res.get_json()
    except Exception:
        return {"status": res.status_code, "text": res.get_data(as_text=True)}

cases = [
    ("/chat", {"message": "hello", "user_id": "test_user"}),
    ("/chat", {"message": "what did i just say", "user_id": "test_user"}),
    ("/chat", {"message": "What is artificial intelligence?", "user_id": "test_user"}),
    ("/search", {"query": "artificial intelligence", "user_id": "test_user"}),
    ("/search", {"query": "capital of France", "user_id": "test_user"}),
]

for path, payload in cases:
    data = post_json(path, payload)
    print("\n==>", path, payload.get("message") or payload.get("query"))
    if path == "/chat":
        print("reply:", data.get("reply"))
    else:
        print("answer:", data.get("answer"))
        print("sources:", len(data.get("results") or []))
