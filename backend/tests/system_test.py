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

print("\nGENERAL CHAT")
print(post_json("/chat", {"message": "Hi there", "user_id": "test_user"}).get("reply"))

print("\nCODING CHAT")
print(post_json("/chat", {"message": "Write a Python function to reverse a string", "user_id": "test_user"}).get("reply"))

print("\nBROWSING SEARCH")
print(post_json("/search", {"query": "capital of Japan", "user_id": "test_user"}).get("answer"))

print("\nACADEMIC RESEARCH")
academic = post_json("/search", {
    "query": "Summarize peer review in academic research",
    "user_id": "acad_user",
    "tone": "academic",
    "citation_style": "APA",
    "academic_template": "summary"
})
print(academic.get("answer"))
print("Citations:", academic.get("citations")[:2] if academic.get("citations") else None)

print("\nACADEMIC ESSAY OUTLINE")
outline = post_json("/search", {
    "query": "Create an outline for an essay on data privacy",
    "user_id": "acad_user",
    "tone": "academic",
    "citation_style": "MLA",
    "academic_template": "outline"
})
print(outline.get("answer"))
