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
    ("/search", {"query": "Explain reinforcement learning in two sentences", "user_id": "acad_user", "tone": "academic", "citation_style": "APA", "academic_template": "summary"}),
    ("/search", {"query": "Provide an academic definition of machine learning", "user_id": "acad_user", "tone": "academic", "citation_style": "MLA", "academic_template": "abstract"}),
    ("/search", {"query": "Summarize the importance of peer review in research", "user_id": "acad_user", "tone": "academic", "citation_style": "CHICAGO", "academic_template": "literature_review"}),
    ("/search", {"query": "Write a research questions list on climate change adaptation", "user_id": "acad_user", "tone": "academic", "citation_style": "APA", "academic_template": "research_questions"}),
    ("/search", {"query": "Create an outline for an essay on data privacy", "user_id": "acad_user", "tone": "academic", "citation_style": "APA", "academic_template": "outline"}),
    ("/search", {"query": "Thesis statement on renewable energy adoption", "user_id": "acad_user", "tone": "academic", "citation_style": "APA", "academic_template": "thesis"}),
]

for path, payload in cases:
    data = post_json(path, payload)
    print("\n==>", payload.get("query"))
    print("answer:\n", data.get("answer"))
    if data.get("citations"):
        print("citations:", data.get("citations")[:2])
    if data.get("bibtex"):
        print("bibtex sample:", data.get("bibtex").split('\n')[0])
    print("sources:", len(data.get("results") or []))
