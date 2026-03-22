import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.server.app import app

client = app.test_client()

QUESTIONS = [
    ("Logical Reasoning (Mislabeled Boxes)", "Three boxes are wrongly labeled 'apples,' 'oranges,' and 'mixed.' How can you determine the true content by picking one fruit from only one box?"),
    ("Mathematical Optimization", "Given 150m of fencing to build a 3-sided rectangular pen against a barn, what dimensions maximize the area?"),
    ("Complex Instruction Following", "Summarize a text in three bullets, extract proper nouns alphabetically, and determine the sentiment."),
    ("Creative Generation", "Write a noir-style dialogue between a robot detective and a human, requiring specific technical terminology and a gritty tone."),
    ("Coding/Refactoring", "Refactor a provided inefficient Python function, highlighting 'code smells'."),
    ("Ethical Reasoning", "Analyze the ethical dilemma of autonomous vehicles choosing between passenger or pedestrian safety using two philosophical frameworks."),
    ("Fact-Checking/Anomalies", "Evaluate the plausibility of a 'mammal that lays eggs and lives underwater'."),
    ("Physics/Common Sense", "What is the speed of sound in a vacuum? Explain why."),
    ("Counterfactual Scenarios", "Describe a world without the internet, focusing on alternative communication technologies."),
    ("Pattern Recognition", "Find the next number in the sequence 2, 6, 12, 20, 30, and define the rule."),
]


def post_chat(message, tone="default", use_web=True):
    payload = {
        "message": message,
        "user_id": "eval_user",
        "tone": tone,
        "use_web": use_web,
    }
    res = client.post("/chat", json=payload)
    try:
        return res.get_json().get("reply", "")
    except Exception:
        return res.get_data(as_text=True)


if __name__ == "__main__":
    for idx, (title, question) in enumerate(QUESTIONS, start=1):
        print(f"\n=== {title} ===")
        user_id = f"eval_user_{idx}"
        payload = {
            "message": question,
            "user_id": user_id,
            "tone": "default",
            "use_web": True,
        }
        res = client.post("/chat", json=payload)
        data = res.get_json(silent=True)
        if isinstance(data, dict) and "reply" in data:
            print(data["reply"])
        else:
            print(res.get_data(as_text=True))
