import os
from backend.core import config


def collect_feedback(response, rating):
    os.makedirs(os.path.dirname(config.FEEDBACK_PATH), exist_ok=True)
    with open(config.FEEDBACK_PATH, "a", encoding="utf-8") as f:
        f.write(f"{response} | {rating}\n")


def save_feedback(input_text, response, rating):
    os.makedirs(os.path.dirname(config.FEEDBACK_PATH), exist_ok=True)
    with open(config.FEEDBACK_PATH, "a", encoding="utf-8") as f:
        f.write(f"{input_text} | {response} | {rating}\n")


def reward_score(response):
    text = (response or "").lower()
    if not text:
        return -1
    if any(bad in text for bad in ["robotic", "as an ai", "i cannot", "i'm unable"]):
        return -1
    return 1
