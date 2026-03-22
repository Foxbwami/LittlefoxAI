import os
from backend.core import config


def log_interaction(user, ai):
    os.makedirs(os.path.dirname(config.LEARNING_LOG_PATH), exist_ok=True)
    with open(config.LEARNING_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"User: {user}\nAI: {ai}\n")
    _maybe_signal_retrain()


def _maybe_signal_retrain():
    try:
        counter_path = os.path.join(os.path.dirname(config.LEARNING_LOG_PATH), "interaction.count")
        count = 0
        if os.path.exists(counter_path):
            with open(counter_path, "r", encoding="utf-8") as f:
                count = int((f.read() or "0").strip() or 0)
        count += 1
        with open(counter_path, "w", encoding="utf-8") as f:
            f.write(str(count))
        if count % config.RETRAIN_EVERY_INTERACTIONS == 0:
            with open(config.RETRAIN_SIGNAL_PATH, "w", encoding="utf-8") as f:
                f.write(str(count))
    except Exception:
        return
