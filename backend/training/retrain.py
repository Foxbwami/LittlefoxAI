import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.core import config
from backend.training.train import train_model


def retrain():
    print("Retraining on new data...")
    train_model(
        config.LEARNING_LOG_PATH,
        epochs=config.RETRAIN_EPOCHS,
        steps_per_epoch=config.RETRAIN_STEPS,
    )


if __name__ == "__main__":
    retrain()
