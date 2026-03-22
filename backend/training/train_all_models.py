import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.training import train_ner, train_safety, train_emotion, train_factcheck


def main():
    train_ner.main()
    train_safety.main()
    train_emotion.main()
    train_factcheck.main()
    print("All models trained.")


if __name__ == "__main__":
    main()
