import os
from backend.core import config

_gen = None
_gen_failed = False


def get_rewriter():
    global _gen, _gen_failed
    if _gen_failed:
        return None
    if _gen is None:
        try:
            from transformers import pipeline
            model_path = config.HUMANIZER_MODEL_PATH
            if model_path and os.path.exists(model_path):
                _gen = pipeline("text2text-generation", model=model_path)
            else:
                _gen = pipeline("text2text-generation", model=config.HF_GENERATION_MODEL)
        except Exception:
            _gen_failed = True
            return None
    return _gen
