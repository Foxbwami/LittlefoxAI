from backend.core import config


def transcribe_audio(audio_bytes):
    if not config.SPEECH_ENABLED:
        return {"error": "Speech model not enabled."}
    return {"error": "Speech model is a stub. Wire a real ASR model to enable."}
