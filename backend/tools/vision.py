from backend.core import config


def describe_image(image_bytes):
    if not config.VISION_ENABLED:
        return {"error": "Vision model not enabled."}
    return {"error": "Vision model is a stub. Wire a real model to enable."}
