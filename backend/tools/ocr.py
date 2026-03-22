from backend.core import config


def extract_text_from_image(image_bytes):
    if not config.OCR_ENABLED:
        return {"error": "OCR not enabled."}
    try:
        import pytesseract
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(img)
        return {"text": text.strip()}
    except Exception:
        return {"error": "OCR model not available."}
