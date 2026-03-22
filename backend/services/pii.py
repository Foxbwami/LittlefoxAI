import re


def redact_pii(text):
    if not text:
        return text
    # emails
    text = re.sub(r"[\w\.-]+@[\w\.-]+", "[email]", text)
    # phone numbers (simple)
    # Require 10+ digits to avoid masking years like 2025
    text = re.sub(r"\b\+?\d[\d\s\-\(\)]{9,}\b", "[phone]", text)
    # possible SSN
    text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[ssn]", text)
    return text
