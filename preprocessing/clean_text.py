import re

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s\-\.]', '', text)
    return text
