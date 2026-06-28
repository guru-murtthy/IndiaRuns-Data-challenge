import re

def clean_text(text):
    """
    Cleans text by stripping whitespace and normalizing spaces to a single space.
    """
    if not text:
        return ""
    return re.sub(r'\s+', ' ', text.strip().lower())
