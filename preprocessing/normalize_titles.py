from .clean_text import clean_text

def normalize_title(title_name):
    """
    Normalizes candidate role titles to standard forms.
    """
    title = clean_text(title_name)
    if "backend" in title:
        return "backend engineer"
    if "machine learning" in title or " ml " in title or title.startswith("ml ") or title.endswith(" ml") or "ai engineer" in title or "artificial intelligence" in title:
        return "ml engineer"
    if "data engineer" in title:
        return "data engineer"
    if "operations" in title:
        return "operations manager"
    if "marketing" in title:
        return "marketing manager"
    if "support" in title or "helpdesk" in title:
        return "customer support"
    if "manager" in title or "director" in title or "lead" in title or "head" in title:
        return "lead/manager"
    return "other"
