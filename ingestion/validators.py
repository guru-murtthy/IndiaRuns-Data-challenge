import datetime

def parse_date(date_str):
    """
    Parses a date string in YYYY-MM-DD format.
    """
    if not date_str:
        return None
    try:
        return datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
    except Exception:
        return None
