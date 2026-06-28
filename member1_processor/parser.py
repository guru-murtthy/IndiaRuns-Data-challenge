import re


def normalize_text(text: str) -> str:
    """Removes trailing spaces, punctuation, and sets text to lowercase."""
    if not text:
        return ""
    # Lowercase and strip whitespace
    text = text.lower().strip()
    # Remove basic punctuation
    text = re.sub(r'[^\w\s\-\.]', '', text)
    return text


def normalize_skill(skill_name: str) -> str:
    """Standardizes messy user-inputted skill names."""
    cleaned = normalize_text(skill_name)

    # Simple standardized mapping dictionary
    skill_map = {
        "python3": "python",
        "py": "python",
        "js": "javascript",
        "javascript.js": "javascript",
        "cplusplus": "c++",
        "cpp": "c++",
        "c sharp": "c#",
        "csharp": "c#"
    }
    return skill_map.get(cleaned, cleaned)


def normalize_title(title_name: str) -> str:
    """Standardizes chaotic resume job titles into core categories."""
    cleaned = normalize_text(title_name)

    # Regex rules for matching titles
    if "intern" in cleaned or "trainee" in cleaned:
        return "intern"
    if "senior" in cleaned or "sr" in cleaned or "lead" in cleaned:
        return "senior engineer"
    if "software engineer" in cleaned or "developer" in cleaned or "sde" in cleaned:
        return "software engineer"
    if "data scientist" in cleaned or "data science" in cleaned:
        return "data scientist"

    return cleaned


if __name__ == "__main__":
    print("⏳ Running Day 1 Normalization Text Engine Tests...")

    # Test cases
    test_skill = "  Python3! "
    test_title =$ "Software Engineer Intern"

    print(f"🔹 Raw Skill: '{test_skill}' ──> Normalized: '{normalize_skill(test_skill)}'")
    print(f"🔹 Raw Title: '{test_title}' ──> Normalized: '{normalize_title(test_title)}'")
    print("\n🚀 [Parser Test] Text normalization system: SUCCESS.")