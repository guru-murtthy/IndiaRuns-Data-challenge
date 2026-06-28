from preprocessing.normalize_skills import normalize_skill

# Re-expose skill normalization mapping if needed, or define custom taxonomy mappings
def get_normalized_skill(skill_name):
    return normalize_skill(skill_name)
