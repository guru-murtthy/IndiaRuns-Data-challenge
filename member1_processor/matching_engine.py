class StudentMatcher:
    def __init__(self, jd_data):
        """
        jd_data is a dictionary containing the parsed Job Description.
        """
        # Convert JD skills to a set for instant O(1) lookups
        self.jd_skills = set(skill.lower().strip() for skill in jd_data.get("skills", []))
        self.min_exp = jd_data.get("min_exp", 0)
        self.target_location = jd_data.get("location", "").lower().strip()
        self.required_degree = jd_data.get("degree", "").lower().strip()
        self.target_title = jd_data.get("title", "").lower().strip()

    def match_student(self, student_data):
        """
        student_data is a single parsed student/candidate dictionary.
        """
        # 1. SKILL MATCH (Find the overlap percentage)
        student_skills = set(skill.lower().strip() for skill in student_data.get("skills", []))
        if not self.jd_skills:
            skill_score = 1.0
        else:
            overlap = self.jd_skills.intersection(student_skills)
            skill_score = len(overlap) / len(self.jd_skills)

        # 2. EXPERIENCE MATCH
        student_exp = student_data.get("years_of_experience", 0)
        exp_match = 1.0 if student_exp >= self.min_exp else (student_exp / self.min_exp if self.min_exp > 0 else 0.0)

        # 3. TITLE MATCH (Check if target title is inside their past titles)
        current_title = student_data.get("current_title", "").lower().strip()
        title_match = 1.0 if self.target_title in current_title or current_title in self.target_title else 0.0

        # 4. LOCATION MATCH (Boolean check)
        student_location = student_data.get("location", "").lower().strip()
        location_match = 1.0 if student_location == self.target_location or self.target_location == "remote" else 0.0

        # 5. EDUCATION MATCH (Degree validation)
        student_degree = student_data.get("degree", "").lower().strip()
        education_match = 1.0 if self.required_degree in student_degree else 0.0

        # Output payload for Member 2's ranker and Member 3's reasoning engine
        return {
            "candidate_id": student_data.get("id"),
            "metrics": {
                "skill_match_score": round(skill_score, 2),
                "experience_match_score": round(exp_match, 2),
                "title_match_score": title_match,
                "location_match_score": location_match,
                "education_match_score": education_match
            }
        }