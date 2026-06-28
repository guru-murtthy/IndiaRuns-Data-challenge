from typing import Dict, Any

class CandidateTextBuilder:
    """
    Builder class to convert a Candidate object or dictionary into a structured text
    representation suitable for generating embeddings.
    """
    
    @staticmethod
    def build_text(candidate: Dict[str, Any]) -> str:
        """
        Converts a candidate dictionary into a formatted string.
        Gracefully ignores missing values.
        
        Args:
            candidate: A dictionary containing candidate information.
            
        Returns:
            A well-formatted string summarizing the candidate's profile.
        """
        parts = []
        
        title = candidate.get('title')
        if title:
            parts.append(str(title))
            
        skills = candidate.get('skills')
        if skills:
            if isinstance(skills, list):
                skills_str = "\n".join(str(skill) for skill in skills)
            else:
                skills_str = str(skills)
            parts.append(f"Skills:\n{skills_str}")
            
        experience = candidate.get('experience')
        if experience:
            parts.append(f"Experience:\n{experience}")
            
        projects = candidate.get('projects')
        if projects:
            parts.append(f"Projects:\n{projects}")
            
        education = candidate.get('education')
        if education:
            parts.append(f"Education:\n{education}")
            
        location = candidate.get('location')
        if location:
            parts.append(f"Location:\n{location}")
            
        company = candidate.get('company')
        if company:
            parts.append(f"Company:\n{company}")
            
        return "\n\n".join(parts)
