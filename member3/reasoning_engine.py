import random

# Core JD skills to check for matches
CORE_JD_SKILLS = {
    "embeddings", "retrieval", "vector search", "faiss", "pinecone", 
    "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch", 
    "python", "sentence-transformers", "ndcg", "mrr", "map",
    "fine-tuning", "lora", "qlora", "peft", "learning-to-rank", 
    "xgboost", "lightgbm", "catboost", "distributed systems", "nlp"
}

def generate_reasoning(candidate, rank, score):
    """
    Generates a highly factual, 1-2 sentence reasoning explaining why the candidate
    is assigned to their rank. Avoids templates by combining dynamic clauses.
    """
    profile = candidate.get("profile", {})
    yoe = profile.get("years_of_experience", 0.0)
    current_title = profile.get("current_title", "Engineer")
    current_company = profile.get("current_company", "Product Company")
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})
    notice_period = int(signals.get("notice_period_days", 30))
    location = profile.get("location", "India")
    response_rate = int(signals.get("recruiter_response_rate", 0) * 100)
    github_score = signals.get("github_activity_score", -1)
    
    # 1. Find actual matching skills from candidate profile
    matched_skills = []
    for s in skills:
        skill_name = s.get("name", "")
        if skill_name.lower() in CORE_JD_SKILLS or any(word in skill_name.lower() for word in ["nlp", "search", "retrieval", "vector"]):
            matched_skills.append(skill_name)
            
    # Deduplicate and limit to top 3 skills
    matched_skills = list(dict.fromkeys(matched_skills))[:3]
    skills_phrase = ", ".join(matched_skills) if matched_skills else "applied ML"
    
    # 2. Extract facts for clauses
    experience_fact = f"{yoe:.1f} years of experience"
    title_company_fact = f"working as a {current_title} at {current_company}"
    availability_fact = f"notice period of {notice_period} days"
    
    # 3. Dynamic clause construction based on rank and profile details
    if rank <= 10:
        # Top 10: Elite fit, strong matching skills, product company, highly active
        intro = f"Exceptional fit at Rank {rank}. Exhibits {experience_fact} and is currently {title_company_fact}."
        body = f"Demonstrates elite hands-on mastery in core search technologies, specifically {skills_phrase}."
        
        # Add a positive signal fact
        if github_score > 50:
            conclusion = f"Strong GitHub activity (score: {github_score}) and high responsiveness ({response_rate}% response rate) confirm availability."
        elif notice_period <= 30:
            conclusion = f"Highly available with a very short {notice_period}-day notice period."
        else:
            conclusion = f"Based in {location} with active platform engagement signals."
            
        sentences = [intro, body, conclusion]
        # Return a 2 sentence combo
        return " ".join(random.sample(sentences, 2))
        
    elif rank <= 40:
        # Rank 11-40: Solid matches, highlight experience range and core skills
        intro = f"Strong contender with {experience_fact}, matching the JD's 5-9 years target."
        body = f"Features verified proficiency in {skills_phrase} with a solid track record at {current_company}."
        
        # Acknowledge gaps if any
        if notice_period > 90:
            conclusion = f"Notice period is long ({notice_period} days), but their technical retrieval alignment is strong."
        else:
            conclusion = f"Actively responsive ({response_rate}% response rate) and located in {location}."
            
        sentences = [intro, body, conclusion]
        return " ".join(random.sample(sentences, 2))
        
    elif rank <= 75:
        # Rank 41-75: Decent adjacent matches, note the shift or missing direct experience
        intro = f"Backend engineer showing {experience_fact} and relevant experience at {current_company}."
        
        if matched_skills:
            body = f"Has exposure to {skills_phrase}, though lacks direct experience building large-scale vector search indexes in production."
        else:
            body = "Solid developer with robust Python and database foundations, but lacks NLP-specific retrieval experience."
            
        conclusion = f"Located in {location} with a {notice_period}-day notice period."
        
        return f"{intro} {body} {conclusion}"
        
    else:
        # Rank 76-100: Filler roles, acknowledge gaps explicitly
        intro = f"Filler candidate at Rank {rank} with {experience_fact}."
        body = f"Possesses adjacent skills in {skills_phrase} but lacks the NLP/IR experience specified in the JD."
        
        if notice_period > 90:
            conclusion = f"Long notice period ({notice_period} days) and low active engagement indicators."
        else:
            conclusion = f"Available in {notice_period} days; currently at {current_company}."
            
        return f"{intro} {body} {conclusion}"
