"""Skill lexicon for overlap features.

Spans the 24 resume categories in the corpus, not just tech -- the posting corpus is
mostly non-technical (healthcare, trades, hospitality, finance, education), and a
tech-only lexicon would make the overlap features silently useless for most pairs.

Multi-word entries are matched as phrases; single words on token boundaries. Everything
is lowercased at match time.
"""
from __future__ import annotations

import re

SKILLS: set[str] = {
    # --- software / data ---
    "python", "java", "javascript", "typescript", "c++", "c#", "golang", "rust", "ruby",
    "php", "swift", "kotlin", "scala", "perl", "matlab", "r", "sql", "nosql", "html",
    "css", "react", "angular", "vue", "node.js", "django", "flask", "fastapi", "spring",
    ".net", "rails", "express", "graphql", "rest api", "microservices",
    "aws", "azure", "gcp", "google cloud", "docker", "kubernetes", "terraform", "ansible",
    "jenkins", "ci/cd", "git", "github", "gitlab", "linux", "unix", "bash", "powershell",
    "postgresql", "mysql", "mongodb", "redis", "oracle", "sql server", "snowflake",
    "databricks", "spark", "hadoop", "kafka", "airflow", "etl", "data warehouse",
    "machine learning", "deep learning", "nlp", "computer vision", "tensorflow",
    "pytorch", "scikit-learn", "pandas", "numpy", "data analysis", "data science",
    "statistics", "regression", "a/b testing", "data visualization", "tableau",
    "power bi", "looker", "qlik", "excel", "vba", "sas", "spss",
    "agile", "scrum", "kanban", "jira", "confluence", "devops", "sre",
    "cybersecurity", "penetration testing", "siem", "firewall", "encryption",
    "network security", "incident response", "soc", "iso 27001", "nist",
    # --- finance / accounting ---
    "accounting", "bookkeeping", "accounts payable", "accounts receivable", "payroll",
    "general ledger", "journal entries", "reconciliation", "financial reporting",
    "financial analysis", "forecasting", "budgeting", "variance analysis", "gaap",
    "ifrs", "sox", "audit", "internal controls", "tax", "quickbooks", "sap", "netsuite",
    "erp", "cpa", "cfa", "valuation", "financial modeling", "underwriting", "credit analysis",
    "risk management", "compliance", "aml", "kyc", "treasury", "investments", "portfolio",
    # --- healthcare ---
    "patient care", "clinical", "nursing", "rn", "lpn", "bls", "acls", "cpr",
    "phlebotomy", "medical records", "emr", "ehr", "epic", "cerner", "hipaa",
    "medical coding", "icd-10", "cpt", "medical billing", "pharmacy", "triage",
    "vital signs", "infection control", "care plans", "medication administration",
    # --- sales / marketing / pr ---
    "salesforce", "hubspot", "crm", "lead generation", "cold calling", "prospecting",
    "account management", "business development", "b2b", "b2c", "negotiation",
    "seo", "sem", "google analytics", "google ads", "social media", "content marketing",
    "email marketing", "brand management", "market research", "copywriting",
    "public relations", "media relations", "press releases", "event planning",
    # --- design / creative ---
    "adobe photoshop", "illustrator", "indesign", "adobe creative", "figma", "sketch",
    "after effects", "premiere pro", "ux", "ui", "wireframing", "prototyping",
    "typography", "graphic design", "branding", "video editing", "3d modeling", "autocad",
    "solidworks", "revit",
    # --- hr ---
    "recruiting", "talent acquisition", "onboarding", "employee relations", "benefits",
    "compensation", "performance management", "hris", "workday", "adp", "fmla",
    "eeo", "osha", "labor relations", "training and development", "succession planning",
    # --- operations / logistics / construction / trades ---
    "supply chain", "procurement", "inventory management", "logistics", "warehouse",
    "lean", "six sigma", "kaizen", "process improvement", "quality assurance",
    "quality control", "iso 9001", "project management", "pmp", "ms project",
    "blueprints", "osha 30", "construction management", "estimating", "scheduling",
    "hvac", "welding", "electrical", "plumbing", "carpentry", "forklift", "cdl",
    "preventive maintenance", "troubleshooting", "schematics",
    # --- education / hospitality / aviation / legal ---
    "curriculum development", "lesson planning", "classroom management", "iep",
    "differentiated instruction", "assessment", "tutoring",
    "food safety", "servsafe", "menu planning", "culinary", "catering", "pos systems",
    "customer service", "hospitality", "front desk", "reservations",
    "faa", "pilot", "flight operations", "aircraft maintenance", "a&p",
    "litigation", "legal research", "contracts", "paralegal", "westlaw", "depositions",
    "due diligence", "regulatory",
    # --- general professional ---
    "leadership", "team management", "communication", "problem solving", "time management",
    "strategic planning", "stakeholder management", "presentation", "coaching", "mentoring",
    "vendor management", "cross-functional", "reporting", "documentation",
}

_MULTI = sorted((s for s in SKILLS if " " in s or "." in s or "/" in s or "+" in s or "#" in s),
                key=len, reverse=True)
_SINGLE = {s for s in SKILLS if s not in set(_MULTI)}

_TOKEN_RE = re.compile(r"[a-z0-9+#.\-]+")


def extract(text: str) -> set[str]:
    """Return the set of lexicon skills present in `text`."""
    if not text:
        return set()
    low = text.lower()
    found = {phrase for phrase in _MULTI if phrase in low}
    tokens = set(_TOKEN_RE.findall(low))
    found |= tokens & _SINGLE
    return found
