import re

def infer_paper_metadata(filename):
    """Infer year/semester/session from common academic-paper filenames."""
    stem = re.sub(r"\.[^.]+$", "", filename)
    lower = stem.lower()

    year_match = re.search(r"(20\d{2})", stem)
    academic_year = year_match.group(1) if year_match else ""

    sem_match = re.search(r"(?:sem(?:ester)?)[ _-]?([1-8])(?:\b|[_-])", lower)
    semester = f"Semester {sem_match.group(1)}" if sem_match else ""

    normalized = lower.replace("-", "").replace("_", "").replace(" ", "")
    session = ""
    for token, label in [
        ("forenoon", "Forenoon"), ("afternoon", "Afternoon"),
        ("shift1", "Shift 1"), ("shift2", "Shift 2"),
        ("set1", "Set 1"), ("set2", "Set 2")
    ]:
        if token in normalized:
            session = label
            break

    return academic_year, semester, session
