import re
from sklearn.feature_extraction.text import TfidfVectorizer

STOPWORDS = {
    "the", "is", "are", "was", "were", "a", "an", "and", "or", "of",
    "to", "in", "on", "for", "with", "by", "from", "this", "that",
    "what", "how", "why", "explain", "describe", "define", "discuss",
    "using", "use", "give", "example", "examples"
}

def extract_keywords(text: str, top_n=6):
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", text.lower())
    words = [w for w in words if w not in STOPWORDS]

    if not words:
        return []

    # For a single question, TF-IDF behaves mostly like term weighting.
    # Frequency + technical-token preservation is used as an explainable fallback.
    counts = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1

    ranked = sorted(counts, key=lambda x: (-counts[x], -len(x), x))
    return ranked[:top_n]
