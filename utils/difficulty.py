
import re

HARD_TERMS = [
    "design", "derive", "optimize", "critically evaluate",
    "architecture", "prove", "complexity analysis", "develop",
    "viterbi", "cyk", "lesk algorithm", "hobbs algorithm",
    "good-turing", "word sense disambiguation"
]
MEDIUM_TERMS = [
    "explain", "analyze", "analyse", "compare", "implement",
    "demonstrate", "calculate", "algorithm", "differentiate",
    "disambiguate", "perplexity", "morphological", "parse", "tag"
]

def predict_difficulty(text: str, question_type: str, marks):
    lower = text.lower()
    score = 0.0

    word_count = len(text.split())
    score += min(word_count / 18.0, 1.5)

    if marks is not None:
        score += min(float(marks) / 5.0, 1.5)

    score += sum(0.8 for term in HARD_TERMS if term in lower)
    score += sum(0.35 for term in MEDIUM_TERMS if term in lower)

    if question_type in {"Numerical", "Algorithm", "Compare"}:
        score += 0.5

    if score < 1.5:
        return "Easy", min(score / 3.0, 1.0)
    if score < 3.0:
        return "Medium", min(score / 4.0, 1.0)
    return "Hard", min(score / 5.0, 1.0)
