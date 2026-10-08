import re

PATTERNS = [
    ("Numerical", r"\b(calculate|compute|find|solve|determine|probability|perplexity|numerical)\b"),
    ("Definition", r"\b(define|what is|what are|meaning of|definition)\b"),
    ("Compare", r"\b(compare|differentiate|distinguish|difference between|contrast)\b"),
    ("Advantages/Disadvantages", r"\b(advantages|disadvantages|pros and cons|merits|demerits)\b"),
    ("Algorithm", r"\b(algorithm|steps of|procedure|working of|viterbi|cyk|lesk|hobbs)\b"),
    ("Application", r"\b(application|use cases?|real[- ]world|where is.*used|design a|construct a)\b"),
    ("Explain", r"\b(explain|describe|discuss|elaborate)\b"),
]

def classify_question_type(text: str) -> str:
    lower = text.lower()
    for label, pattern in PATTERNS:
        if re.search(pattern, lower):
            return label
    return "Conceptual"
