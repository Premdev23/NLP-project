import re

BLOOM_PATTERNS = [
    ("Create", ["design", "create", "develop", "construct", "formulate", "propose"]),
    ("Evaluate", ["evaluate", "justify", "critique", "assess", "defend", "validate"]),
    ("Analyze", ["analyze", "analyse", "compare", "differentiate", "examine", "distinguish", "disambiguate"]),
    ("Apply", ["calculate", "solve", "implement", "demonstrate", "apply", "use", "parse", "tag", "classify", "retrieve"]),
    ("Understand", ["explain", "describe", "discuss", "summarize", "interpret", "outline"]),
    ("Remember", ["define", "list", "identify", "state", "name", "mention", "recall"]),
]

def predict_bloom_level(text: str):
    lower = text.lower()
    for level, verbs in BLOOM_PATTERNS:
        for verb in verbs:
            if re.search(r"\b" + re.escape(verb) + r"\b", lower):
                return level, verb
    return "Understand", ""
