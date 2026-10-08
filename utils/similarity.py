"""Question similarity and cross-paper repeat matching."""

import re
import unicodedata
from difflib import SequenceMatcher
from itertools import combinations

import numpy as np
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


_STEMMER = SnowballStemmer("english")
_STOP_WORDS = ENGLISH_STOP_WORDS.union({
    "answer", "calculate", "compare", "define", "describe", "determine",
    "discuss", "differentiate", "explain", "find", "following", "give",
    "illustrate", "list", "show", "state", "using", "write"
})


def _normalize_question(question):
    text = unicodedata.normalize("NFKD", str(question or ""))
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    text = re.sub(r"\b(?:q(?:uestion)?\s*\.?\s*)\d{1,2}(?:\s*\([a-z]\))?", " ", text)
    text = re.sub(r"\b\d+\s*(?:marks?|m)\b|\[\s*\d+\s*\]|\(\s*\d+\s*\)", " ", text)
    text = text.replace("part of speech", "pos")
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [
        _STEMMER.stem(token)
        for token in text.split()
        if token not in _STOP_WORDS and len(token) > 1
    ]
    return " ".join(tokens)


def _similarity_matrix(questions):
    """Combine term overlap, OCR-tolerant character overlap, and token order."""
    normalized = [_normalize_question(question) for question in questions]
    count = len(normalized)
    scores = np.zeros((count, count), dtype=float)
    if count < 2:
        return scores

    try:
        words = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        word_matrix = words.fit_transform(normalized)
        word_scores = cosine_similarity(word_matrix)
    except ValueError:
        word_scores = scores.copy()

    try:
        chars = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=1,
            sublinear_tf=True
        )
        char_matrix = chars.fit_transform(normalized)
        char_scores = cosine_similarity(char_matrix)
    except ValueError:
        char_scores = scores.copy()

    token_sets = [set(text.split()) for text in normalized]
    for i in range(count):
        for j in range(i + 1, count):
            if normalized[i] and normalized[i] == normalized[j]:
                score = 1.0
            else:
                union = token_sets[i] | token_sets[j]
                overlap = (
                    len(token_sets[i] & token_sets[j]) / len(union)
                    if union else 0.0
                )
                order = SequenceMatcher(
                    None, normalized[i], normalized[j], autojunk=False
                ).ratio()
                score = (
                    0.55 * float(word_scores[i, j])
                    + 0.20 * float(char_scores[i, j])
                    + 0.15 * order
                    + 0.10 * overlap
                )
            scores[i, j] = scores[j, i] = score

    np.fill_diagonal(scores, 1.0)
    return scores


def _paper_groups(records):
    groups = {}
    for index, row in enumerate(records):
        groups.setdefault(str(row.get("Paper", "")), []).append(index)
    return groups


def find_similar_questions(df, threshold=0.78):
    """Find similar question pairs within each uploaded paper."""
    records = df.to_dict(orient="records")
    if len(records) < 2:
        return []

    questions = [row.get("Question", "") for row in records]
    scores = _similarity_matrix(questions)
    pairs = []
    for indices in _paper_groups(records).values():
        for i, j in combinations(indices, 2):
            score = float(scores[i, j])
            if score >= threshold:
                pairs.append((records[i], records[j], score))
    return sorted(pairs, key=lambda pair: pair[2], reverse=True)


def find_cross_paper_repeats(df, threshold=0.78):
    """Find one-to-one best matches between each pair of different papers."""
    records = df.to_dict(orient="records")
    groups = list(_paper_groups(records).values())
    if len(groups) < 2:
        return []

    questions = [row.get("Question", "") for row in records]
    scores = _similarity_matrix(questions)
    results = []

    for indices_a, indices_b in combinations(groups, 2):
        candidates = sorted(
            (
                (float(scores[i, j]), i, j)
                for i in indices_a
                for j in indices_b
                if float(scores[i, j]) >= threshold
            ),
            reverse=True
        )
        matched_a = set()
        matched_b = set()
        for score, i, j in candidates:
            if i in matched_a or j in matched_b:
                continue
            results.append((records[i], records[j], score))
            matched_a.add(i)
            matched_b.add(j)

    return sorted(results, key=lambda pair: pair[2], reverse=True)
