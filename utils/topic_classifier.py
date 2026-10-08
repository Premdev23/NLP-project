"""NLP syllabus topic classification for Mumbai University question papers."""

import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


TOPIC_KEYWORDS = {
    "NLP Foundations and Ambiguity": [
        "introduction to nlp", "natural language processing", "history of nlp",
        "stages of nlp", "language knowledge and grammar", "ambiguity",
        "lexical ambiguity", "syntactic ambiguity", "semantic ambiguity",
        "challenges of nlp", "regional language preprocessing"
    ],
    "Word-Level Analysis and Morphology": [
        "tokenization", "stemming", "lemmatization", "morphology",
        "inflectional morphology", "derivational morphology", "regular expression",
        "finite state morphology", "finite state transducer", "morphological parsing",
        "porter stemmer", "edit distance", "dictionary lookup", "word formation"
    ],
    "Language Models and N-grams": [
        "language model", "language modelling", "language modeling", "bigram",
        "trigram", "n gram", "n-gram", "unigram", "perplexity", "laplace smoothing",
        "good turing", "unknown words", "open vocabulary", "closed vocabulary",
        "noisy channel", "training corpus", "smoothing"
    ],
    "POS Tagging and Chunking": [
        "part of speech", "pos tagging", "pos tagger", "tag set", "penn treebank",
        "hidden markov model", "hmm", "viterbi", "maximum entropy model",
        "conditional random field", "crf", "transformation based tagging",
        "rule based tagging", "stochastic tagging", "chunking", "sequence labeling"
    ],
    "Parsing and Syntax": [
        "syntax analysis", "syntactic analysis", "parsing", "parser", "parse tree",
        "constituency", "context free grammar", "cfg", "cyk", "pcfg",
        "probabilistic context free grammar", "shift reduce", "top down parser",
        "bottom up parser", "predictive parser", "earley parser", "grammar rules"
    ],
    "Semantic Analysis and Word Sense": [
        "semantic analysis", "meaning representation", "lexical semantics", "wordnet",
        "babelnet", "homonymy", "polysemy", "synonymy", "hyponymy", "word sense",
        "word sense disambiguation", "wsd", "lesk algorithm", "yarowsky",
        "distributional semantics", "topic model", "semantic role"
    ],
    "Discourse and Pragmatics": [
        "pragmatics", "discourse processing", "discourse segmentation", "coherence",
        "reference resolution", "anaphora resolution", "coreference resolution",
        "hobbs algorithm", "centering algorithm", "discourse relation",
        "context resolution", "antecedent"
    ],
    "Information Retrieval": [
        "information retrieval", "document retrieval", "search engine", "query expansion",
        "precision and recall", "mean average precision", "map score", "ndcg",
        "ranking documents", "relevance feedback", "tf idf", "term frequency"
    ],
    "Information Extraction and Named Entity Recognition": [
        "information extraction", "named entity recognition", "ner",
        "entity extraction", "entity linking", "entity detection",
        "relation extraction", "extract entities"
    ],
    "Sentiment Analysis and Text Classification": [
        "sentiment analysis", "opinion mining", "text classification", "document classification",
        "naive bayes", "polarity classification", "subjectivity classification",
        "positive sentiment", "negative sentiment", "text categorization"
    ],
    "Machine Translation and Multilingual NLP": [
        "machine translation", "statistical machine translation", "rule based translation",
        "translation model", "source language", "target language", "multilingual nlp",
        "regional language", "indian languages", "cross lingual", "bleu score"
    ],
    "Summarization and Question Answering": [
        "text summarization", "text summary", "extractive summarization", "abstractive summarization",
        "question answering", "question answering system", "answer extraction",
        "reading comprehension", "information extraction", "named entity recognition",
        "ner", "entity extraction"
    ],
    "Neural NLP and Applications": [
        "deep neural network", "recurrent neural network", "rnn", "lstm", "neural network",
        "nlp application", "speech recognition", "chatbot", "dialogue system",
        "real world application", "text application"
    ],
}

TOPIC_MODULES = {
    "NLP Foundations and Ambiguity": "Module 1 — Introduction to NLP",
    "Word-Level Analysis and Morphology": "Module 2 — Word-Level Analysis",
    "Language Models and N-grams": "Module 2 — Word-Level Analysis",
    "POS Tagging and Chunking": "Module 3 — Syntax Analysis",
    "Parsing and Syntax": "Module 3 — Syntax Analysis",
    "Semantic Analysis and Word Sense": "Module 4 — Semantic Analysis",
    "Discourse and Pragmatics": "Module 5 — Pragmatics and Discourse",
    "Information Retrieval": "Module 6 — NLP Applications",
    "Information Extraction and Named Entity Recognition": "Module 6 — NLP Applications",
    "Sentiment Analysis and Text Classification": "Module 6 — NLP Applications",
    "Machine Translation and Multilingual NLP": "Module 6 — NLP Applications",
    "Summarization and Question Answering": "Module 6 — NLP Applications",
    "Neural NLP and Applications": "Module 6 — NLP Applications",
}


class TopicClassifier:
    """Explainable syllabus-aware classifier used when the ML model is unsure."""

    OTHER_TOPIC = "Other / Review"

    def __init__(self):
        self.topic_names = list(TOPIC_KEYWORDS)
        self.documents = [" ".join(TOPIC_KEYWORDS[t]) for t in self.topic_names]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self.matrix = self.vectorizer.fit_transform(self.documents)

    def predict(self, text: str):
        clean = str(text or "").lower()
        query = self.vectorizer.transform([clean])
        scores = cosine_similarity(query, self.matrix)[0]

        # Exact syllabus terms help distinguish related topics such as parsing,
        # word morphology, language models, and semantic analysis.
        for i, topic in enumerate(self.topic_names):
            hits = 0
            for keyword in TOPIC_KEYWORDS[topic]:
                if len(keyword) <= 3 and " " not in keyword:
                    matched = re.search(r"\b" + re.escape(keyword) + r"\b", clean)
                else:
                    matched = keyword in clean
                hits += bool(matched)
            scores[i] += min(hits * 0.12, 0.60)

        best = int(scores.argmax())
        confidence = float(min(scores[best], 1.0))
        if not clean.strip() or confidence < 0.12:
            return self.OTHER_TOPIC, confidence
        return self.topic_names[best], confidence
