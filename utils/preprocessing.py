
import re

def preprocess_text(text: str) -> str:
    """
    Beginner-friendly NLP preprocessing.
    Uses NLTK when its optional resources are available.
    Falls back to a lightweight regex tokenizer so the app still runs.
    """
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    try:
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer

        try:
            stops = set(stopwords.words("english"))
            lemmatizer = WordNetLemmatizer()
            tokens = nltk.word_tokenize(text)
            tokens = [
                lemmatizer.lemmatize(t)
                for t in tokens
                if t.isalnum() and t not in stops
            ]
            return " ".join(tokens)
        except LookupError:
            pass
    except ImportError:
        pass

    # Fallback: simple stopword list
    stop = {
        "the", "is", "are", "was", "were", "a", "an", "and", "or",
        "of", "to", "in", "on", "for", "with", "by", "from", "this",
        "that", "what", "how", "why", "explain", "describe", "define",
        "discuss", "using", "use", "give"
    }
    tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", text)
    return " ".join(t for t in tokens if t not in stop)
