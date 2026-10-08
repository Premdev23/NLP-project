
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from utils.preprocessing import preprocess_text
from utils.topic_classifier import TopicClassifier

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_questions.csv"

class AcademicMLModels:
    def __init__(self):
        self.topic_model_name = "NLP syllabus TF-IDF + Logistic Regression"
        self.training_samples = 0
        self.topic_model = None
        self.fallback = TopicClassifier()
        self._train_topic_model()

    def _train_topic_model(self):
        data = pd.read_csv(DATA_PATH).dropna(subset=["question", "topic"])
        data["processed"] = data["question"].map(preprocess_text)
        data = data.dropna(subset=["processed", "topic"])

        self.training_samples = len(data)

        # The bundled examples seed a subject-specific model; they are not an
        # independently evaluated or official past-paper training set.
        X = data["processed"]
        y = data["topic"]

        self.topic_model = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=1,
                sublinear_tf=True
            )),
            ("clf", LogisticRegression(
                max_iter=1500,
                class_weight="balanced"
            ))
        ])

        self.topic_model.fit(X, y)


    def predict_topic(self, text):
        text = str(text or "").strip()
        if not text:
            return self.fallback.OTHER_TOPIC, 0.0, "Needs review"

        fallback_topic, fallback_score = self.fallback.predict(text)
        if fallback_topic == self.fallback.OTHER_TOPIC:
            return fallback_topic, fallback_score, "Needs review"
        if fallback_score >= 0.35:
            return fallback_topic, fallback_score, "Syllabus keyword match"

        if not self.topic_model:
            return fallback_topic, fallback_score, "Syllabus keyword fallback"

        probabilities = self.topic_model.predict_proba([text])[0]
        classes = self.topic_model.classes_
        index = probabilities.argmax()
        topic = classes[index]
        confidence = float(probabilities[index])

        # Short question wording often makes the small seed model uncertain.
        # Prefer an explicit syllabus term match in that case.
        if confidence < 0.45:
            return fallback_topic, fallback_score, "Syllabus keyword fallback"

        return topic, confidence, "Supervised ML"
