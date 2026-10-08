# Project notes

## Course scope

The default taxonomy targets Mumbai University B.E. Computer Engineering Semester VII Natural Language Processing, REV-2019 'C' Scheme. Its syllabus areas are represented as the topic labels in `utils/topic_classifier.py` and the labelled examples in `data/sample_questions.csv`.

## Analysis pipeline

1. Read text from PDF/TXT files or OCR scanned PDF/image pages with Tesseract.
2. Split paper text into numbered questions and lettered sub-questions.
3. Normalize text, then suggest an NLP topic with TF-IDF/Logistic Regression and a syllabus keyword fallback.
4. Estimate question type, difficulty, Bloom level, and keywords.
5. Compare questions and summarize topic coverage across papers.

The default training examples are illustrative and hand-written. They are not a verified Mumbai University question bank. Topic suggestions and scores should be reviewed; difficulty, Bloom level, and OCR are heuristic or scan-dependent.

## Improving topic predictions

Add actual question/sub-question text and its reviewed syllabus topic to `data/sample_questions.csv`. Keep the existing five columns and choose the topic label from `utils/topic_classifier.py`. Include examples from multiple years and every module. Restart the app after changing the dataset. Do not report the seed-model predictions as validated accuracy.

## No external LLM required

The project uses local scikit-learn models, keyword rules, and text similarity. It does not send question papers to an LLM service.
