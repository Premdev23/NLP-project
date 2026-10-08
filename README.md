# Mumbai University NLP Question Paper Analyzer

A Streamlit tool for analyzing Mumbai University B.E. Computer Engineering Semester VII Natural Language Processing papers. The topic labels follow the REV-2019 'C' Scheme subject outline: NLP foundations, word-level analysis, language models, POS tagging, parsing, semantics, discourse, and applications.

The syllabus taxonomy is based on the [Mumbai University NLP course outline](https://www.munotes.in/syllabus/BE-Computer-Engineering/Fourth-Year-CE/Semester-7/Natural-Language-Processing/), reproduced there for the REV-2019 'C' Scheme.

## Run the app

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Upload one or more PDF, text, or image papers, or paste extracted paper text. Use filenames such as `NLP_Sem7_2024_May.pdf` so the analyzer can infer semester and year. Add default metadata in the sidebar when a filename does not include it.

For scanned PDFs and images, install Tesseract OCR separately. The app checks PATH and common Windows install folders. If Tesseract is installed elsewhere, set its path before starting Streamlit:

```powershell
$env:TESSERACT_CMD = "C:\Program Files\Tesseract-OCR\tesseract.exe"
streamlit run app.py
```

## What it analyzes

- Extracts numbered main questions and lettered sub-questions, including common `Q.1(a)` layouts
- Extracts selectable PDF text and uses OCR for full-page scans or pages whose text layer has no readable question numbering
- Suggests syllabus topic, question type, estimated difficulty, Bloom level, and keywords
- Summarizes question coverage by both syllabus module and topic
- Finds close question matches within each paper and likely repeats between papers, with paper and question-number references
- Compares marks, difficulty, and topic coverage across uploaded papers
- Exports results as CSV or Excel

The topic areas are:

- NLP Foundations and Ambiguity
- Word-Level Analysis and Morphology
- Language Models and N-grams
- POS Tagging and Chunking
- Parsing and Syntax
- Semantic Analysis and Word Sense
- Discourse and Pragmatics
- Information Retrieval
- Information Extraction and Named Entity Recognition
- Sentiment Analysis and Text Classification
- Machine Translation and Multilingual NLP
- Summarization and Question Answering
- Neural NLP and Applications

## Training data and limits

`data/sample_questions.csv` contains illustrative, hand-written example questions used to seed the topic model. They are **not** official Mumbai University questions, past-paper extracts, or a validated training corpus. The model's predicted topic and score are suggestions; check the editable topic labels against the actual question. Difficulty and Bloom level are heuristic estimates. OCR and question splitting can also need correction for faint scans or unusual page layouts.

Similarity results compare normalized wording and tolerate some OCR and word-form differences. Treat them as candidates to review, not proof that two differently worded questions have the same meaning.

For better topic predictions on your paper set, add manually labelled questions from your own Mumbai University papers to `data/sample_questions.csv`. Keep its columns and use one of the topic labels listed above. Use one question or sub-question per row. Restart the app after editing the file. Build the set from multiple papers and modules, and review the topic label in the app before using the charts as study guidance.

This setup assumes the B.E. Computer Engineering Semester VII REV-2019 NLP course. If your branch, semester, or syllabus revision differs, adjust the topic labels and keyword profiles in `utils/topic_classifier.py` before interpreting module coverage.

## Optional NLTK resources

The app has a built-in preprocessing fallback. For full NLTK tokenization and lemmatization, run:

```powershell
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```
