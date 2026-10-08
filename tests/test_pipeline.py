from utils.question_extractor import extract_questions
from utils.question_classifier import classify_question_type
from utils.bloom import predict_bloom_level

def test_main_and_subquestion_extraction():
    text = """
    Q1. Define NLP. [2]
    Q2. (a) Explain TF-IDF. [5]
        (b) Compare TF-IDF and embeddings. [5]
    """
    result = extract_questions(text)
    assert len(result) == 3
    assert result[0]["marks"] == 2
    assert result[1]["sub_question"] == "a"
    assert result[2]["sub_question"] == "b"

def test_question_type():
    assert classify_question_type("Define NLP") == "Definition"
    assert classify_question_type("Compare TCP and UDP") == "Compare"

def test_bloom():
    level, verb = predict_bloom_level("Calculate the probability.")
    assert level == "Apply"
    assert verb == "calculate"


def test_filename_metadata():
    from utils.paper_metadata import infer_paper_metadata
    year, semester, session = infer_paper_metadata("NLP_Sem7_2024_Set1.pdf")
    assert year == "2024"
    assert semester == "Semester 7"
    assert session == "Set 1"


def test_scanned_pdf_ocr_fallback():
    """The supplied scanned paper should produce text through the PDF OCR path."""
    from pathlib import Path
    from utils.pdf_extractor import extract_pdf_text

    pdf = Path(__file__).resolve().parents[2].parent / "be_computer-engineering_semester-7_2025_may_dloc-iii-natural-language-processing-rev-2019-c-scheme_copy.pdf"
    if not pdf.exists():
        # The source PDF is supplied separately during development; skip this
        # integration test when it is not present in the project workspace.
        return

    with pdf.open("rb") as fh:
        text = extract_pdf_text(fh, ocr_fallback=True, ocr_dpi=120)

    assert len(text.strip()) > 500
    assert "NLP" in text or "Natural Language" in text
