
import re

MAIN_Q = re.compile(
    r"(?im)^\s*(?:Q(?:uestion(?:\s+No\.?)?)?\s*\.?\s*)?"
    r"(\d{1,2})\s*(?:[\.:\)\-]\s*|\s+|(?=\([a-z]\)))(?=\S)"
)
SUB_Q = re.compile(
    r"(?im)(?:^|\s)(?:\(([a-z])\)|([a-z])[\.\)])\s*"
)

def parse_marks(text):
    patterns = [
        r"\[\s*(\d+)\s*(?:marks?|m)?\s*\]",
        r"\(\s*(\d+)\s*(?:marks?|m)?\s*\)\s*$",
        r"\b(?:for|of|marks?:?)\s*(\d+)\s*marks?\b",
        r"\b(\d+)\s*M(?:arks?)?\b",
        r"\bmarks?\s*[:=]\s*(\d+)\b"
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.I)
        if m:
            return int(m.group(1))
    return None

def strip_marks(text):
    return re.sub(
        r"\s*(?:\[\s*\d+\s*(?:marks?|m)?\s*\]|\(\s*\d+\s*(?:marks?|m)?\s*\)|\d+\s*marks?|marks?\s*[:=]\s*\d+)\s*$",
        "",
        text,
        flags=re.I
    ).strip()

def extract_subquestions(body):
    matches = list(SUB_Q.finditer(body))
    if not matches:
        return [("", body.strip(), parse_marks(body))]

    results = []
    prefix = body[:matches[0].start()].strip()
    if prefix:
        results.append(("", prefix, parse_marks(prefix)))

    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        part = re.sub(r"\s+", " ", body[start:end]).strip()
        sub_question = (match.group(1) or match.group(2)).lower()
        results.append((sub_question, strip_marks(part), parse_marks(part)))

    return results

def extract_questions(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    matches = list(MAIN_Q.finditer(text))
    questions = []

    if not matches:
        # fallback for simple numbered lists
        matches = list(re.finditer(r"(?im)^\s*(\d+)[\.\)]\s+", text))

    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = re.sub(r"\s+", " ", text[start:end]).strip()

        number = match.group(1)
        subparts = extract_subquestions(body)

        for sub_letter, question, marks in subparts:
            if not question:
                continue

            questions.append({
                "question_number": f"Q{number}",
                "sub_question": sub_letter,
                "question": strip_marks(question),
                "marks": marks
            })

    return questions
