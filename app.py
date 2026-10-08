
import io
import pandas as pd
import plotly.express as px
import streamlit as st

from utils.pdf_extractor import extract_pdf_text
from utils.ocr import extract_text_from_image
from utils.preprocessing import preprocess_text
from utils.question_extractor import extract_questions
from utils.paper_metadata import infer_paper_metadata
from utils.ml_models import AcademicMLModels, DATA_PATH
from utils.topic_classifier import TOPIC_KEYWORDS, TOPIC_MODULES, TopicClassifier
from utils.question_classifier import classify_question_type
from utils.difficulty import predict_difficulty
from utils.bloom import predict_bloom_level
from utils.keyword_extractor import extract_keywords
from utils.similarity import find_similar_questions, find_cross_paper_repeats
from dashboard.charts import render_charts

st.set_page_config(
    page_title="Mumbai University NLP Paper Analyzer",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Mumbai University NLP Question Paper Analyzer")
st.caption(
    "Built for B.E. Computer Engineering Semester VII NLP papers "
    "(REV-2019 'C' Scheme). Upload past papers to compare topics, marks, "
    "question patterns, and repetition across semesters."
)

@st.cache_resource
def load_models(training_data_version):
    return AcademicMLModels()

models = load_models(DATA_PATH.stat().st_mtime_ns)

def read_uploaded_file(uploaded):
    suffix = uploaded.name.lower().split(".")[-1]
    data = uploaded.getvalue()

    if suffix == "pdf":
        return extract_pdf_text(io.BytesIO(data), ocr_fallback=True)
    if suffix in {"png", "jpg", "jpeg"}:
        return extract_text_from_image(data)
    return data.decode("utf-8", errors="ignore")

def add_analysis(questions, paper_name, academic_year="", semester="", session=""):
    rows = []

    for q in questions:
        raw = q["question"]
        processed = preprocess_text(raw)

        # ML topic prediction, with rule-based fallback inside the model class.
        topic, topic_conf, topic_source = models.predict_topic(processed)

        qtype = classify_question_type(raw)
        difficulty, difficulty_score = predict_difficulty(
            raw, qtype, q.get("marks")
        )
        bloom, bloom_verb = predict_bloom_level(raw)
        keywords = extract_keywords(processed, top_n=8)

        rows.append({
            "Paper": paper_name,
            "Academic Year": academic_year,
            "Semester": semester,
            "Session": session,
            "Question No.": q["question_number"],
            "Question": raw,
            "Sub-question": q.get("sub_question", ""),
            "Marks": q.get("marks"),
            "Topic": topic,
            "Syllabus Module": TOPIC_MODULES.get(topic, "Review topic"),
            "Topic Score": round(topic_conf, 3),
            "Topic Method": topic_source,
            "Question Type": qtype,
            "Difficulty": difficulty,
            "Difficulty Score": round(difficulty_score, 3),
            "Bloom Level": bloom,
            "Bloom Verb": bloom_verb or "Not detected",
            "Keywords": ", ".join(keywords)
        })

    return rows


def _question_label(row):
    number = str(row.get("Question No.", "")).strip()
    sub_question = str(row.get("Sub-question", "")).strip()
    if sub_question and sub_question.lower() != "nan":
        return f"{number}({sub_question})"
    return number


# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.header("📥 Input")
    uploads = st.file_uploader(
        "Upload one or more question papers",
        type=["pdf", "png", "jpg", "jpeg", "txt"],
        accept_multiple_files=True
    )

    st.divider()
    st.header("⚙️ Analysis Settings")
    similarity_threshold = st.slider(
        "Question similarity threshold",
        0.60, 0.95, 0.78, 0.01,
        help="Raise this to show only closer wording matches. Lower it to include more possible repeats."
    )

    st.divider()
    st.header("🧠 NLP Topic Model")
    st.write(
        "Topics follow the Mumbai University NLP syllabus: word analysis, "
        "language models, syntax, semantics, discourse, and applications. "
        "Predicted labels are suggestions; review them against each question."
    )

    st.divider()
    st.header("🗓️ Paper Metadata")
    default_year = st.text_input("Default academic year", placeholder="e.g. 2024")
    default_semester = st.selectbox(
        "Default semester",
        ["", "Semester 1", "Semester 2", "Semester 3", "Semester 4",
         "Semester 5", "Semester 6", "Semester 7", "Semester 8"],
    )
    default_session = st.text_input("Default session / set", placeholder="e.g. Set 1")

    analyze = st.button(
        "🔍 Analyze Papers",
        type="primary",
        width="stretch"
    )

st.session_state["similarity_threshold"] = similarity_threshold

st.subheader("✍️ Or paste one question paper")
manual_text = st.text_area(
    "Paste paper text",
    height=180,
    placeholder=(
        "Q1. Define Natural Language Processing. [2]\n"
        "Q2. (a) Explain TF-IDF. [5]\n"
        "    (b) Compare TF-IDF and word embeddings. [5]\n"
        "Q3. Calculate precision and recall from the following values. [5]"
    )
)

if analyze:
    if not uploads and not manual_text.strip():
        st.error("Please upload at least one paper or paste paper text.")
        st.stop()

    all_rows = []

    # Uploaded papers
    for uploaded in uploads or []:
        try:
            text = read_uploaded_file(uploaded)
            questions = extract_questions(text)

            if not questions:
                st.warning(
                    f"Could not identify numbered questions in {uploaded.name}. "
                    "Check that the extracted text is readable and question "
                    "numbers were recognized. For scanned pages, check Tesseract OCR."
                )
                with st.expander(f"Extracted text preview: {uploaded.name}"):
                    st.text(text[:2500] if text.strip() else "No text was extracted from this file.")
                continue

            year, semester, session = infer_paper_metadata(uploaded.name)
            year = year or default_year
            semester = semester or default_semester
            session = session or default_session
            all_rows.extend(add_analysis(
                questions, uploaded.name, year, semester, session
            ))

        except Exception as e:
            st.error(f"Could not analyze {uploaded.name}: {e}")

    # Manual paper
    if manual_text.strip():
        questions = extract_questions(manual_text)
        if questions:
            all_rows.extend(add_analysis(
                questions, "Manual Input",
                default_year, default_semester, default_session
            ))
        else:
            st.warning("No questions detected in the pasted text.")

    if all_rows:
        st.session_state.pop("topic_review_editor", None)
        st.session_state["analysis_df"] = pd.DataFrame(all_rows)
        st.session_state["similarity_threshold"] = similarity_threshold
        st.success(
            f"Analysis complete — {len(all_rows)} questions detected "
            f"across {st.session_state['analysis_df']['Paper'].nunique()} paper(s)."
        )
    else:
        st.error("No questions could be extracted.")

df = st.session_state.get("analysis_df")

if df is not None and not df.empty:
    # ---------------------------
    # Overview
    # ---------------------------
    st.header("1. 📊 Overview")

    papers = df["Paper"].nunique()
    recognized_modules = df.loc[
        df["Syllabus Module"].isin(set(TOPIC_MODULES.values())),
        "Syllabus Module"
    ].nunique()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Papers", papers)
    c2.metric("Questions", len(df))
    c3.metric("Topics", df["Topic"].nunique())
    c4.metric("Syllabus Modules", recognized_modules)
    c5.metric("Repeated Pairs", len(find_cross_paper_repeats(
        df, st.session_state["similarity_threshold"]
    )))

    # ---------------------------
    # ML model information
    # ---------------------------
    with st.expander("🧠 Model Information"):
        st.write(f"Topic model: **{models.topic_model_name}**")
        st.write(
            f"The model starts from {models.training_samples} illustrative "
            "questions, not official past-paper data. Topic scores are model "
            "scores, not calibrated probabilities or validated accuracy."
        )

    # ---------------------------
    # Question table
    # ---------------------------
    st.header("2. 📝 Question-wise Analysis")

    st.caption(
        "Review the suggested Topic values. You can change a topic in the "
        "table; the charts and downloads will use your corrected labels."
    )
    df = st.data_editor(
        df,
        width="stretch",
        hide_index=True,
        key="topic_review_editor",
        disabled=[column for column in df.columns if column != "Topic"],
        column_config={
            "Topic": st.column_config.SelectboxColumn(
                "Topic",
                options=[*TOPIC_KEYWORDS, TopicClassifier.OTHER_TOPIC],
                required=True
            )
        }
    )
    df["Syllabus Module"] = df["Topic"].map(TOPIC_MODULES).fillna("Review topic")
    st.session_state["analysis_df"] = df

    # ---------------------------
    # Filters
    # ---------------------------
    st.header("3. 🔎 Filters")

    f1, f2, f3, f4, f5 = st.columns(5)

    def options(column):
        return ["All"] + sorted(df[column].dropna().astype(str).unique().tolist())

    selected_paper = f1.selectbox("Paper", options("Paper"))
    selected_topic = f2.selectbox("Topic", options("Topic"))
    selected_module = f3.selectbox("Syllabus Module", options("Syllabus Module"))
    selected_difficulty = f4.selectbox("Difficulty", options("Difficulty"))
    selected_bloom = f5.selectbox("Bloom Level", options("Bloom Level"))

    filtered = df.copy()

    for col, value in [
        ("Paper", selected_paper),
        ("Topic", selected_topic),
        ("Syllabus Module", selected_module),
        ("Difficulty", selected_difficulty),
        ("Bloom Level", selected_bloom)
    ]:
        if value != "All":
            filtered = filtered[filtered[col].astype(str) == value]

    st.caption(f"Showing {len(filtered)} of {len(df)} questions")
    st.dataframe(filtered, width="stretch", hide_index=True)

    # ---------------------------
    # Analytics
    # ---------------------------
    st.header("4. 📈 Analytics Dashboard")
    render_charts(df)

    # ---------------------------
    # Similarity within all papers
    # ---------------------------
    st.header("5. 🔄 Similar / Duplicate Questions Within Each Paper")

    same_or_all = find_similar_questions(
        df,
        threshold=st.session_state["similarity_threshold"]
    )

    if same_or_all:
        sim_rows = [
            {
                "Paper": question_a["Paper"],
                "Question A No.": _question_label(question_a),
                "Topic": question_a["Topic"],
                "Question A": question_a["Question"],
                "Question B No.": _question_label(question_b),
                "Question B": question_b["Question"],
                "Similarity": f"{score * 100:.1f}%"
            }
            for question_a, question_b, score in same_or_all
        ]
        st.dataframe(
            pd.DataFrame(sim_rows),
            width="stretch",
            hide_index=True
        )
    else:
        st.info(
            "No similar question pairs found within the same paper at this "
            "threshold. Lower the threshold to include weaker matches."
        )

    # ---------------------------
    # Cross-paper repeated questions
    # ---------------------------
    st.header("6. 📚 Repeated Questions Across Papers")

    repeated = find_cross_paper_repeats(
        df,
        threshold=st.session_state["similarity_threshold"]
    )

    if repeated:
        repeated_df = pd.DataFrame([
            {
                "Paper A": question_a["Paper"],
                "Question A No.": _question_label(question_a),
                "Topic A": question_a["Topic"],
                "Question A": question_a["Question"],
                "Paper B": question_b["Paper"],
                "Question B No.": _question_label(question_b),
                "Topic B": question_b["Topic"],
                "Question B": question_b["Question"],
                "Similarity": f"{score * 100:.1f}%"
            }
            for question_a, question_b, score in repeated
        ])
        st.dataframe(repeated_df, width="stretch", hide_index=True)
    else:
        if df["Paper"].nunique() < 2:
            st.info(
                "Upload two or more papers to identify repeated questions across years/semesters."
            )
        else:
            st.info(
                "No likely repeated questions were found at this threshold. "
                "Lower the threshold to include weaker matches. Standard exam "
                "instructions are excluded from matching."
            )

    # ---------------------------
    # Topic coverage
    # ---------------------------
    st.header("7. 🎯 Topic Coverage")

    topic_table = (
        df.groupby(["Paper", "Syllabus Module", "Topic"])
        .size()
        .reset_index(name="Questions")
        .sort_values(["Paper", "Questions"], ascending=[True, False])
    )
    st.dataframe(topic_table, width="stretch", hide_index=True)

    # ---------------------------
    # Semester / year comparison
    # ---------------------------
    st.header("8. 🗓️ Semester / Year Comparison")

    comparison_cols = [c for c in ["Academic Year", "Semester"] if c in df.columns]
    comparison_ready = df.copy()
    for col in comparison_cols:
        comparison_ready[col] = comparison_ready[col].fillna("").astype(str).replace("nan", "")

    if comparison_cols and comparison_ready[comparison_cols].replace("", pd.NA).notna().any().any():
        group_cols = comparison_cols
        comparison_table = (
            comparison_ready.groupby(group_cols, dropna=False)
            .agg(
                Questions=("Question", "count"),
                Total_Marks=("Marks", lambda s: pd.to_numeric(s, errors="coerce").sum()),
                Topics=("Topic", "nunique"),
                Modules=(
                    "Syllabus Module",
                    lambda values: values[values.isin(set(TOPIC_MODULES.values()))].nunique()
                ),
                Avg_Difficulty=("Difficulty Score", "mean")
            )
            .reset_index()
        )
        comparison_table["Avg_Difficulty"] = comparison_table["Avg_Difficulty"].round(2)
        st.dataframe(comparison_table, width="stretch", hide_index=True)

        if len(group_cols) == 2:
            coverage = (
                comparison_ready.groupby(group_cols + ["Topic"])
                .size()
                .reset_index(name="Questions")
            )
            st.plotly_chart(
                px.bar(
                    coverage,
                    x="Academic Year",
                    y="Questions",
                    color="Topic",
                    facet_col="Semester",
                    barmode="stack",
                    title="Topic Distribution by Academic Year and Semester"
                ),
                width="stretch"
            )
        else:
            coverage = (
                comparison_ready.groupby(group_cols + ["Topic"])
                .size()
                .reset_index(name="Questions")
            )
            st.plotly_chart(
                px.bar(
                    coverage,
                    x=group_cols[0],
                    y="Questions",
                    color="Topic",
                    barmode="stack",
                    title=f"Topic Distribution by {group_cols[0]}"
                ),
                width="stretch"
            )

        difficulty_year = (
            comparison_ready.groupby(group_cols + ["Difficulty"])
            .size()
            .reset_index(name="Questions")
        )
        st.plotly_chart(
            px.bar(
                difficulty_year,
                x=group_cols[0],
                y="Questions",
                color="Difficulty",
                barmode="group",
                title="Difficulty Mix Across Papers"
            ),
            width="stretch"
        )
    else:
        st.info(
            "Year/semester metadata is empty. Upload files named like "
            "'NLP_Sem7_2024.pdf' or fill the sidebar metadata fields."
        )

    # ---------------------------
    # Overall summary
    # ---------------------------
    st.header("9. 🧾 Overall Summary")

    top_topic = df["Topic"].value_counts().idxmax()
    classified_modules = df.loc[
        df["Syllabus Module"].isin(set(TOPIC_MODULES.values())),
        "Syllabus Module"
    ]
    top_module = (
        classified_modules.value_counts().idxmax()
        if not classified_modules.empty else "No classified module yet"
    )
    top_diff = df["Difficulty"].value_counts().idxmax()
    top_bloom = df["Bloom Level"].value_counts().idxmax()

    st.info(
        f"The analyzer found **{len(df)} questions** across **{papers} paper(s)**. "
        f"The most frequent syllabus module is **{top_module}**; "
        f"the most frequent topic is **{top_topic}**. "
        f"The dominant difficulty is **{top_diff}**, while the dominant "
        f"Bloom's Taxonomy level is **{top_bloom}**."
    )

    # ---------------------------
    # Downloads
    # ---------------------------
    st.header("10. ⬇️ Download Reports")

    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV",
        csv_data,
        "mumbai_university_nlp_analysis.csv",
        "text/csv"
    )

    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Analysis")
        topic_table.to_excel(writer, index=False, sheet_name="Topic Coverage")
        if repeated:
            pd.DataFrame([
                {
                    "Paper A": question_a["Paper"],
                    "Question A No.": _question_label(question_a),
                    "Topic A": question_a["Topic"],
                    "Question A": question_a["Question"],
                    "Paper B": question_b["Paper"],
                    "Question B No.": _question_label(question_b),
                    "Topic B": question_b["Topic"],
                    "Question B": question_b["Question"],
                    "Similarity": score
                }
                for question_a, question_b, score in repeated
            ]).to_excel(writer, index=False, sheet_name="Repeated Questions")

    st.download_button(
        "Download Excel Report",
        excel_buffer.getvalue(),
        "mumbai_university_nlp_analysis.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
else:
    st.info(
        "Upload one or more question papers or paste text, then click "
        "**Analyze Papers**."
    )
