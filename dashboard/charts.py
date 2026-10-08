
import plotly.express as px
import streamlit as st

from utils.topic_classifier import TOPIC_MODULES

def render_charts(df):
    c1, c2 = st.columns(2)

    module_rows = df[df["Syllabus Module"].isin(set(TOPIC_MODULES.values()))]
    modules = module_rows["Syllabus Module"].value_counts().reset_index()
    modules.columns = ["Syllabus Module", "Questions"]
    with c1:
        if modules.empty:
            st.info("No questions are assigned to one of the six syllabus modules yet.")
        else:
            st.plotly_chart(
                px.bar(
                    modules,
                    x="Syllabus Module",
                    y="Questions",
                    title="Mumbai University Syllabus Module Coverage"
                ),
                width="stretch"
            )

    topic = df["Topic"].value_counts().reset_index()
    topic.columns = ["Topic", "Questions"]
    with c2:
        st.plotly_chart(
            px.bar(topic, x="Topic", y="Questions", title="NLP Topic Distribution"),
            width="stretch"
        )

    difficulty = df["Difficulty"].value_counts().reset_index()
    difficulty.columns = ["Difficulty", "Questions"]
    st.plotly_chart(
        px.pie(
            difficulty,
            names="Difficulty",
            values="Questions",
            title="Difficulty Distribution"
        ),
        width="stretch"
    )

    c3, c4 = st.columns(2)

    with c3:
        bloom = df["Bloom Level"].value_counts().reset_index()
        bloom.columns = ["Bloom Level", "Questions"]
        st.plotly_chart(
            px.bar(
                bloom,
                x="Bloom Level",
                y="Questions",
                title="Bloom's Taxonomy"
            ),
            width="stretch"
        )

    with c4:
        qtypes = df["Question Type"].value_counts().reset_index()
        qtypes.columns = ["Question Type", "Questions"]
        st.plotly_chart(
            px.bar(
                qtypes,
                x="Question Type",
                y="Questions",
                title="Question Type Distribution"
            ),
            width="stretch"
        )

    # Cross-paper topic coverage if multiple papers were supplied
    if df["Paper"].nunique() > 1:
        coverage = (
            df.groupby(["Paper", "Topic"])
            .size()
            .reset_index(name="Questions")
        )
        st.plotly_chart(
            px.bar(
                coverage,
                x="Paper",
                y="Questions",
                color="Topic",
                barmode="stack",
                title="Topic Coverage by Paper"
            ),
            width="stretch"
        )
