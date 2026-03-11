# Copyright 2025 FIZ-Karlsruhe (Mustafa Sofean)

import streamlit as st
import markdown
import json
import sys

sys.path.append("..")

from src.agents.main import run_deep_research, run_chat_deep_research

st.set_page_config(page_title="Agentic AI for Deep Research on Patents", page_icon="🐈", layout="wide")
st.title('Patent Deep Research')
form = st.form(key="Form1")
with st.form(key='columns_in_form'):
    text_query = st.text_area("Enter your topic here:", value="")
    c1, c2 = st.columns(2)

    with c1:
        deep_research_task = st.selectbox(
            'Deep Research Task:', ('Scientific Report', 'QA')
        )
    with c2:
        model_name = st.selectbox(
            'LLM:', ('gemini', 'gpt')
        )

    search_buttom = st.form_submit_button(label='Search')

if search_buttom and text_query.strip():
    SYSTEM_PROMPT = """You will act as a patent expert for analysing patents and perform a deep research"""
    USER_PROMPT = text_query.strip()

    if deep_research_task == "Scientific Report":
        results = run_deep_research(SYSTEM_PROMPT, USER_PROMPT, model_name)
        output = results['patent_deep_review']["patent_running_summary"]
        output_parts = output.split("## Sources:")
        report = output_parts[0].strip()
        sources = output_parts[1].strip().replace("\n", "") if len(output_parts) > 1 else ""
        sources = (sources.replace("[", "").replace("]", "")
                   .replace("#", "<br>")
                   .replace("'", ""))
        sources = f"""  {sources} """

        report = report.replace("```text ", "").replace("```", "").strip().removeprefix("text ").strip()
        readme_html = markdown.markdown(report)
        st.markdown(readme_html, unsafe_allow_html=True)

        # st.markdown(report, unsafe_allow_html=True)
        st.write("\n\n <b> Contexts:</b>\n" + sources.replace('\n', '<br>'), unsafe_allow_html=True)
        report_und_context = report + "\n\n" + sources
        # st.code(report_und_context, language="markdown")

        file_name = USER_PROMPT + "_REPORT.md"
        # Create a download button
        st.download_button(
            label="📥 " + file_name,
            data=report_und_context,
            file_name=file_name,
            mime="text/markdown"
        )

    else:
        if deep_research_task == "QA":
            results = run_chat_deep_research(SYSTEM_PROMPT, USER_PROMPT, model_name)
            output = results['finalize_answer']["answer"]
            contexts = results['finalize_answer']["patent_sources_gathered"]

            output_cleaned = output.replace("```json", "").replace("```", "")
            output_json = json.loads(output_cleaned)

            st.markdown("<br> <b> Answer:</b> <br>", unsafe_allow_html=True)
            st.markdown(
                "\n\n" + f"<div style='max-height:600px; overflow-y:auto;'>\n" + str(output_json["answer"]) + "</div>",
                unsafe_allow_html=True)
            st.markdown("\n\n <br> <b> Sources:</b> <br> \n" + "<br>".join(output_json["sources"]),
                        unsafe_allow_html=True)

            #sources = (contexts.replace("[", "").replace("]", "")
                       #.replace("#", "<br>")
                       #.replace("'", ""))
            #sources = f"""  {sources} """

            #st.write("\n\n <b> Contexts:</b>\n" + sources.replace('\n', '<br>'), unsafe_allow_html=True)