# Copyright 2025 FIZ-Karlsruhe (Mustafa Sofean)

import os
import time
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from src.agents.prompt import get_summary_prompt_template
from src.agents.state import DeepSearchState

load_dotenv()

import openai


import google.generativeai as genai

gemini_api_key = os.getenv('GOOGLE_API_KEY')
gemini_model = os.getenv('GEMINI_API_MODEL')
os.environ["GOOGLE_API_KEY"] = gemini_api_key

openai.api_key = os.getenv('OPENAI_API_KEY')
openai_model = os.getenv('OPENAI_API_MODEL')


def patent_summary_agent(ti: str,
                         ab: str,
                         detd: str,
                         clms: str,
                         model: str):
    """
    LangGraph node that generates summaries for selected patents.
    Takes the most relevant patents and produces concise summaries
    :param ti:
    :param ab:
    :param detd:
    :param clms:
    :param model:
    :return:
    """
    if 'gpt' in model:
        return patent_summary_agent_by_openai(ti, ab, detd, clms)
    elif 'gemini' in model:
        return patent_summary_agent_by_gemini(ti, ab, detd, clms)


def patent_summary_agent_by_openai(ti: str,
                                   ab: str,
                                   detd: str,
                                   clms: str):
    """
    Summarize the patent document by Openai model
    :param ti:
    :param ab:
    :param detd:
    :param clms:
    :return:
    """

    context = f"""
               Title:  {ti}
               Abstract: {ab}
               Description: {detd}
               Claims: {clms}
               """
    context = str(context)
    summary_prompt_template = get_summary_prompt_template(context=context)
    prompt_template = PromptTemplate(
        input_variables=["context"],
        template=summary_prompt_template
    )
    llm_model = ChatOpenAI(
        model_name=openai_model,
        temperature=0.1,
    )
    summary_chain = prompt_template | llm_model
    result = summary_chain.invoke({"context": context})

    time.sleep(10)
    return result.content.replace('\n', '')


def patent_summary_agent_by_gemini(ti: str,
                                   ab: str,
                                   detd: str,
                                   clms: str):
    """
    Summarize patent by Gemini model
    :param ti:
    :param ab:
    :param detd:
    :param clms:
    :return:
    """
    os.environ['HTTP_PROXY'] = 'http://proxy.fiz-karlsruhe.de:8888'
    os.environ['HTTPS_PROXY'] = 'http://proxy.fiz-karlsruhe.de:8888'

    context = f"""
               Title:  {ti}
               Abstract: {ab}
               Description: {detd}
               Claims: {clms}
               """
    summary_prompt_template = get_summary_prompt_template(context=context)
    model = genai.GenerativeModel(gemini_model,
                                  )
    response = model.generate_content(summary_prompt_template,
                                      generation_config=genai.types.GenerationConfig(
                                          candidate_count=1,
                                          top_p=0.6,
                                          top_k=5,
                                          temperature=0.2)
                                      )
    time.sleep(10)

    return response.text


def article_summary_agent_by_gemini(ti: str,
                                    ab: str,
                                    body: str == ""
                                    ):
    """
    summarize paten with GPT model
    :param ti:
    :param ab:
    :param body:
    :return:
    """
    os.environ['HTTP_PROXY'] = 'http://proxy.fiz-karlsruhe.de:8888'
    os.environ['HTTPS_PROXY'] = 'http://proxy.fiz-karlsruhe.de:8888'

    context = f"""
               Title:  {ti}
               Abstract: {ab}
               Body: {body}
               """
    summary_prompt_template = f"""
        Summarize the following research article's title and abstract in a concise and informative paragraph. 
        Focus on the main objective, methods, key findings, and significance.\n 
        Use clear and accessible language suitable for a general academic audience.\n
        Instructions:
            - Keep it concise.
            - Do not hallucinate.
            - Do not include any irrelevant information.

        Title:  '''{ti}'''\n
        Abstract: '''{ab}''' \n
        Body Text: '''{body}'''\n
        
        CONCISE SUMMARY:
        """
    model = genai.GenerativeModel(gemini_model,
                                  )
    response = model.generate_content(summary_prompt_template,
                                      generation_config=genai.types.GenerationConfig(
                                          candidate_count=1,
                                          top_p=0.4,
                                          top_k=4,
                                          temperature=0.2)
                                      )
    time.sleep(10)

    return response.text


def summarize_patent_summary(state: DeepSearchState):
    """LangGraph node that summarizes patent research results.

    Uses an LLM to create or update a running summary based on the newest patent research
    results, integrating them with any existing summary.
    Args:
        state: Current graph state containing research topic, running summary,
              and patent research results
    Returns:
        Dictionary with state update, including running_summary key containing the updated summary
    """
    topic = [msg.content for msg in state.research_topic if isinstance(msg, HumanMessage)][0]
    # Existing summary
    existing_summary = state.patent_running_summary

    # Most recent web research
    most_recent_web_research = state.web_research_results[-1]

    # Build the human message
    if existing_summary:
        human_message_content = (
            f"<Existing Summary> \n {existing_summary} \n <Existing Summary>\n\n"
            f"<New Context> \n {most_recent_web_research} \n <New Context>"
            f"Update the Existing Summary with the New Context on this topic: \n <User Input> \n {topic} \n <User Input>\n\n"
        )
    else:
        human_message_content = (
            f"<Context> \n {most_recent_web_research} \n <Context>"
            f"Create accurate Summary using the Context on this topic: \n <User Input> \n {topic} \n <User Input>\n\n"
        )

    model = genai.GenerativeModel(gemini_model,
                                  )
    response = model.generate_content(human_message_content,
                                      generation_config=genai.types.GenerationConfig(
                                          candidate_count=1,
                                          top_p=0.5,
                                          top_k=5,
                                          temperature=0.1)
                                      )
    running_summary = response.text

    return {"running_summary": running_summary}

