# Copyright 2025 FIZ-Karlsruhe (Mustafa Sofean)

from dotenv import load_dotenv
import os

from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

import openai

from src.agents.prompt import get_patent_answer_prompt
from src.agents.state import DeepSearchState

load_dotenv()
if os.getenv("GOOGLE_API_KEY") is None:
    raise ValueError("GEMINI_API_KEY is not set")

gemini_api_key = os.getenv('GOOGLE_API_KEY')
gemini_model = os.getenv('GEMINI_API_MODEL')
os.environ["GOOGLE_API_KEY"] = gemini_api_key

openai.api_key = os.getenv('OPENAI_API_KEY')
openai_model = os.getenv('OPENAI_API_MODEL')


def finalize_answer(state: DeepSearchState):
    if 'gpt' in state.llm:
        return finalize_answer_by_openai(state)
    elif 'gemini' in state.llm:
        return finalize_answer_by_gemini(state)


def finalize_answer_by_gemini(state: DeepSearchState):
    """

    :param state:
    :return:
    """

    answer_prompt = get_patent_answer_prompt(patent_research_results=state.patent_research_results,
                                             research_topic=state.research_topic)
    llm_model = ChatGoogleGenerativeAI(
        model=gemini_model,
        temperature=0.6,
        max_retries=2,
        api_key=gemini_api_key)

    result = llm_model.invoke(answer_prompt)

    return {
        "research_topic": state.research_topic,
        "answer": result.content,
        "patent_sources_gathered": state.patent_sources_gathered
    }


def finalize_answer_by_openai(state: DeepSearchState):
    """
    AI agent that generates answer based on provided context.
    :param state: Current graph state
    :return: Dictionary with state update, including the question answer.
    """

    answer_prompt = get_patent_answer_prompt(patent_research_results=state.patent_research_results,
                                             research_topic=state.research_topic)

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", answer_prompt),
        ("user", state.research_topic),
    ])
    llm = ChatOpenAI(model_name=openai_model,
                     temperature=0.6,
                     )
    pipeline = prompt_template | llm
    response = pipeline.invoke({"query": state.research_topic, "context": state.patent_sources_gathered})

    return {
        "research_topic": state.research_topic,
        "answer": response,
        "patent_sources_gathered": state.patent_sources_gathered
    }
