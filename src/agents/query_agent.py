# Copyright 2025 FIZ-Karlsruhe (Mustafa Sofean)

import re
import string
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from pydantic import BaseModel, Field
from typing import List

import openai

from src.agents.prompt import get_patent_query_prompt_template, get_patent_queries_prompt_template
from src.agents.state import DeepSearchState
from langchain_core.prompts import PromptTemplate

load_dotenv()

import google.generativeai as genai

gemini_api_key = os.getenv('GOOGLE_API_KEY')
gemini_model = os.getenv('GEMINI_API_MODEL')
os.environ["GOOGLE_API_KEY"] = gemini_api_key

openai.api_key = os.getenv('OPENAI_API_KEY')
openai_model = os.getenv('OPENAI_API_MODEL')


class SearchQueryList(BaseModel):
    query: List[str] = Field(
        description="A list of search queries to be used for patent research."
    )
    rationale: str = Field(
        description="A brief explanation of why these queries are relevant to the research topic."
    )

def planning_deep_research_agent(state: DeepSearchState,
                                 model: str = "gpt"):
    """
    LangGraph node responsible for planning the deep research process.
    :param state: Current graph state
    :param model:
    :return:
    """
    topic = [msg.content for msg in state.research_topic if isinstance(msg, HumanMessage)][0]

    if 'gpt' in model:
        return {"patent_search_query": create_query_by_openai(topic)}
    elif 'gemini' in model:
        return {"patent_search_query": create_query_by_gemini(topic)}


def planning_chat_deep_research_agent(state: DeepSearchState,
                                      model: str = "gemini"):
    """
    LangGraph node responsible for planning the agentic QA process.
    :param state: Current graph state.
    :param model:
    :return:
    """
    question = [msg.content for msg in state.research_topic if isinstance(msg, HumanMessage)][0]

    if 'gemini' in model:
        return create_queries_by_gemini(question)


def create_query_by_openai(research_topic: str):
    """
    create a search query as a text
    :param research_topic:
    :return:
    """

    patent_query_prompt_template = get_patent_query_prompt_template(research_topic=research_topic)

    prompt_template = PromptTemplate(
        input_variables=["research_topic"],
        template=patent_query_prompt_template
    )
    llm_model = ChatOpenAI(
        model_name=openai_model,
        temperature=0.1
    )
    output_parser = StrOutputParser()
    chain = prompt_template | llm_model | output_parser
    response = chain.invoke({"research_topic": research_topic})
    translator = str.maketrans('', '', string.punctuation)
    response = response.translate(translator)
    response = research_topic + ". " + response

    return response


def create_query_by_gemini(research_topic: str):
    """
    :param research_topic:
    :return:
    """

    patent_query_prompt_template = get_patent_query_prompt_template(research_topic=research_topic)
    model = genai.GenerativeModel(gemini_model,
                                  )
    response = model.generate_content(patent_query_prompt_template,
                                      generation_config=genai.types.GenerationConfig(
                                          candidate_count=1,
                                          top_p=0.6,
                                          top_k=5,
                                          temperature=0.1)
                                      )
    translator = str.maketrans('', '', string.punctuation)
    response = response.text.translate(translator)
    response = research_topic + ". " + response

    return response


def create_queries_by_gemini(research_topic: str):
    """
    Create a search query for user input
    :param research_topic:
    :return:
    """

    patent_query_prompt_template = get_patent_queries_prompt_template(research_topic=research_topic)

    llm_model = ChatGoogleGenerativeAI(
        model=gemini_model,
        temperature=0.9,
        max_retries=2,
        api_key=gemini_api_key)

    result = llm_model.with_structured_output(SearchQueryList).invoke(patent_query_prompt_template)

    return {"search_queries": result.query}


def create_queries_by_openai(research_topic: str):
    """
    Create a search query for user input
    :param research_topic:
    :return:
    """

    patent_query_prompt_template = get_patent_queries_prompt_template(question=research_topic)

    prompt_template = PromptTemplate(
        input_variables=["research_topic"],
        template=patent_query_prompt_template
    )

    llm_model = ChatOpenAI(
        model_name=openai_model,
        temperature=0.1,
    )
    query_chain = prompt_template | llm_model | StrOutputParser()
    result = query_chain.invoke({"research_topic": research_topic})

    return result.content.replace('\n', '')


