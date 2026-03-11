# Copyright 2025 FIZ-Karlsruhe (Mustafa Sofean)

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.types import Send

from src.agents.prompt import get_patent_reflection_prompt
from src.agents.state import DeepSearchState, QueryGenerationState, ReflectionState

from typing import List
from pydantic import BaseModel, Field

import os
from dotenv import load_dotenv
import openai

load_dotenv()
gemini_api_key = os.getenv('GOOGLE_API_KEY')
gemini_model = os.getenv('GEMINI_API_MODEL')
os.environ["GOOGLE_API_KEY"] = gemini_api_key

openai.api_key = os.getenv('OPENAI_API_KEY')
openai_model = os.getenv('OPENAI_API_MODEL')


class Reflection(BaseModel):
    is_sufficient: bool = Field(
        description="Whether the provided summaries are sufficient to answer the user's question."
    )
    knowledge_gap: str = Field(
        description="A description of what information is missing or needs clarification."
    )
    follow_up_queries: List[str] = Field(
        description="A list of follow-up queries to address the knowledge gap."
    )


def patent_deep_reflection(state: DeepSearchState) -> ReflectionState:
    """
    LangGraph reflection node for the deep research workflow.
    Evaluates gathered research results and determines whether additional
    search queries are needed or if the research process can terminate.
    :param state:
    :return:
    """
    if 'gpt' in state.llm:
        return patent_deep_reflection_by_openai(state)
    elif 'gemini' in state.llm:
        return patent_deep_reflection_by_gemini(state)


def patent_deep_reflection_by_gemini(state: DeepSearchState) -> DeepSearchState:
    """
    LangGraph reflection node for the deep research workflow.
    :param state:
    :return:
    """

    patent_reflection_prompt = get_patent_reflection_prompt(research_topic=state.research_topic,
                                                            patent_research_results=state.patent_research_results)
    llm_model = ChatGoogleGenerativeAI(
        model=gemini_model,
        temperature=0.5,
        max_retries=2,
        api_key=gemini_api_key)

    result = llm_model.with_structured_output(Reflection).invoke(patent_reflection_prompt)

    return {
        "is_sufficient": result.is_sufficient,
        "knowledge_gap": result.knowledge_gap,
        "follow_up_query": result.follow_up_queries,
        "research_loop_count": state.research_loop_count + 1,
        "patent_search_query": ' '.join(result.follow_up_queries),
    }


def patent_deep_reflection_by_openai(state: DeepSearchState) -> DeepSearchState:
    """
    LangGraph reflection node for the deep research workflow.
    :param state:
    :return:
    """
    # topic = [msg.content for msg in state.research_topic if isinstance(msg, HumanMessage)][0]
    # state["research_loop_count"] = state.get("research_loop_count", 0) + 1
    patent_reflection_prompt = get_patent_reflection_prompt(research_topic=state.research_topic,
                                                            patent_research_results=state.patent_research_results)
    llm_model = ChatOpenAI(
        model_name=openai_model,
        temperature=0.5,
    )

    result = llm_model.with_structured_output(Reflection).invoke(patent_reflection_prompt)
    return {
        "is_sufficient": result.is_sufficient,
        "knowledge_gap": result.knowledge_gap,
        "follow_up_query": result.follow_up_queries,
        "research_loop_count": state.research_loop_count + 1,
        "patent_search_query": ' '.join(result.follow_up_queries),
    }


def continue_to_patent_research(state: QueryGenerationState):
    """A LangGraph node that forwards search queries to the patent research agent.
    This node is responsible for spawning n patent research nodes—one for each search query.
    """
    return [
        Send("patent_research", {"search_query": search_query, "id": int(idx)})
        for idx, search_query in enumerate(state["search_query"])
    ]
