# Copyright 2025 FIZ-Karlsruhe (Mustafa Sofean)

import json
import re

import os
from dotenv import load_dotenv
load_dotenv()
from langchain_core.messages import HumanMessage

from src.agents.reranker_agent import rerank_by_gemini, patent_reranker
from src.agents.state import DeepSearchState

from src.agents.summarization_agent import article_summary_agent_by_gemini, patent_summary_agent
from src.retrieval.arxiv_retrieval import get_articles
from src.retrieval.patent_retrieval import search_patent_doc, search_patent_passage

from src.utils.utils import patent_search_results_to_str, patent_format_sources, article_search_results_to_str, \
    article_format_sources, passage_format_sources

gemini_api_key = os.getenv('GOOGLE_API_KEY')
gemini_model = os.getenv('GEMINI_API_MODEL')
os.environ["GOOGLE_API_KEY"] = gemini_api_key

vespa_doc_schema_name = os.getenv('VESPA_DOC_SCHEMA_NAME')
vespa_passage_schema_name = os.getenv('VESPA_PASSAGE_SCHEMA_NAME')


def patent_search_agent(
        search_query: str,
        schema_name: str,
        hits: int,
        model: str = 'gpt'):
    """Retrieves the patent documents for a research topic.

    Args:
        query (str):
        schema_name (str): The vespa index schema
        hits (int): number of hits to return
        :param model:
    """
    search_response = search_patent_doc(query=search_query, schema_name=schema_name, hits=hits)
    search_response = json.loads(search_response)

    response = []
    for entry in search_response["data"]:
        pn = entry.get("Patent No", "")
        title = entry.get("Title", "")
        abstract = entry.get("Abstract", "")
        description = entry.get("Description", "")
        claims = entry.get("Claims", "")

        # doc_relevant_score = rerank(topic=research_topic, doc=title+" "+abstract)
        doc_summary = patent_summary_agent(title, abstract, str(description), str(claims), model)
        doc = {"patent number": pn, "title": title, "summary": doc_summary}

        # re-rank the summary with the topic, and store only the relevant ones
        relevance_score = patent_reranker(topic=search_query, doc=title + " " + doc_summary, model=model)
        if relevance_score.isdigit():
            if int(relevance_score) > 2:
                response.append(doc)

    return {"retrieved_patents": response}


def patent_passage_search_agent(
        search_query: str,
        schema_name: str,
        hits: int,
        model: str = 'gemini'):
    """Retrieves the passages for a research topic.

    Args:
        query (str):
        schema_name (str): The vespa index schema
        hits (int): number of hits to return
        :param model:
    """
    search_response = search_patent_passage(query=search_query, schema_name=schema_name, hits=hits)
    search_response = json.loads(search_response)

    response = []
    for entry in search_response["data"]:
        pn = entry.get("Patent No", "")
        passage = entry.get("PASSAGE", "")

        doc = {"patent number": pn, "summary": passage}

        # re-rank the summary with the topic, and store only the relevant ones
        relevance_score = patent_reranker(topic=search_query, doc=passage, model=model)
        if relevance_score.isdigit():
            if int(relevance_score) > 2:
                response.append(doc)

    return {"retrieved_patents": response}


def article_search_agent(
        research_topic: str,
        hits: int):
    """Retrieves the paper documents for a research topic.

    Args:
        research_topic (str):
        hits (int): number of hits to return
    """

    search_response = get_articles(research_topic, topn=hits)
    response = []
    for entry in search_response["retrieved_papers"]:
        url = entry.get("url", "")
        title = entry.get("Title", "")
        abstract = entry.get("Abstract", "")
        doc_summary = article_summary_agent_by_gemini(title, abstract, None)
        doc = {"url": url, "title": title, "summary": doc_summary}

        response.append(doc)

    return {"retrieved_articles": response}


def patent_search(state: DeepSearchState):
    """
    LangGraph node that performs search within patents.
    :param state: Current graph state
    :return:
    """
    topic = [msg.content for msg in state.research_topic if isinstance(msg, HumanMessage)][0]
    search_query = state.patent_search_query
    #print("##############", state.patent_search_query, "#########################")
    research_results = patent_search_agent(search_query=search_query, schema_name= vespa_doc_schema_name, hits=10)
    research_results_str = patent_search_results_to_str(research_results)

    return {"patent_sources_gathered": [patent_format_sources(research_results)],
            "patent_research_results": [research_results_str]}


def article_research(state: DeepSearchState):
    """
    LangGraph node that performs article search
    :param state:
    :return:
    """
    research_results = article_search_agent(state.article_search_query, hits=20)
    research_results_str = article_search_results_to_str(research_results)

    return {"article_sources_gathered": [article_format_sources(research_results)],
            "article_research_results": [research_results_str]}


def patent_passage_search(state: DeepSearchState):
    """
    LangGraph node that performs passage search
    :param state: Current graph state
    :return:
    """
    research_results = patent_passage_search_agent(search_query=state.patent_search_query, schema_name= vespa_passage_schema_name, hits=10)
    research_results_str = patent_search_results_to_str(research_results)

    return {"patent_sources_gathered": [passage_format_sources(research_results)],
            "patent_research_results": [research_results_str]}


def article_research(state: DeepSearchState):
    """
    performs article research
    :param state:
    :return:
    """
    research_results = article_search_agent(state.article_search_query, hits=20)
    research_results_str = article_search_results_to_str(research_results)

    return {"article_sources_gathered": [article_format_sources(research_results)],
            "article_research_results": [research_results_str]}

