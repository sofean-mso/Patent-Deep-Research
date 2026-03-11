# Copyright 2025 FIZ-Karlsruhe (Mustafa Sofean)

from src.agents.search_agent import patent_search_agent, article_search_agent

from langchain_core.tools import tool


@tool
def get_patent_info(
        query: str,
        schema_name: str,
        hits: int,

):
    """Retrieves the patent documents.

    Args:
        query (str):
        schema_name (str): The vespa index schema
        hits (int): number of hits to return
    """
    response = patent_search_agent(search_query=query, schema_name=schema_name, hits=hits)
    return response


@tool
def get_paper_info(
        query: str,
        hits: int):
    """Retrieves research paper documents.

       Args:
           query (str):
           hits (int): number of hits to return
       """

    response = article_search_agent(research_topic=query, hits=hits)
    return response
