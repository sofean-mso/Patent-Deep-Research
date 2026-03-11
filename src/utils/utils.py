# Copyright 2025 FIZ-Karlsruhe (Mustafa Sofean)

from typing import Dict


def patent_format_sources(search_response) -> str:
    """
    Format search results into a bullet-point list of patent number with summary.
    Creates a simple bulleted list of search results with patent summary and Patent number.
    Args:
        search_response (Dict[str, Any]): Search response containing a 'results' key with
                                        a list of search result objects
    Returns:
        str: Formatted string with sources as bullet points in the format "* summary : patent number as citation"
    """
    formatted_text = ""
    for entry in search_response["retrieved_patents"]:
        formatted_text += f" {entry.get('patent number', '')}"
        formatted_text += f" :: {entry.get('title', '')}\n"
        formatted_text += f"#"

    return formatted_text.strip()


def patent_search_results_to_str(search_response) -> str:
    """
    Format search results into a bullet-point list of patent number with summary.
    Creates a simple bulleted list of search results with patent summary and Patent number.
    Args:
        search_response (Dict[str, Any]): Search response containing a 'results' key with
                                        a list of search result objects
    Returns:
        str: Formatted string with sources as bullet points in the format "* summary : patent number as citation"
    """
    formatted_text = ""
    for entry in search_response["retrieved_patents"]:
        formatted_text += f"PATENT NUMBER: {entry.get('patent number', '')} #### "
        formatted_text += f"SUMMARY: {entry.get('summary', '')}\n"

    return formatted_text.strip()


def article_format_sources(search_response) -> str:
    """
    Format search results into a bullet-point list of patent number with summary.
    Creates a simple bulleted list of search results with patent summary and Patent number.
    Args:
        search_response (Dict[str, Any]): Search response containing a 'results' key with
                                        a list of search result objects
    Returns:
        str: Formatted string with sources as bullet points in the format "* summary : patent number as citation"
    """
    formatted_text = "Sources:\n\n"
    for entry in search_response["retrieved_articles"]:
        formatted_text += f" {entry.get('title', '')}"
        formatted_text += f" :: {entry.get('url', '')}\n"

    return formatted_text.strip()


def article_search_results_to_str(search_response) -> str:
    """
    Format search results into a bullet-point list of patent number with summary.
    Creates a simple bulleted list of search results with patent summary and Patent number.
    Args:
        search_response (Dict[str, Any]): Search response containing a 'results' key with
                                        a list of search result objects
    Returns:
        str: Formatted string with sources as bullet points in the format "* summary : patent number as citation"
    """
    formatted_text = ""
    for entry in search_response["retrieved_articles"]:
        formatted_text += f"URL: {entry.get('url', '')}  #### "
        formatted_text += f" SUMMARY: {entry.get('summary', '')}\n"

    return formatted_text.strip()


def passage_format_sources(search_response) -> str:
    """
    Format search results into a bullet-point list of patent number with summary.
    Creates a simple bulleted list of search results with patent summary and Patent number.
    Args:
        search_response (Dict[str, Any]): Search response containing a 'results' key with
                                        a list of search result objects
    Returns:
        str: Formatted string with sources as bullet points in the format "* summary : patent number as citation"
    """
    formatted_text = ""
    for entry in search_response["retrieved_patents"]:
        formatted_text += f" {entry.get('patent number', '')}"
        formatted_text += f"#"

    return formatted_text.strip()

if __name__ == "__main__":
    print('hi')
