# Copyright 2025 FIZ-Karlsruhe (Mustafa Sofean)

from langgraph.constants import START, END
from langgraph.graph import StateGraph

from src.agents.evaluate_deep_research import patent_deep_evaluation
from src.agents.query_agent import planning_deep_research_agent
from src.agents.reflection_agent import patent_deep_reflection
from src.agents.search_agent import patent_search
from src.agents.state import DeepSearchState, DeepSearchStateInput, DeepSearchStateOutput
from src.agents.review_agent import patent_deep_review


def route_research(state: DeepSearchState):
    """LangGraph routing function that determines the next step in the patent research flow.

    Controls the research loop by deciding whether to continue gathering patent information
    or to create the report(patent_deep_review) based on the configured maximum number of research loops.

    Args:
        state: Current graph state containing the research loop count
    Returns:
        String literal indicating the next node to visit ("patent_research" or "patent_deep_review")
    """
    # Add nodes and edges
    builder = StateGraph(DeepSearchState, input=DeepSearchStateInput, output=DeepSearchStateOutput)
    builder.add_node("generate_query", planning_deep_research_agent)
    builder.add_node("patent_research", patent_search)
    builder.add_node("reflection", patent_deep_reflection)
    builder.add_node("patent_deep_review", patent_deep_review)

    # Add edges
    builder.add_edge(START, "generate_query")
    builder.add_edge("generate_query", "patent_research")
    builder.add_edge("patent_research", "reflection")
    builder.add_conditional_edges("reflection", patent_deep_evaluation)
    builder.add_edge("patent_deep_review", END)

    graph = builder.compile()


    return graph
