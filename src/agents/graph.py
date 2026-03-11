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
    """A LangGraph routing function orchestrates the patent research process by dynamically determining the next step in the workflow.

    Manages the research loop by deciding whether to proceed with further patent retrieval or
    to produce the final report (patent_deep_review), depending on the predefined maximum number of research iterations.
    Args:
        state: Current graph state
    Returns:
        graph
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
