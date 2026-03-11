# Copyright 2025 FIZ-Karlsruhe (Mustafa Sofean)

from langgraph.constants import START, END
from langgraph.graph import StateGraph

from src.agents.Answer_agent import finalize_answer
from src.agents.evaluate_deep_research import patent_deep_evaluation_for_QA
from src.agents.query_agent import planning_deep_research_agent, planning_chat_deep_research_agent
from src.agents.reflection_agent import patent_deep_reflection
from src.agents.search_agent import patent_search, patent_passage_search
from src.agents.state import DeepSearchState, DeepSearchStateInput, DeepSearchStateOutput


def route_chat_research(state: DeepSearchState):
    """LangGraph routing function that determines the next step in the patent research flow.

    Controls the research loop by deciding whether to continue gathering information
    or to finalize the answer based on the configured maximum number of research loops.

    Args:
        state: Current graph state

    Returns:
        String literal indicating the next node to visit ("patent_passage_search" or "finalize_answer")
    """
    # Add nodes and edges
    builder = StateGraph(DeepSearchState, input=DeepSearchStateInput, output=DeepSearchStateOutput)
    builder.add_node("generate_query", planning_deep_research_agent)
    builder.add_node("patent_passage_search", patent_passage_search)
    builder.add_node("reflection", patent_deep_reflection)
    builder.add_node("finalize_answer", finalize_answer)

    # Add edges
    builder.add_edge(START, "generate_query")
    builder.add_edge("generate_query", "patent_passage_search")
    builder.add_edge("patent_passage_search", "reflection")
    builder.add_conditional_edges("reflection", patent_deep_evaluation_for_QA)
    builder.add_edge("finalize_answer", END)

    graph = builder.compile()

    return graph
