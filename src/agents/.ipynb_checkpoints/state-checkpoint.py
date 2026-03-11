# Copyright 2025 FIZ-Karlsruhe (Mustafa Sofean)

import operator
from dataclasses import dataclass, field
from typing_extensions import Annotated
from typing import TypedDict, List
from pydantic import Field

@dataclass(kw_only=True)
class DeepSearchState():
    research_topic: str = field(default=None)
    patent_search_query: str = field(default=None)
    patent_research_results: Annotated[list, operator.add] = field(default_factory=list)
    patent_sources_gathered: Annotated[list, operator.add] = field(default_factory=list)
    research_loop_count: int = field(default=0)
    max_research_loops: int = field(default=1)
    patent_running_summary: str = field(default=None)
    llm: str = field(default='gemini')
    reasoning_model: str = field(default='gpt')
    is_sufficient: bool = field(default=None)
    knowledge_gap: str = field(default=None)
    follow_up_query: str = field(default=None)
    answer: str = field(default=None)
    answer_sources:str = field(default=None)
    research_task: str = field(default='report')


@dataclass(kw_only=True)
class DeepSearchStateInput:
    research_topic: str = field(default=None)  # Report topic


@dataclass(kw_only=True)
class DeepSearchStateOutput:
    patent_running_summary: str = field(default=None)  # Final report


class Query(TypedDict):
    query: str
    rationale: str


class QueryGenerationState(TypedDict):
    search_query: list[Query]

class ReflectionState(TypedDict):
    is_sufficient: bool
    knowledge_gap: str
    follow_up_query: str

