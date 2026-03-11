# Copyright 2025 FIZ-Karlsruhe (Mustafa Sofean)

import os
import time
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

from src.agents.prompt import get_rerank_prompt_template
from src.agents.state import DeepSearchState

load_dotenv()

import google.generativeai as genai

gemini_api_key = os.getenv('GOOGLE_API_KEY')
gemini_model = os.getenv('GEMINI_API_MODEL')
openai_model = os.getenv('OPENAI_API_MODEL')
os.environ["GOOGLE_API_KEY"] = gemini_api_key
os.environ["GEMINI_API_KEY"] = gemini_api_key

def patent_reranker(topic:str, doc:str, model:str):
    """
    LangGraph node that reranks retrieved patents based on relevance.
    :param topic:
    :param doc:
    :param model:
    :return:
    """
    if 'gpt' in model:
        return rerank_by_openai(topic, doc)
    elif 'gemini' in model:
        return rerank_by_gemini(topic, doc)



def rerank_by_gemini(topic: str, doc: str):
    """
    Use Gemini model for reranks retrieved patents based on relevance.

    :param topic:
    :param doc:
    :return:
    """

    rerank_prompt_template = get_rerank_prompt_template(topic=topic, doc=doc)
    model = genai.GenerativeModel(gemini_model,
                                  )
    response = model.generate_content(rerank_prompt_template,
                                      generation_config=genai.types.GenerationConfig(
                                          candidate_count=1,
                                          top_p=0.2,
                                          top_k=2,
                                          temperature=0.0)
                                      )
    time.sleep(10)
    return response.text.strip()


def rerank_by_openai(topic: str, doc: str):
    """
    USe GPT model for reranks retrieved patents based on relevance.

    :param topic:
    :param doc:
    :return:
    """

    rerank_prompt_template = get_rerank_prompt_template(topic=topic, doc=doc)
    llm_model = ChatOpenAI(
        model_name=openai_model,
        temperature=0.1,
    )

    prompt = PromptTemplate(
        input_variables=["topic", "doc"],
        template=rerank_prompt_template,
    )
    # Create chain
    chain = prompt | llm_model | StrOutputParser()
    score = chain.invoke({"topic": topic, "document": doc})
    time.sleep(10)
    return score

