# Copyright 2025 FIZ-Karlsruhe (Mustafa Sofean)
import re

import arxiv
import os
import pandas as pd
import json
from dotenv import load_dotenv

load_dotenv()


def get_articles(query: str, topn: int = 20):
    """
    Extracting the Arxiv articles
    :param query:
    :param topn:
    :return:
    """

    # query = re.sub('[^a-zA-Z]', ' ', query)
    client = arxiv.Client(page_size=100,
                          delay_seconds=3.0,
                          num_retries=3)
    search = arxiv.Search(
        query=query,
        max_results=topn,
        sort_by=arxiv.SortCriterion.Relevance
    )
    results = client.results(search)
    # df = pd.DataFrame(columns=["Article-No", "Title", "Abstract", "Publication-Date", "url"])
    articles = []
    for article in results:
        article_id = article.get_short_id()
        title = article.title
        abstract = article.summary
        pub_date = article.published
        url = article.pdf_url
        row = {"Article-No": article_id, "Title": title, "Abstract": abstract, "Publication-Date": pub_date, "url": url}
        # df = df._append(row, ignore_index=True)
        articles.append(row)

    # response = df.to_json(orient='records')  # '{"news":'+df.to_json(orient='records')+"}"
    return {"retrieved_papers": articles}
