# custom_tool.py

import os
from typing import Any
import duckdb
import pandas as pd
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from knowledge.vehicle_knowledge_source import ManualIndex


# ──────────────────────────────── Manual Q&A Tool ────────────────────────────────

class ManualQATool(BaseTool):
    name: str = Field(
        "manual_qa",
        description="The unique name of this tool."
    )
    description: str = Field(
        "Given a repair query, pick the right PDF via ManualIndex and answer via RetrievalQA.",
        description="A short description of what this tool does."
    )

    # These become the tool's "arguments schema"
    manual_index: ManualIndex = Field(
        ..., description="Index object to locate the correct PDF"
    )
    chunk_size: int = Field(
        800, description="Chunk size for splitting PDF text"
    )
    chunk_overlap: int = Field(
        100, description="Overlap size for splitting"
    )
    top_k: int = Field(
        4, description="How many chunks to retrieve from the vector store"
    )
    model_name: str = Field(
        "gpt-4.1-mini", description="OpenAI model to use"
    )
    temperature: float = Field(
        0.0, description="Temperature for the LLM"
    )

    def _run(self, query: str) -> str:
        # 1) find the best PDF
        hits = self.manual_index.find_manuals(query, k=1)
        if not hits:
            return "Sorry, I couldn't find a matching manual."
        pdf_path = hits[0].metadata["path"]

        # 2) load & chunk
        pages = PyPDFLoader(pdf_path).load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        docs = splitter.split_documents(pages)

        # 3) embed & vector‐store
        embeddings = OpenAIEmbeddings()
        vstore = Chroma.from_documents(docs, embeddings)

        # 4) RetrievalQA
        llm = ChatOpenAI(model=self.model_name, temperature=self.temperature)
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vstore.as_retriever(search_kwargs={"k": self.top_k}),
        )

        return qa.run(query)

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("ManualQATool does not support async.")


# ────────────────────────────── SQL Manual Lookup Tool ────────────────────────────

class SQLManualTool(BaseTool):
    name: str = Field(
        "sql_manual",
        description="The unique name of this tool."
    )
    description: str = Field(
        "Run a SELECT against manuals.duckdb to find PDF paths.",
        description="A short description of what this tool does."
    )

    def _run(self, query: str) -> str:
        conn = duckdb.connect("manuals.duckdb", read_only=True)
        try:
            df: pd.DataFrame = conn.execute(query).fetch_df()
        finally:
            conn.close()
        if df.empty:
            return ""
        return df.to_csv(index=False)

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("SQLManualTool does not support async.")

# ────────────────────────────── SQL Tool ──────────────────────────────
