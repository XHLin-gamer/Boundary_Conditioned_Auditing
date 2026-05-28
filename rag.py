from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from models import BackboneLLM
from prompts import COMPRESSOR_PROMPT, GENERATOR_PROMPT


dotenv.load_dotenv()


@dataclass
class RAGStep:
    question: str
    retrieval_query: object
    retrieved_documents: list
    summary: str
    answer: str


class ReActRAG:
    def __init__(
        self,
        data_dir: str,
        db_dir: str,
        top_k: int = 5,
        embedding_model: str = "gemini-embedding-001",
        chat_model: str = "deepseek-v4-flash",
        chat_provider: str = "auto",
        chat_temperature: float = 0,
        chat_enable_thinking: bool = False,
        chat_reasoning_effort: str | None = None,
        chunk_size: int = 900,
        chunk_overlap: int = 120,
    ):
        self.data_dir = data_dir
        self.db_dir = db_dir
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=embedding_model,
            api_key=os.environ.get("GEMINI_API_KEY"),
        )
        self.llm = BackboneLLM(
            model=chat_model,
            provider=chat_provider,
            temperature=chat_temperature,
            reasoning_effort=chat_reasoning_effort,
            enable_thinking=chat_enable_thinking,
        )
        self.vectorstore = None

    def build_index(self):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        chunks = splitter.split_documents(self.load_docs())
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.db_dir,
        )
        return self

    def load_index(self):
        self.vectorstore = Chroma(
            persist_directory=self.db_dir,
            embedding_function=self.embeddings,
        )
        return self

    def load_docs(self):
        docs = []
        for path in Path(self.data_dir).rglob("*"):
            if path.suffix == ".pdf":
                docs.extend(PyPDFLoader(str(path)).load())
            elif path.suffix in {".txt", ".md"}:
                docs.extend(TextLoader(str(path), encoding="utf-8").load())
        return docs

    def answer(self, question: str, retrieval_query=None, return_trace: bool = True):
        retrieval_query = retrieval_query or question
        docs = self.retrieve(retrieval_query)
        summary = self.compress(question, docs)
        answer = self.generate(question, summary)
        if not return_trace:
            return answer
        return answer, [
            RAGStep(
                question=question,
                retrieval_query=retrieval_query,
                retrieved_documents=[
                    self.serialize_document(index, doc)
                    for index, doc in enumerate(docs)
                ],
                summary=summary,
                answer=answer,
            )
        ]

    def retrieve(self, query):
        if self.vectorstore is None:
            raise RuntimeError("Build or load the vector index before answering.")
        queries = query if isinstance(query, (list, tuple)) else [query]
        queries = [item for item in queries if item]
        if len(queries) == 1:
            docs = self.retrieve_one(queries[0], k=self.top_k * 4)
        else:
            batches = [self.retrieve_one(item, k=self.top_k * 2) for item in queries]
            docs = self.round_robin_documents(batches)
        return self.limit_by_source(docs)

    def retrieve_one(self, query: str, k: int):
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        for attempt in range(6):
            try:
                return retriever.invoke(query)
            except Exception as exc:
                message = str(exc)
                if "RESOURCE_EXHAUSTED" not in message and "429" not in message:
                    raise
                if attempt == 5:
                    raise
                time.sleep(10 * (2**attempt))

    def compress(self, question: str, docs) -> str:
        if not docs:
            return "No relevant documents were retrieved."
        documents = "\n\n".join(
            self.format_document(index, doc)
            for index, doc in enumerate(docs)
        )
        return self.call_llm(
            f"""{COMPRESSOR_PROMPT}

Question:
{question}

Documents:
{documents}
"""
        )

    def generate(self, question: str, summary: str) -> str:
        return self.call_llm(
            f"""{GENERATOR_PROMPT}

Question:
{question}

Evidence notes:
{summary}

Answer:
"""
        )

    def call_llm(self, prompt: str) -> str:
        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)

    def limit_by_source(self, docs, max_per_source: int = 2):
        selected = []
        selected_keys = set()
        source_counts = {}
        for doc in docs:
            key = self.document_key(doc)
            source = doc.metadata.get("source")
            if key in selected_keys or source_counts.get(source, 0) >= max_per_source:
                continue
            selected.append(doc)
            selected_keys.add(key)
            source_counts[source] = source_counts.get(source, 0) + 1
            if len(selected) >= self.top_k:
                return selected
        for doc in docs:
            key = self.document_key(doc)
            if key in selected_keys:
                continue
            selected.append(doc)
            selected_keys.add(key)
            if len(selected) >= self.top_k:
                return selected
        return selected

    @staticmethod
    def round_robin_documents(batches):
        merged = []
        seen = set()
        max_length = max((len(batch) for batch in batches), default=0)
        for index in range(max_length):
            for batch in batches:
                if index >= len(batch):
                    continue
                doc = batch[index]
                key = ReActRAG.document_key(doc)
                if key in seen:
                    continue
                merged.append(doc)
                seen.add(key)
        return merged

    @staticmethod
    def document_key(doc):
        return (
            doc.metadata.get("source"),
            doc.metadata.get("page"),
            doc.metadata.get("start_index"),
            doc.page_content[:120],
        )

    @staticmethod
    def format_document(index: int, doc) -> str:
        source = doc.metadata.get("source")
        page = doc.metadata.get("page")
        metadata = []
        if source:
            metadata.append(f"Source: {source}")
        if page is not None:
            metadata.append(f"Page: {page}")
        meta = f"\n{' | '.join(metadata)}" if metadata else ""
        return f"[Document {index + 1}]{meta}\n{doc.page_content}"

    @staticmethod
    def serialize_document(index: int, doc):
        return {
            "rank": index + 1,
            "source": doc.metadata.get("source"),
            "page": doc.metadata.get("page"),
            "metadata": dict(doc.metadata),
            "content": doc.page_content,
        }

