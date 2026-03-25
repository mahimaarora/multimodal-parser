"""
RAG Agent with LangGraph for Multimodal Document QA.
"""

import os
import re
import logging
from typing import List, Dict, Any, Optional

import pandas as pd
import pandasql as ps
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

from .model_manager import ModelManager
from .chunk_models import BaseChunk, TextChunk, TableChunk, ImageChunk

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

COLLECTION_NAME = "document_chunks"


class RAGAgent:
    """RAG agent that retrieves chunks and uses LangGraph to answer queries."""

    def __init__(
        self,
        qdrant_path: str = "./qdrant_data",
        google_api_key: Optional[str] = None,
        top_k: int = 5,
        image_score_threshold: float = 0.5,
    ):
        self.api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY required.")

        self.top_k = top_k
        self.image_score_threshold = image_score_threshold
        self.model_manager = ModelManager(self.api_key)
        self.client = QdrantClient(path=qdrant_path)

        # Verify collection exists
        collections = [c.name for c in self.client.get_collections().collections]
        if COLLECTION_NAME not in collections:
            raise ValueError(f"Collection '{COLLECTION_NAME}' not found. Run indexer first.")

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            api_key=self.api_key,
            temperature=0,
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _extract_text_content(self, content) -> str:
        """Extract text from LLM response content (handles string or list of blocks)."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    texts.append(block.get("text", ""))
                elif isinstance(block, str):
                    texts.append(block)
            return "\n".join(texts)
        return str(content)

    # -------------------------------------------------------------------------
    # Retrieval
    # -------------------------------------------------------------------------

    def _payload_to_chunk(self, payload: dict) -> Optional[BaseChunk]:
        """Convert a Qdrant payload dict to a typed chunk."""
        chunk_type = payload.get("chunk_type")

        if chunk_type == "text":
            return TextChunk(
                chunk_id=payload.get("chunk_id", ""),
                content=payload.get("content", ""),
                sequence_number=payload.get("sequence_number", 0),
                source_document=payload.get("source_document"),
                source_page=payload.get("source_page"),
                parent_heading=payload.get("parent_heading"),
            )
        elif chunk_type == "table":
            df = pd.DataFrame(payload.get("table_data", []))
            return TableChunk(
                chunk_id=payload.get("chunk_id", ""),
                content=payload.get("content", ""),
                sequence_number=payload.get("sequence_number", 0),
                source_document=payload.get("source_document"),
                source_page=payload.get("source_page"),
                parent_heading=payload.get("parent_heading"),
                dataframe=df,
                columns=payload.get("columns"),
            )
        elif chunk_type == "image":
            return ImageChunk(
                chunk_id=payload.get("chunk_id", ""),
                content=payload.get("content", ""),
                sequence_number=payload.get("sequence_number", 0),
                source_document=payload.get("source_document"),
                source_page=payload.get("source_page"),
                parent_heading=payload.get("parent_heading"),
                image_path=payload.get("image_path"),
                image_base64=payload.get("image_base64"),
                image_format=payload.get("image_format"),
                image_type=payload.get("image_type", "other"),
            )
        return None

    def search_chunks(self, query: str) -> List[BaseChunk]:
        """Perform semantic search and return top-K relevant chunks.

        Does a primary retrieval for all chunk types, then a secondary
        retrieval filtered to images only, merging results so the LLM
        always has relevant images available to display.
        """
        query_embedding = self.model_manager.embeddings.embed_query(query)

        # Primary retrieval — all chunk types
        results = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            limit=self.top_k,
        )

        seen_ids = set()
        chunks = []
        for r in results.points:
            chunk = self._payload_to_chunk(r.payload)
            if chunk:
                seen_ids.add(chunk.chunk_id)
                chunks.append(chunk)

        # Secondary retrieval — images above score threshold
        image_results = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            limit=6,
            score_threshold=self.image_score_threshold,
            query_filter=Filter(
                must=[FieldCondition(key="chunk_type", match=MatchValue(value="image"))]
            ),
        )
        for r in image_results.points:
            chunk = self._payload_to_chunk(r.payload)
            if chunk and chunk.chunk_id not in seen_ids:
                seen_ids.add(chunk.chunk_id)
                chunks.append(chunk)

        return chunks

    def _build_context(self, chunks: List[BaseChunk]) -> str:
        """Build context string from retrieved chunks."""
        text_chunks = [c for c in chunks if isinstance(c, TextChunk)]
        image_chunks = [c for c in chunks if isinstance(c, ImageChunk)]
        table_chunks = [c for c in chunks if isinstance(c, TableChunk)]

        parts = []

        if text_chunks:
            parts.append("=== TEXT CHUNKS ===")
            for c in text_chunks:
                parts.append(f"[{c.source_document}, page {c.source_page}]: {c.content}")

        if image_chunks:
            parts.append("\n=== IMAGE CHUNKS (call display_image to display images that is relevant to the question) ===")
            for i, c in enumerate(image_chunks):
                parts.append(f"Image {i} [type: {c.image_type}]: {c.content}")

        if table_chunks:
            parts.append("\n=== TABLE CHUNKS ===")
            for i, c in enumerate(table_chunks):
                cols = c.get_columns()
                parts.append(f"Table {i}: {c.content}")
                parts.append(f"  Columns: {', '.join(cols)}")
                parts.append(f"  Rows: {c.num_rows}")

        return "\n".join(parts) if parts else "No relevant content found."

    # -------------------------------------------------------------------------
    # Agent
    # -------------------------------------------------------------------------

    def query(self, user_query: str) -> Dict[str, Any]:
        """Process user query and return answer with supporting materials."""
        # Step 1: Retrieve chunks
        chunks = self.search_chunks(user_query)

        if not chunks:
            return {
                "answer": "I couldn't find any relevant information in the knowledge base to answer your query.",
                "images": [],
                "tables": [],
                "sources": [],
            }

        # Separate by type
        image_chunks = [c for c in chunks if isinstance(c, ImageChunk)]
        table_chunks = [c for c in chunks if isinstance(c, TableChunk)]

        # Per-query state captured by tool closures
        displayed_images: List[Dict[str, Any]] = []
        queried_tables: List[Dict[str, Any]] = []

        # Step 2: Define tools as closures over retrieved chunks
        @tool
        def display_image(image_index: int) -> str:
            """Display an image to the user. Call when an image is relevant to answering the query."""
            if 0 <= image_index < len(image_chunks):
                chunk = image_chunks[image_index]
                img_index = len(displayed_images)
                displayed_images.append({
                    "chunk_id": chunk.chunk_id,
                    "path": chunk.image_path,
                    "base64": chunk.image_base64,
                    "description": chunk.content,
                    "image_type": chunk.image_type,
                    "source": f"{chunk.source_document} (page {chunk.source_page})",
                })
                return f"Image ready. Use the placeholder [IMAGE:{img_index}] in your answer where this image should appear inline. Description: {chunk.content[:100]}..."
            return f"Invalid image index {image_index}. Available: 0-{len(image_chunks)-1}"

        @tool
        def query_table(table_index: int, sql: str) -> str:
            """Execute SQL query on a table. Use 'df' as table name. Call when table data helps answer the query."""
            if 0 <= table_index < len(table_chunks):
                chunk = table_chunks[table_index]
                try:
                    result_df = ps.sqldf(sql, {"df": chunk.dataframe})
                    queried_tables.append({
                        "chunk_id": chunk.chunk_id,
                        "description": chunk.content,
                        "columns": chunk.get_columns(),
                        "preview": chunk.dataframe.head(2).to_dict(orient="records"),
                        "total_rows": len(chunk.dataframe),
                        "sql": sql,
                        "result": result_df.to_dict(orient="records"),
                        "source": f"{chunk.source_document} (page {chunk.source_page})",
                    })
                    return f"Query result:\n{result_df.to_string(index=False)}"
                except Exception as e:
                    return f"Tool error: {e}"
            return f"Invalid table index {table_index}. Available: 0-{len(table_chunks)-1}"

        # Step 3: Build system prompt
        context = self._build_context(chunks)

        system_prompt = f"""You are a document assistant. Answer using ONLY the retrieved context below.

RULES:
1. Answer from context only. If insufficient, say so. Never use prior knowledge. Address ALL parts of the user's question.
2. Images: to show an image you MUST call the display_image tool — writing [IMAGE:X] by itself does nothing. Only call display_image when the image genuinely helps answer the question — e.g. the user asks to see/show something, or the image illustrates a concept central to the answer. Do NOT display images just because they exist in context. Place each returned [IMAGE:X] placeholder on its own line between paragraphs, never inside a sentence. Never use positional language like "shown below", "depicted in", "see Figure X" — the placeholder speaks for itself.
3. Tables: call query_table with SQL (use 'df' as table name) for questions involving numbers, rankings, filtering, or comparisons. Prefer querying over reading table summaries.
4. Never describe, reference, or allude to any visual (diagram, chart, figure, heatmap) unless you have called display_image for it and placed its [IMAGE:X] placeholder. If an image is unavailable, omit it entirely — do not describe it from its caption.

RETRIEVED CONTEXT:
{context}"""

        # Step 4: Build and run a minimal LangGraph tool-calling agent
        llm_with_tools = self.llm.bind_tools([display_image, query_table])

        def call_model(state: MessagesState):
            return {"messages": [llm_with_tools.invoke(state["messages"])]}

        graph_builder = StateGraph(MessagesState)
        graph_builder.add_node("agent", call_model)
        graph_builder.add_node("tools", ToolNode([display_image, query_table]))
        graph_builder.add_edge(START, "agent")
        graph_builder.add_conditional_edges(
            "agent", tools_condition, {"tools": "tools", "__end__": END}
        )
        graph_builder.add_edge("tools", "agent")

        graph = graph_builder.compile()
        result = graph.invoke(
            {"messages": [SystemMessage(content=system_prompt), HumanMessage(content=user_query)]},
            {"recursion_limit": 10},
        )
        messages = result["messages"]

        # If visuals were requested and we have image chunks, enforce one retry if no image got displayed.
        visual_request = bool(
            re.search(
                r"\b(show|see|figure|figures|diagram|framework|pipeline|chart)\b",
                user_query,
                re.IGNORECASE,
            )
        )
        if visual_request and image_chunks and not displayed_images:
            retry_instruction = HumanMessage(
                content=(
                    "Retry: you must call display_image for relevant image chunks first, "
                    "then answer with returned [IMAGE:X] placeholders inline."
                )
            )
            result = graph.invoke({"messages": messages + [retry_instruction]}, {"recursion_limit": 10})
            messages = result["messages"]

        # Step 5: Extract answer from final AI message
        answer = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and not msg.tool_calls:
                answer = self._extract_text_content(msg.content)
                break
        if not answer:
            answer = "No answer produced."

        # Strip any [IMAGE:X] placeholders that don't have a matching displayed image
        def _clean_placeholder(match):
            idx = int(match.group(1))
            if 0 <= idx < len(displayed_images):
                return match.group(0)
            return ""

        answer = re.sub(r'\s*\[IMAGE:\s*(\d+)\s*\]\s*', _clean_placeholder, answer).strip()

        # Build sources list
        sources = list(set(
            f"{c.source_document} (page {c.source_page})"
            for c in chunks if c.source_document
        ))

        return {
            "answer": answer,
            "images": displayed_images,
            "tables": queried_tables,
            "sources": sources,
        }

    def close(self):
        """Close connections."""
        if self.client:
            self.client.close()


# -----------------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    agent = RAGAgent()
    try:
        query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Query: ")
        result = agent.query(query)

        print(f"\n{'='*60}")
        print(f"Answer: {result['answer']}")
        print(f"{'='*60}")

        if result["images"]:
            print(f"\nDisplayed Images: {len(result['images'])}")
            for img in result["images"]:
                print(f"  - {img['description'][:50]}... ({img['source']})")

        if result["tables"]:
            print(f"\nQueried Tables: {len(result['tables'])}")
            for tbl in result["tables"]:
                print(f"  - SQL: {tbl['sql']} ({tbl['source']})")

        if result["sources"]:
            print(f"\nSources: {', '.join(result['sources'])}")

    finally:
        agent.close()
