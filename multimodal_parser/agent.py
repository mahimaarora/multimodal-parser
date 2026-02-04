"""
RAG Agent with Tool Calling for Multimodal Document QA.
"""

import os
import logging
from typing import List, Dict, Any, Optional

import pandas as pd
import pandasql as ps
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from .model_manager import ModelManager
from .chunk_models import BaseChunk, TextChunk, TableChunk, ImageChunk, ChunkType

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

COLLECTION_NAME = "document_chunks"


class RAGAgent:
    """RAG agent that retrieves chunks and uses tools to answer queries."""

    def __init__(
        self,
        qdrant_path: str = "./qdrant_data",
        google_api_key: Optional[str] = None,
        top_k: int = 5,
    ):
        self.api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY required.")

        self.top_k = top_k
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

        # Response state
        self._displayed_images: List[Dict[str, Any]] = []
        self._queried_tables: List[Dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _extract_text_content(self, content) -> str:
        """Extract text from LLM response content (handles string or list of blocks)."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            # Extract text from content blocks
            texts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    texts.append(block.get("text", ""))
                elif isinstance(block, str):
                    texts.append(block)
            return "\n".join(texts)
        return str(content)

    # -------------------------------------------------------------------------
    # Tools
    # -------------------------------------------------------------------------

    def search_chunks(self, query: str) -> List[BaseChunk]:
        """Perform semantic search and return top-K relevant chunks."""
        query_embedding = self.model_manager.embeddings.embed_query(query)
        results = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            limit=self.top_k,
        )

        chunks = []
        for r in results.points:
            payload = r.payload
            chunk_type = payload.get("chunk_type")

            if chunk_type == "text":
                chunks.append(TextChunk(
                    chunk_id=payload.get("chunk_id", ""),
                    content=payload.get("content", ""),
                    sequence_number=payload.get("sequence_number", 0),
                    source_document=payload.get("source_document"),
                    source_page=payload.get("source_page"),
                    parent_heading=payload.get("parent_heading"),
                ))
            elif chunk_type == "table":
                df = pd.DataFrame(payload.get("table_data", []))
                chunks.append(TableChunk(
                    chunk_id=payload.get("chunk_id", ""),
                    content=payload.get("content", ""),
                    sequence_number=payload.get("sequence_number", 0),
                    source_document=payload.get("source_document"),
                    source_page=payload.get("source_page"),
                    parent_heading=payload.get("parent_heading"),
                    dataframe=df,
                    columns=payload.get("columns"),
                ))
            elif chunk_type == "image":
                chunks.append(ImageChunk(
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
                ))

        return chunks

    def display_image(self, image_chunk: ImageChunk) -> str:
        """Render an image to the user. Returns confirmation message."""
        self._displayed_images.append({
            "chunk_id": image_chunk.chunk_id,
            "path": image_chunk.image_path,
            "base64": image_chunk.image_base64,
            "description": image_chunk.content,
            "image_type": image_chunk.image_type,
            "source": f"{image_chunk.source_document} (page {image_chunk.source_page})",
        })
        return f"Image displayed: {image_chunk.content[:100]}..."

    def query_table(self, sql: str, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Execute SQL query against a table's dataframe. Table is named 'df'."""
        return ps.sqldf(sql, {"df": dataframe})

    # -------------------------------------------------------------------------
    # Agent Logic
    # -------------------------------------------------------------------------

    def _build_tool_definitions(self):
        """Build tool definitions for LLM."""
        return [
            {
                "name": "display_image",
                "description": "Display an image to the user. Call when an image is relevant to answering the query.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_index": {
                            "type": "integer",
                            "description": "Index of the image chunk to display (0-based)",
                        }
                    },
                    "required": ["image_index"],
                },
            },
            {
                "name": "query_table",
                "description": "Execute SQL query on a table. Use 'df' as table name. Call when table data helps answer the query.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "table_index": {
                            "type": "integer",
                            "description": "Index of the table chunk to query (0-based)",
                        },
                        "sql": {
                            "type": "string",
                            "description": "SQL query to execute (use 'df' as table name)",
                        },
                    },
                    "required": ["table_index", "sql"],
                },
            },
        ]

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
            parts.append("\n=== IMAGE CHUNKS ===")
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

    def _execute_tool_call(
        self,
        tool_name: str,
        args: Dict[str, Any],
        image_chunks: List[ImageChunk],
        table_chunks: List[TableChunk],
    ) -> str:
        """Execute a tool call and return result."""
        try:
            if tool_name == "display_image":
                idx = args.get("image_index", 0)
                if 0 <= idx < len(image_chunks):
                    return self.display_image(image_chunks[idx])
                return f"Invalid image index {idx}. Available: 0-{len(image_chunks)-1}"

            elif tool_name == "query_table":
                idx = args.get("table_index", 0)
                sql = args.get("sql", "")
                if 0 <= idx < len(table_chunks):
                    table_chunk = table_chunks[idx]
                    result_df = self.query_table(sql, table_chunk.dataframe)
                    self._queried_tables.append({
                        "chunk_id": table_chunk.chunk_id,
                        "description": table_chunk.content,
                        "columns": table_chunk.get_columns(),
                        "preview": table_chunk.dataframe.head(2).to_dict(orient="records"),
                        "total_rows": len(table_chunk.dataframe),
                        "sql": sql,
                        "result": result_df.to_dict(orient="records"),
                        "source": f"{table_chunk.source_document} (page {table_chunk.source_page})",
                    })
                    return f"Query result:\n{result_df.to_string(index=False)}"
                return f"Invalid table index {idx}. Available: 0-{len(table_chunks)-1}"

            return f"Unknown tool: {tool_name}"
        except Exception as e:
            return f"Tool error: {e}"

    def query(self, user_query: str) -> Dict[str, Any]:
        """Process user query and return answer with supporting materials."""
        # Reset state
        self._displayed_images = []
        self._queried_tables = []

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

        # Step 2: Build context and system prompt
        context = self._build_context(chunks)

        system_prompt = f"""You are a document assistant. Answer the user's question using ONLY the retrieved context below.

RULES:
1. Use ONLY information from the context. Do NOT use prior knowledge.
2. If the context doesn't contain enough information to answer, respond: "The retrieved documents do not contain enough information to answer this question."
3. If an image is relevant to the answer (e.g., a diagram, chart, or illustration), call display_image with its index.
4. If table data can help answer the question, call query_table with appropriate SQL (table name is 'df').
5. Synthesize information from multiple chunks when needed.

RETRIEVED CONTEXT:
{context}"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_query),
        ]

        # Step 3: Run agent loop with tool calling
        tools = self._build_tool_definitions()
        llm_with_tools = self.llm.bind_tools(tools)

        max_iterations = 5
        for _ in range(max_iterations):
            response = llm_with_tools.invoke(messages)
            messages.append(response)

            if not response.tool_calls:
                break

            # Execute tool calls
            for tool_call in response.tool_calls:
                result = self._execute_tool_call(
                    tool_call["name"],
                    tool_call["args"],
                    image_chunks,
                    table_chunks,
                )
                messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))

        # Build sources list
        sources = list(set(
            f"{c.source_document} (page {c.source_page})"
            for c in chunks if c.source_document
        ))

        return {
            "answer": self._extract_text_content(response.content),
            "images": self._displayed_images,
            "tables": self._queried_tables,
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
