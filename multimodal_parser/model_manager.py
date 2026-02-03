"""Model Manager for LLM and Embedding operations."""

import os
import logging
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages LLM and embedding models for document processing."""
    
    def __init__(self, google_api_key: Optional[str] = None):
        self.api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        self._llm = None
        self._embeddings = None
    
    @property
    def llm(self) -> ChatGoogleGenerativeAI:
        """Get or create the LLM instance."""
        if self._llm is None:
            self._llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.1,
                max_tokens=1024,
                timeout=60,
                max_retries=2,
                api_key=self.api_key,
            )
        return self._llm
    
    @property
    def embeddings(self) -> GoogleGenerativeAIEmbeddings:
        """Get or create the embeddings instance."""
        if self._embeddings is None:
            self._embeddings = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                google_api_key=self.api_key,
            )
        return self._embeddings
    
    def describe_table(self, headers: list, num_rows: int, table_preview: str, caption: str = "") -> str:
        """Generate a retrieval-optimized description for a table."""
        prompt = f"""Generate a retrieval-optimized description for this table.

                    TABLE SCHEMA
                    - Columns: {', '.join(headers)} ({len(headers)} total)
                    - Rows: {num_rows}
                    {f"- Caption: {caption}" if caption else ""}

                    SAMPLE ROWS
                    {table_preview}

                    Write 2-3 sentences explaining what this table contains and what queries it can answer.
                    Include the domain, data types, and example questions it addresses."""

        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            logger.warning(f"Table description failed: {e}")
            return f"Table with {num_rows} rows and {len(headers)} columns. Headers: {', '.join(headers)}"
    
    def infer_image_type(self, description: str, caption: str = "") -> str:
        """Infer image type from description and caption."""
        if not description and not caption:
            return "other"
        
        valid_types = ["photo", "diagram", "chart", "logo", "screenshot", "other"]
        
        context = ""
        if description:
            context += f"DESCRIPTION: {description}\n"
        if caption:
            context += f"CAPTION: {caption}\n"
        
        prompt = f"""Classify this image into ONE type:
                        - photo: real objects, people, places, scenes
                        - diagram: flowcharts, architecture, process flows, schematics
                        - chart: bar charts, pie charts, graphs, data visualizations
                        - logo: company logos, brand marks, icons, symbols
                        - screenshot: UI screenshots, application windows, web pages
                        - other: anything else

                    {context}
                    Respond with ONLY the type (one word, lowercase): photo, diagram, chart, logo, screenshot, or other"""

        try:
            result = self.llm.invoke(prompt).content.strip().lower()
            if result in valid_types:
                return result
            for t in valid_types:
                if t in result:
                    return t
            return "other"
        except Exception as e:
            logger.warning(f"Image type inference failed: {e}")
            return "other"


_default_manager: Optional[ModelManager] = None


def get_model_manager(api_key: Optional[str] = None) -> ModelManager:
    """Get or create the default ModelManager instance."""
    global _default_manager
    if _default_manager is None or api_key:
        _default_manager = ModelManager(api_key)
    return _default_manager
