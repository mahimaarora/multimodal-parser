"""
Document Indexer - Parse documents and store chunks in Qdrant vector database.
"""

import os
import uuid
import shutil
import logging
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
load_dotenv()

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from .docling_parser import DoclingParser
from .model_manager import ModelManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

COLLECTION_NAME = "document_chunks"


class DocumentIndexer:
    """Index documents into Qdrant vector database."""
    
    def __init__(
        self,
        qdrant_path: str = "./qdrant_data",
        google_api_key: Optional[str] = None,
        images_output_dir: Optional[str] = None,
        force_reset: bool = False,
    ):
        self.api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY required. Set env var or pass google_api_key parameter.")
        
        self.qdrant_path = Path(qdrant_path)
        
        # Clear existing data if force_reset or if storage is locked
        if force_reset and self.qdrant_path.exists():
            shutil.rmtree(self.qdrant_path)
            logger.info(f"Cleared existing Qdrant data: {qdrant_path}")
        
        self.parser = DoclingParser(
            google_api_key=self.api_key,
            generate_descriptions=True,
            images_output_dir=images_output_dir or "data/images",
        )
        self.model_manager = ModelManager(self.api_key)
        
        # Try to connect, clear and retry if locked
        try:
            self.client = QdrantClient(path=qdrant_path)
        except RuntimeError as e:
            if "already accessed" in str(e) and self.qdrant_path.exists():
                logger.warning("Storage locked, clearing and retrying...")
                shutil.rmtree(self.qdrant_path)
                self.client = QdrantClient(path=qdrant_path)
            else:
                raise
        
        self._ensure_collection()
    
    def close(self):
        """Close the Qdrant client connection."""
        if hasattr(self, 'client') and self.client:
            self.client.close()
            logger.info("Qdrant connection closed")
    
    def _get_vector_size(self) -> int:
        """Get embedding vector size by embedding a sample text."""
        sample = self.model_manager.embeddings.embed_query("test")
        return len(sample)
    
    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        collections = [c.name for c in self.client.get_collections().collections]
        if COLLECTION_NAME not in collections:
            vector_size = self._get_vector_size()
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            logger.info(f"Created collection: {COLLECTION_NAME} (vector_size={vector_size})")
    
    def index_folder(self, folder_path: str, file_extensions: List[str] = None) -> int:
        """
        Index all documents in a folder.
        
        Args:
            folder_path: Path to folder containing documents
            file_extensions: List of extensions to process (default: ['.pdf'])
            
        Returns:
            Number of chunks indexed
        """
        folder = Path(folder_path)
        if not folder.exists():
            raise ValueError(f"Folder not found: {folder_path}")
        
        extensions = file_extensions or [".pdf"]
        files = [f for f in folder.iterdir() if f.suffix.lower() in extensions]
        
        logger.info(f"Found {len(files)} files to process")
        
        total_chunks = 0
        for file_path in files:
            try:
                count = self.index_file(str(file_path))
                total_chunks += count
                logger.info(f"Indexed {count} chunks from {file_path.name}")
            except Exception as e:
                logger.error(f"Failed to index {file_path}: {e}")
        
        return total_chunks
    
    def index_file(self, file_path: str) -> int:
        """Index a single document file."""
        chunks = self.parser.parse(file_path)
        if not chunks:
            return 0
        
        # Prepare points for Qdrant
        points = []
        contents = [chunk.content for chunk in chunks]
        
        # Batch embed all contents
        embeddings = self.model_manager.embeddings.embed_documents(contents)
        
        for chunk, embedding in zip(chunks, embeddings):
            # Serialize chunk using Pydantic model_dump
            payload = chunk.model_dump(exclude={"dataframe"})
            
            # Convert datetime to string
            if payload.get("extraction_timestamp"):
                payload["extraction_timestamp"] = str(payload["extraction_timestamp"])
            
            # Handle table data separately (DataFrame -> records)
            if hasattr(chunk, 'dataframe') and chunk.dataframe is not None:
                payload["table_data"] = chunk.dataframe.to_dict(orient="records")
            
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload=payload,
            ))
        
        # Upsert to Qdrant
        self.client.upsert(collection_name=COLLECTION_NAME, points=points)
        return len(points)
    
    def search(self, query: str, limit: int = 5) -> List[dict]:
        """Search for similar chunks."""
        query_embedding = self.model_manager.embeddings.embed_query(query)
        
        results = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            limit=limit,
        )
        
        return [{"score": r.score, **r.payload} for r in results.points]


if __name__ == "__main__":
    import sys
    
    folder_path = sys.argv[1] if len(sys.argv) > 1 else "data/files/"
    
    indexer = DocumentIndexer()
    try:
        count = indexer.index_folder(folder_path)
        print(f"\nIndexed {count} chunks total")
    finally:
        indexer.close()
