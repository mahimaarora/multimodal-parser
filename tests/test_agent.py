"""Test script for the RAG Agent."""

from pathlib import Path
from typing import List, Tuple, Dict, Any

from multimodal_parser.agent import RAGAgent
from html_report import generate_html_report


def main():
    agent = RAGAgent(qdrant_path="./qdrant_data", top_k=5)
    output_dir = Path(__file__).parent / "test_outputs"
    output_dir.mkdir(exist_ok=True)

    try:
        queries = [
            # Q1: Text-based answer - synthesizes information from multiple text chunks
            "What is RAG and how does it combine retrieval with generation to improve factual accuracy?",
            
            # Q2: Image display - asks about a figure/diagram that should trigger display_image
            "Show me the architecture diagram of RAG and explain how the retriever and generator components work together",

            # Q3: Table query - asks for specific data that requires querying a table with SQL
            "From the LLM-as-a-Judge paper results table, which model has the highest agreement rate with human judges? List the top 3 models with their exact scores.",
        ]

        results: List[Tuple[str, Dict[str, Any]]] = []

        for query in queries:
            print(f"\n{'='*60}\nQuery: {query}\n{'='*60}")
            result = agent.query(query)
            results.append((query, result))

            print(f"\nAnswer:\n{result['answer']}")
            if result["images"]:
                print(f"\nImages: {len(result['images'])} displayed")
            if result["tables"]:
                print(f"\nTables: {len(result['tables'])} queried")
            if result["sources"]:
                print(f"\nSources: {', '.join(result['sources'])}")

        output_path = output_dir / "response.html"
        generate_html_report(results, str(output_path))
        print(f"\n{'='*60}\nReport saved: {output_path}")

    finally:
        agent.close()


if __name__ == "__main__":
    main()
