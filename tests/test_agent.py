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
            # Q1: Text synthesis - no images needed
            "What is RAG and how does it combine retrieval with generation to improve factual accuracy?",

            # Q2: Single image - architecture diagram
            "Show me the architecture diagram of RAG and explain how the retriever and generator components work together",

            # Q3: Table query with SQL
            "From the LLM-as-a-Judge paper results table, which model has the highest agreement rate with human judges? List the top 3 models with their exact scores.",

            # Q4: Multi-image (2-3) - RAG paper has Fig 1 (architecture), Fig 2 (document posterior heatmap), Fig 3 (performance charts)
            "Walk me through all the key figures in the RAG paper: the architecture overview, the document posterior visualization for Jeopardy generation, and the performance charts showing the effect of retrieving more documents.",

            # Q5: Multi-image (2-3) - LLM-as-a-Judge has Fig 2 (evaluation pipeline), Fig 3 (scoring), Fig 4 (yes/no + pairwise)
            "Explain the different In-Context Learning evaluation methods in LLM-as-a-Judge: how scoring works, how yes/no and pairwise comparison approaches differ, and show the evaluation pipeline diagram.",

            # Q6: Multi-image (2-3) - LLM-as-a-Judge Fig 1 (framework), Fig 5 (four scenarios), Fig 9 (improvement structure)
            "Show me the overall framework of the LLM-as-a-Judge survey, the four typical evaluation scenarios, and the structure for how to improve and evaluate LLM-as-a-Judge systems.",

            # Q7: Agent decides - text is sufficient, images exist but are not needed
            "What are the main types of biases that affect LLM-as-a-Judge systems? List position bias, length bias, and other judgment-specific biases.",

            # Q8: Agent decides - pure text answer from RAG paper results
            "What are RAG-Sequence and RAG-Token models, and how do they differ in their approach to marginalizing over retrieved documents?",

            # Q9: Agent decides - text-heavy, formal definition doesn't need a figure
            "What is the formal definition of LLM-as-a-Judge? Explain the mathematical formulation with the basic and enhanced definitions.",

            # Q10: Cross-document comparison - text only, no images needed
            "Compare the evaluation approaches used in the RAG paper and the LLM-as-a-Judge paper. How does each paper measure the quality of generated text?",

            # Q11: Table with complex filtering
            "From the results tables, list all models that have an agreement rate below 50%. Show their names and scores.",

            # Q12 (last): Out-of-domain - should say not enough info, no tools needed
            "What is the capital of France and how does its economy compare to London?",
        ]

        results: List[Tuple[str, Dict[str, Any]]] = []

        for i, query in enumerate(queries, 1):
            print(f"\n{'='*60}\nQ{i}: {query}\n{'='*60}")
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