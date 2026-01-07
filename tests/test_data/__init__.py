"""
Test data package for RAG Optimizer.

Contains:
- papers/: Academic PDFs for testing (Attention is All You Need, RAG, BERT, etc.)
- Sample documents and queries for unit tests
"""

from pathlib import Path

PAPERS_DIR = Path(__file__).parent / "papers"

# Available papers
AVAILABLE_PAPERS = {
    "attention": PAPERS_DIR / "attention_is_all_you_need.pdf",
    "rag": PAPERS_DIR / "rag_paper.pdf",
    "bert": PAPERS_DIR / "bert_paper.pdf",
    "gpt3": PAPERS_DIR / "gpt3_paper.pdf",
    "llama2": PAPERS_DIR / "llama2_paper.pdf",
    "self_rag": PAPERS_DIR / "self_rag_paper.pdf",
    "cot": PAPERS_DIR / "chain_of_thought.pdf",
    "crag": PAPERS_DIR / "crag_paper.pdf",
}


def get_paper_path(name: str) -> Path:
    """Get path to a test paper by name."""
    if name not in AVAILABLE_PAPERS:
        raise ValueError(f"Unknown paper: {name}. Available: {list(AVAILABLE_PAPERS.keys())}")
    return AVAILABLE_PAPERS[name]


def list_available_papers() -> list[str]:
    """List all available test papers."""
    return [name for name, path in AVAILABLE_PAPERS.items() if path.exists()]
