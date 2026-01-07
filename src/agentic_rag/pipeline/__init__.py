"""Pipeline orchestration and fluent builder API."""

from agentic_rag.pipeline.agentic import AgenticPipeline
from agentic_rag.pipeline.base import BasePipeline, IngestResult, PipelineResult
from agentic_rag.pipeline.builder import PipelineBuilder, RAGPipeline
from agentic_rag.pipeline.corrective import (
    CorrectivePipeline,
    CRAGAssessment,
    RetrievalConfidence,
)
from agentic_rag.pipeline.standard import StandardPipeline

__all__ = [
    # Base
    "BasePipeline",
    "PipelineResult",
    "IngestResult",
    # Builder
    "PipelineBuilder",
    "RAGPipeline",
    # Variants
    "StandardPipeline",
    "AgenticPipeline",
    "CorrectivePipeline",
    # CRAG types
    "CRAGAssessment",
    "RetrievalConfidence",
]
