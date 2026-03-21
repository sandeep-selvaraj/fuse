"""Fuse: Train small LLMs and deploy them for fast structured extraction on CPU."""

from fuse._version import __version__
from fuse.config import ExportConfig, ExtractConfig, InferenceConfig, TrainConfig
from fuse.extraction.extractor import Extractor
from fuse.extraction.schema import SchemaBuilder
from fuse.inference.llama_cpp import LlamaCppBackend

__all__ = [
    "ExportConfig",
    "ExtractConfig",
    "Extractor",
    "InferenceConfig",
    "LlamaCppBackend",
    "SchemaBuilder",
    "TrainConfig",
    "__version__",
]
