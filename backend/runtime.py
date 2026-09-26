"""Production graph factory."""

from backend.agents.main import create_answer_generator
from backend.config import Settings, get_settings
from backend.graph import build_graph
from backend.knowledge import LocalEvidenceStore
from backend.retrieval.chroma import ChromaVectorStore


def create_runtime_graph(settings: Settings | None = None):
    runtime_settings = settings or get_settings()
    store = LocalEvidenceStore(
        runtime_settings.data_dir, ChromaVectorStore(runtime_settings)
    )
    return build_graph(
        settings=runtime_settings,
        retrieve=store.search,
        generate=create_answer_generator(runtime_settings),
    )