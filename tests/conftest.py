from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def mock_settings():
    with patch("src.settings.settings") as mock:
        mock.groq_api_key = "test-key"
        mock.groq_model = "test-model"
        yield mock


@pytest.fixture
def mock_groq_client():
    patcher1 = patch("src.classifiers.risk_detector.Groq", autospec=True)
    patcher2 = patch("src.classifiers.intent_classifier.Groq", autospec=True)
    mock_groq_cls1 = patcher1.start()
    mock_groq_cls2 = patcher2.start()
    mock_client = MagicMock()
    mock_groq_cls1.return_value = mock_client
    mock_groq_cls2.return_value = mock_client
    yield mock_client
    patcher1.stop()
    patcher2.stop()


@pytest.fixture(autouse=True)
def reset_cached_clients():
    import src.classifiers.risk_detector as rd
    import src.classifiers.intent_classifier as ic
    rd._groq_client = None
    ic._groq_client = None
    yield
    rd._groq_client = None
    ic._groq_client = None


def make_groq_response(text: str) -> MagicMock:
    choice = MagicMock()
    choice.message.content = text
    response = MagicMock()
    response.choices = [choice]
    return response


# ---------------------------------------------------------------------------
# Fixtures del panel de administración (store.py / api/routes_documents.py)
# ---------------------------------------------------------------------------
# Aditivas: no tocan mock_settings. Los módulos bajo prueba (store.py,
# routes_documents.py, routes_admin.py) hacen "from src.settings import
# settings" a nivel de módulo, así que conservan su propia referencia al
# Settings real aunque mock_settings reemplace src.settings.settings por
# un MagicMock — hay que parchear el nombre en el módulo bajo prueba.

@pytest.fixture
def fake_store_factory():
    """Fábrica de FAISSStore reales y chicos (faiss.IndexFlatIP real,
    no un mock) para tests de administración. Cada doc_id en `docs` se
    indexa como uno o más chunks con vectores determinísticos."""
    import faiss
    import numpy as np
    from src.ingestion.store import FAISSStore

    def _make(docs: dict[str, list[dict]], dim: int = 4) -> "FAISSStore":
        index = faiss.IndexFlatIP(dim)
        metadata: dict[int, dict] = {}
        faiss_id = 0
        for doc_id, chunks in docs.items():
            for chunk in chunks:
                rng = np.random.default_rng(faiss_id)
                vec = rng.random(dim).astype(np.float32)
                vec = (vec / np.linalg.norm(vec)).reshape(1, -1)
                index.add(vec)
                metadata[faiss_id] = {"doc_id": doc_id, **chunk}
                faiss_id += 1
        return FAISSStore(index=index, metadata=metadata)

    return _make


@pytest.fixture
def patched_store_settings(monkeypatch, tmp_path):
    """Redirige settings.faiss_store_path (en el módulo store) a un
    directorio temporal, para tests que ejercitan save()/save_metadata()
    sin tocar faiss_store/ real."""
    monkeypatch.setattr("src.ingestion.store.settings.faiss_store_path", tmp_path)
    return tmp_path
