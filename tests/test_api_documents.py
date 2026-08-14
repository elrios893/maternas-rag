"""
test_api_documents.py — Cobertura de los endpoints /documents* y su auth.

TestClient(app) se usa SIN el bloque `with`, para saltear el lifespan de
FastAPI (que intentaría cargar el índice FAISS real desde disco). Los
routers reciben su store a través de _get_store, parcheado por test hacia
un FAISSStore real y chico (fixture `wired_store`, construida con
`fake_store_factory` de conftest.py).

Cualquier test que efectivamente llegue a store.save()/save_metadata()
usa `patched_store_settings` (conftest.py) para no escribir sobre el
faiss_store/ real del proyecto.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.ingestion.store import is_active


@pytest.fixture
def client():
    return TestClient(app)   # sin "with": no dispara el lifespan


@pytest.fixture
def store(fake_store_factory):
    return fake_store_factory({
        "doc1": [
            {"source_dataset": "medmcqa", "language": "es", "text": "contenido uno",
             "chunk_index": 0, "is_chunk": False, "chunk_id": "c-doc1-0"},
        ],
        "doc2": [
            {"source_dataset": "upload", "language": "es", "text": "contenido dos a",
             "chunk_index": 0, "is_chunk": True, "chunk_id": "c-doc2-0"},
            {"source_dataset": "upload", "language": "es", "text": "contenido dos b",
             "chunk_index": 1, "is_chunk": True, "chunk_id": "c-doc2-1"},
        ],
    })


@pytest.fixture
def wired_store(monkeypatch, store):
    """Conecta `store` como el objeto que usan los routers Y /chat — la
    misma instancia, para poder testear que un upload/toggle vía API
    muta el singleton real de src.rag.retriever y no una copia aparte."""
    monkeypatch.setattr("src.api.routes_documents._get_store", lambda: store)
    monkeypatch.setattr("src.api.routes_admin._get_store", lambda: store)
    monkeypatch.setattr("src.rag.retriever._store", store)
    return store


@pytest.fixture
def admin_token(monkeypatch):
    token = "s3cr3t-test-token"
    monkeypatch.setattr("src.api.auth.settings.admin_api_token", token)
    return token


@pytest.fixture
def patched_embedder(monkeypatch):
    """Evita cargar el SentenceTransformer real en los tests de upload."""
    def fake_embed_documents(texts, batch_size=64, show_progress=True):
        rng = np.random.default_rng(len(texts))
        vecs = rng.random((len(texts), 4)).astype(np.float32)
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    monkeypatch.setattr("src.ingestion.store.embed_documents", fake_embed_documents)


def _headers(token: str) -> dict:
    return {"X-Admin-Token": token}


SMALL_TXT = ("contenido de prueba para subir un documento nuevo " * 3).encode("utf-8")

ENDPOINT_CALLS = {
    "stats":  lambda c, h: c.get("/documents/stats", headers=h),
    "list":   lambda c, h: c.get("/documents", headers=h),
    "detail": lambda c, h: c.get("/documents/doc1", headers=h),
    "patch":  lambda c, h: c.patch("/documents/doc1", json={"active": False}, headers=h),
    "upload": lambda c, h: c.post(
        "/documents/upload", headers=h,
        files={"file": ("nuevo.txt", SMALL_TXT, "text/plain")},
    ),
}


# ---------------------------------------------------------------------------
# Matriz de autenticación — los 5 endpoints, 4 escenarios de token
# ---------------------------------------------------------------------------

class TestAuthMatrix:
    @pytest.mark.parametrize("name", ENDPOINT_CALLS.keys())
    def test_sin_header_401(self, client, wired_store, admin_token, name):
        resp = ENDPOINT_CALLS[name](client, {})
        assert resp.status_code == 401

    @pytest.mark.parametrize("name", ENDPOINT_CALLS.keys())
    def test_token_incorrecto_401(self, client, wired_store, admin_token, name):
        resp = ENDPOINT_CALLS[name](client, _headers("token-equivocado"))
        assert resp.status_code == 401

    @pytest.mark.parametrize("name", ENDPOINT_CALLS.keys())
    def test_sin_admin_api_token_configurado_503(self, client, wired_store, monkeypatch, name):
        monkeypatch.setattr("src.api.auth.settings.admin_api_token", "")
        resp = ENDPOINT_CALLS[name](client, _headers("cualquiera"))
        assert resp.status_code == 503

    @pytest.mark.parametrize("name", ENDPOINT_CALLS.keys())
    def test_token_correcto_200(
        self, client, wired_store, admin_token, name, patched_store_settings, patched_embedder,
    ):
        resp = ENDPOINT_CALLS[name](client, _headers(admin_token))
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Ruteo — /stats no debe ser tragado por /{doc_id}
# ---------------------------------------------------------------------------

class TestRuteo:
    def test_stats_no_es_tratado_como_doc_id(self, client, fake_store_factory, monkeypatch, admin_token):
        # Incluye a propósito un documento literalmente llamado "stats".
        store = fake_store_factory({
            "stats": [{"source_dataset": "upload", "language": "es", "text": "x", "chunk_index": 0}],
        })
        monkeypatch.setattr("src.api.routes_documents._get_store", lambda: store)

        resp = client.get("/documents/stats", headers=_headers(admin_token))

        assert resp.status_code == 200
        assert "total_vectors" in resp.json()   # forma de DocumentStatsResponse, no de detalle


# ---------------------------------------------------------------------------
# Validación de query params
# ---------------------------------------------------------------------------

class TestValidacion:
    def test_page_cero_422(self, client, wired_store, admin_token):
        resp = client.get("/documents?page=0", headers=_headers(admin_token))
        assert resp.status_code == 422

    def test_per_page_101_422(self, client, wired_store, admin_token):
        resp = client.get("/documents?per_page=101", headers=_headers(admin_token))
        assert resp.status_code == 422

    def test_detalle_per_page_51_422(self, client, wired_store, admin_token):
        resp = client.get("/documents/doc1?per_page=51", headers=_headers(admin_token))
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Filtrado y 404
# ---------------------------------------------------------------------------

class TestListadoYDetalle:
    def test_filtro_case_insensitive_por_doc_id(self, client, wired_store, admin_token):
        resp = client.get("/documents?search=DOC1", headers=_headers(admin_token))
        ids = [d["doc_id"] for d in resp.json()["documents"]]
        assert ids == ["doc1"]

    def test_filtro_case_insensitive_por_source_dataset(self, client, wired_store, admin_token):
        resp = client.get("/documents?search=MEDMCQA", headers=_headers(admin_token))
        ids = [d["doc_id"] for d in resp.json()["documents"]]
        assert ids == ["doc1"]

    def test_detalle_404_doc_desconocido(self, client, wired_store, admin_token):
        resp = client.get("/documents/no-existe", headers=_headers(admin_token))
        assert resp.status_code == 404

    def test_detalle_agrega_todos_los_chunks(self, client, wired_store, admin_token):
        resp = client.get("/documents/doc2", headers=_headers(admin_token))
        body = resp.json()
        assert body["chunk_count"] == 2
        assert len(body["chunks"]) == 2


# ---------------------------------------------------------------------------
# PATCH — toggle, persistencia, reversión ante fallo
# ---------------------------------------------------------------------------

class TestPatch:
    def test_togglea_y_llama_save_metadata_una_vez(
        self, client, wired_store, admin_token, patched_store_settings,
    ):
        calls = {"n": 0}
        original = wired_store.save_metadata

        def spy(*a, **kw):
            calls["n"] += 1
            return original(*a, **kw)

        wired_store.save_metadata = spy

        resp = client.patch("/documents/doc1", json={"active": False}, headers=_headers(admin_token))

        assert resp.status_code == 200
        assert resp.json()["active"] is False
        assert calls["n"] == 1
        assert not is_active(wired_store.metadata[0])

    def test_falla_persistencia_revierte_en_memoria(self, client, wired_store, admin_token):
        def boom(*a, **kw):
            raise RuntimeError("disco lleno")

        wired_store.save_metadata = boom

        resp = client.patch("/documents/doc1", json={"active": False}, headers=_headers(admin_token))

        assert resp.status_code == 500
        assert is_active(wired_store.metadata[0])   # revertido: sigue activo


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

class TestUpload:
    def _upload(self, client, admin_token, filename: str, content: bytes, content_type="text/plain"):
        return client.post(
            "/documents/upload",
            headers=_headers(admin_token),
            files={"file": (filename, content, content_type)},
        )

    def test_no_txt_400(self, client, wired_store, admin_token):
        resp = self._upload(client, admin_token, "documento.pdf", SMALL_TXT, "application/pdf")
        assert resp.status_code == 400

    def test_nombre_con_separador_de_ruta_400(self, client, wired_store, admin_token):
        resp = self._upload(client, admin_token, "sub/dir.txt", SMALL_TXT)
        assert resp.status_code == 400

    def test_archivo_sobre_el_limite_413(self, client, wired_store, admin_token):
        from src.api.routes_documents import MAX_UPLOAD_BYTES
        content = b"a" * (MAX_UPLOAD_BYTES + 1)
        resp = self._upload(client, admin_token, "grande.txt", content)
        assert resp.status_code == 413

    def test_demasiados_chunks_413(self, client, wired_store, admin_token):
        # ~900k caracteres, muy por encima de MAX_UPLOAD_CHUNKS*chunk_size
        # y todavía bajo MAX_UPLOAD_BYTES (2 MB) — no llega a embeddear.
        content = ("0123456789 " * 82_000).encode("utf-8")
        resp = self._upload(client, admin_token, "enorme.txt", content)
        assert resp.status_code == 413

    def test_solo_espacios_400(self, client, wired_store, admin_token):
        resp = self._upload(client, admin_token, "vacio.txt", b"   \n   \n  ")
        assert resp.status_code == 400

    def test_duplicado_case_insensitive_409(self, client, wired_store, admin_token):
        # "doc1" ya existe en el store de la fixture.
        resp = self._upload(client, admin_token, "DOC1.txt", SMALL_TXT)
        assert resp.status_code == 409

    def test_camino_feliz(
        self, client, wired_store, admin_token, patched_store_settings, patched_embedder,
    ):
        total_antes = wired_store.total

        resp = self._upload(client, admin_token, "nuevo.txt", SMALL_TXT)

        assert resp.status_code == 200
        body = resp.json()
        assert body["doc_id"] == "nuevo"
        assert body["chunks_created"] >= 1
        assert body["total_vectors"] == total_antes + body["chunks_created"]

    def test_muta_el_singleton_de_retriever(
        self, client, wired_store, admin_token, patched_store_settings, patched_embedder,
    ):
        """Regresión: el upload debe mutar src.rag.retriever._store — el
        mismo objeto que usa /chat — y no una copia obtenida con un
        FAISSStore.load() aparte."""
        import src.rag.retriever as retriever

        total_antes = retriever._store.total

        resp = self._upload(client, admin_token, "regresion.txt", SMALL_TXT)

        assert resp.status_code == 200
        body = resp.json()
        assert retriever._store is wired_store
        assert retriever._store.total == total_antes + body["chunks_created"]

    def test_falla_al_guardar_revierte_el_indice(self, client, wired_store, admin_token, patched_embedder):
        total_antes = wired_store.total

        def boom(*a, **kw):
            raise RuntimeError("disco lleno")

        wired_store.save = boom

        resp = self._upload(client, admin_token, "rollback.txt", SMALL_TXT)

        assert resp.status_code == 500
        assert wired_store.total == total_antes
