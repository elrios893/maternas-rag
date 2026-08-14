"""
test_store_admin.py — Cobertura del filtro de documentos activos, la
persistencia atómica y rollback_append en FAISSStore.

Usa un índice FAISS real y chico (IndexFlatIP de 4 dims), no un mock: da
semántica real de ntotal/search/remove_ids, que es justo contra lo que
hay que testear rollback_append.

Solo se mockea el embedder. Ojo con la asimetría de import:
  - embed_documents se importa arriba en store.py -> parchear
    "src.ingestion.store.embed_documents".
  - embed_query se importa DENTRO de search() -> parchear en el origen,
    "src.ingestion.embedder.embed_query".

No se depende de la fixture autouse mock_settings de conftest.py: como
patchea el atributo en el módulo src.settings después de que store.py ya
hizo "from src.settings import settings", store.py conserva su propia
referencia al Settings real. Se parchea settings.faiss_store_path
directamente en el módulo bajo prueba.
"""

from __future__ import annotations

import pickle

import faiss
import numpy as np
import pytest

from src.ingestion.store import FAISSStore, is_active


DIM = 4


def _vec(seed: int) -> np.ndarray:
    """Vector determinístico y normalizado, de shape (1, DIM)."""
    rng = np.random.default_rng(seed)
    v = rng.random(DIM).astype(np.float32)
    v = v / np.linalg.norm(v)
    return v.reshape(1, -1)


def _make_store(n: int = 5, extra_meta: dict | None = None) -> FAISSStore:
    """Store con n vectores y metadata mínima (doc_id, text, source_dataset)."""
    index = faiss.IndexFlatIP(DIM)
    metadata = {}
    for i in range(n):
        index.add(_vec(i))
        meta = {
            "doc_id": f"doc{i}",
            "chunk_id": f"chunk{i}",
            "text": f"texto del fragmento {i}",
            "source_dataset": "upload",
            "language": "es",
        }
        if extra_meta:
            meta.update(extra_meta.get(i, {}))
        metadata[i] = meta
    return FAISSStore(index=index, metadata=metadata)


@pytest.fixture(autouse=True)
def _patch_embed_query(monkeypatch):
    """search() importa embed_query dentro del método -> parchear en el origen."""
    def fake_embed_query(text: str) -> np.ndarray:
        return _vec(hash(text) % 1000)[0]
    monkeypatch.setattr("src.ingestion.embedder.embed_query", fake_embed_query)


# ---------------------------------------------------------------------------
# is_active — el default es el guardarraíl del modo de fallo silencioso
# ---------------------------------------------------------------------------

class TestIsActive:
    def test_sin_clave_es_activo(self):
        assert is_active({}) is True

    def test_active_false(self):
        assert is_active({"active": False}) is False

    def test_active_true(self):
        assert is_active({"active": True}) is True


# ---------------------------------------------------------------------------
# update_document_status
# ---------------------------------------------------------------------------

class TestUpdateDocumentStatus:
    def test_cuenta_afectados(self):
        store = _make_store(5)
        affected = store.update_document_status("doc2", active=False)
        assert affected == 1
        assert store.metadata[2]["active"] is False

    def test_case_insensitive(self):
        store = _make_store(5)
        affected = store.update_document_status("DOC3", active=False)
        assert affected == 1
        assert store.metadata[3]["active"] is False

    def test_sin_match_no_afecta_nada(self):
        store = _make_store(5)
        affected = store.update_document_status("no-existe", active=False)
        assert affected == 0

    def test_no_altera_doc_id_canonico(self):
        store = _make_store(5)
        store.update_document_status("DOC1", active=False)
        assert store.metadata[1]["doc_id"] == "doc1"

    def test_no_escribe_a_disco(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.ingestion.store.settings.faiss_store_path", tmp_path)
        store = _make_store(5)
        store.update_document_status("doc1", active=False)
        assert not (tmp_path / "metadata.pkl").exists()

    def test_bumpea_mutation_seq_solo_si_cambio_algo(self):
        store = _make_store(5)
        seq0 = store.mutation_seq
        store.update_document_status("no-existe", active=False)
        assert store.mutation_seq == seq0
        store.update_document_status("doc1", active=False)
        assert store.mutation_seq == seq0 + 1


# ---------------------------------------------------------------------------
# search() — filtro de active, sobre-muestreo, orden de score
# ---------------------------------------------------------------------------

class TestSearchActiveFilter:
    def test_excluye_desactivados(self):
        store = _make_store(5)
        store.metadata[0]["active"] = False
        results = store.search("query", k=5)
        assert all(r["doc_id"] != "doc0" for r in results)

    def test_incluye_entradas_sin_clave_active(self):
        """El test de mayor valor de la suite: el índice actual no tiene
        la clave 'active' en ninguna entrada, y así debe seguir
        funcionando exactamente igual que hoy."""
        store = _make_store(5)
        assert all("active" not in m for m in store.metadata.values())
        results = store.search("query", k=5)
        assert len(results) == 5

    def test_devuelve_como_mucho_k(self):
        store = _make_store(20)
        results = store.search("query", k=5)
        assert len(results) <= 5

    def test_sigue_devolviendo_k_con_desactivados_en_el_top(self):
        """Valida el sobre-muestreo: aunque la mitad del índice esté
        desactivada, k resultados activos deben seguir llegando."""
        store = _make_store(20)
        deactivated = {f"doc{i}" for i in range(10)}
        for i in range(10):
            store.metadata[i]["active"] = False

        results = store.search("query", k=5)

        assert len(results) == 5
        assert all(r["doc_id"] not in deactivated for r in results)

    def test_equivalente_a_hoy_sin_nada_desactivado(self):
        """Con cero desactivados, search(k) debe seguir devolviendo
        exactamente k resultados en orden de score descendente (el
        sobre-muestreo interno no debe alterar el resultado observable)."""
        store = _make_store(10)
        results = store.search("query", k=5)
        assert len(results) == 5
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_score_real_aunque_metadata_traiga_clave_score(self):
        store = _make_store(3, extra_meta={0: {"score": "no-deberia-aparecer"}})
        results = store.search("query", k=3)
        target = next(r for r in results if r["doc_id"] == "doc0")
        assert isinstance(target["score"], float)

    def test_indice_vacio_devuelve_lista_vacia(self):
        store = FAISSStore(index=faiss.IndexFlatIP(DIM), metadata={})
        assert store.search("query", k=5) == []


# ---------------------------------------------------------------------------
# Persistencia atómica
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_round_trip(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.ingestion.store.settings.faiss_store_path", tmp_path)
        store = _make_store(5)
        store.save(embedding_model="test-model")

        assert (tmp_path / "index.faiss").exists()
        assert (tmp_path / "metadata.pkl").exists()
        assert (tmp_path / "build_info.json").exists()

        with open(tmp_path / "metadata.pkl", "rb") as f:
            loaded = pickle.load(f)
        assert loaded == store.metadata

    def test_save_no_deja_tmp(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.ingestion.store.settings.faiss_store_path", tmp_path)
        store = _make_store(5)
        store.save()
        assert not list(tmp_path.glob("*.tmp"))

    def test_save_metadata_no_reescribe_indice(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.ingestion.store.settings.faiss_store_path", tmp_path)
        store = _make_store(5)
        store.save()
        index_mtime = (tmp_path / "index.faiss").stat().st_mtime_ns

        store.update_document_status("doc1", active=False)
        store.save_metadata()

        assert (tmp_path / "index.faiss").stat().st_mtime_ns == index_mtime


# ---------------------------------------------------------------------------
# rollback_append
# ---------------------------------------------------------------------------

class TestRollbackAppend:
    def test_restaura_ntotal_y_borra_metadata(self):
        store = _make_store(5)
        start_id = store.total
        vec = _vec(99)
        store.index.add(vec)
        store.metadata[start_id] = {"doc_id": "temp", "text": "x"}

        assert store.total == 6
        store.rollback_append(start_id, 1)

        assert store.total == 5
        assert start_id not in store.metadata

    def test_lanza_si_no_es_el_ultimo_append(self):
        store = _make_store(5)
        with pytest.raises(RuntimeError):
            store.rollback_append(0, 1)

    def test_count_cero_no_hace_nada(self):
        store = _make_store(5)
        store.rollback_append(store.total, 0)
        assert store.total == 5

    def test_bumpea_mutation_seq(self):
        store = _make_store(5)
        start_id = store.total
        store.index.add(_vec(99))
        store.metadata[start_id] = {"doc_id": "temp", "text": "x"}
        seq_before = store.mutation_seq
        store.rollback_append(start_id, 1)
        assert store.mutation_seq == seq_before + 1


# ---------------------------------------------------------------------------
# add_documents — mutation_seq
# ---------------------------------------------------------------------------

class TestAddDocumentsMutationSeq:
    def test_bumpea_mutation_seq(self, monkeypatch):
        from src.ingestion.formatters import Document

        def fake_embed_documents(texts, batch_size=64, show_progress=True):
            return np.vstack([_vec(i) for i in range(len(texts))])

        monkeypatch.setattr("src.ingestion.store.embed_documents", fake_embed_documents)

        store = _make_store(0)
        seq0 = store.mutation_seq
        docs = [Document(text="hola", metadata={"doc_id": "nuevo", "source_dataset": "upload"})]
        store.add_documents(docs, show_progress=False)
        assert store.mutation_seq == seq0 + 1
