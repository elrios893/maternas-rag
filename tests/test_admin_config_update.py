"""
test_admin_config_update.py — Cobertura de PATCH /admin/config y de
GET /admin/logs.

TestClient(app) se usa SIN el bloque `with`, igual que
tests/test_api_documents.py, para saltear el lifespan de FastAPI. El
.env real del proyecto nunca se toca: ENV_PATH se monkeypatchea hacia
un archivo temporal por test.
"""

from __future__ import annotations

import dotenv
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.settings import settings


@pytest.fixture
def client():
    return TestClient(app)   # sin "with": no dispara el lifespan


@pytest.fixture(autouse=True)
def no_real_store(monkeypatch):
    """admin_config() (llamado por GET y por el propio PATCH al final)
    intenta _get_store().build_info() — sin esto, cada test de este
    archivo cargaría el índice FAISS real de 780MB del proyecto.
    admin_config() ya tolera la excepción (build_info queda {})."""
    def _raise():
        raise RuntimeError("sin índice cargado en tests")
    monkeypatch.setattr("src.api.routes_admin._get_store", _raise)


@pytest.fixture
def admin_token(monkeypatch):
    token = "s3cr3t-test-token"
    monkeypatch.setattr("src.api.auth.settings.admin_api_token", token)
    return token


@pytest.fixture
def temp_env(tmp_path, monkeypatch):
    """.env temporal con un par de claves preexistentes, para verificar
    que el PATCH no las toca — y para no escribir jamás sobre el .env
    real del proyecto durante los tests."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "SOME_OTHER_VAR=no_tocar\nADMIN_API_TOKEN=no_debe_cambiar\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("src.api.routes_admin.ENV_PATH", env_file)
    return env_file


def _headers(token: str) -> dict:
    return {"X-Admin-Token": token}


# ---------------------------------------------------------------------------
# Auth matrix
# ---------------------------------------------------------------------------

PATCH_BODY = {"notifier_email_to": "nuevo@correo.com"}


class TestAuthMatrix:
    def test_sin_header_401(self, client, temp_env):
        resp = client.patch("/admin/config", json=PATCH_BODY)
        assert resp.status_code == 401

    def test_token_incorrecto_401(self, client, temp_env, admin_token):
        resp = client.patch("/admin/config", json=PATCH_BODY, headers=_headers("otro"))
        assert resp.status_code == 401

    def test_sin_admin_api_token_configurado_503(self, client, temp_env, monkeypatch):
        monkeypatch.setattr("src.api.auth.settings.admin_api_token", "")
        resp = client.patch("/admin/config", json=PATCH_BODY, headers=_headers("cualquiera"))
        assert resp.status_code == 503

    def test_token_correcto_200(self, client, temp_env, admin_token):
        resp = client.patch("/admin/config", json=PATCH_BODY, headers=_headers(admin_token))
        assert resp.status_code == 200

    @pytest.mark.parametrize("path,method", [
        ("/admin/logs", "get"),
        ("/admin/bot/status", "get"),
        ("/admin/bot/logs", "get"),
        ("/admin/bot/start", "post"),
        ("/admin/bot/stop", "post"),
        ("/admin/bot/restart", "post"),
    ])
    def test_bot_and_logs_endpoints_require_auth(self, client, path, method, monkeypatch):
        # start/stop/restart no deben ni intentar tocar un proceso real
        # cuando el auth falla — si lo hicieran, esto mismo lo detectaría
        # (el mock lanzaría si se llamara).
        import src.api.bot_supervisor as bs
        monkeypatch.setattr(bs, "start_bot", lambda: (_ for _ in ()).throw(AssertionError("no debió llamarse")))
        monkeypatch.setattr(bs, "stop_bot", lambda: (_ for _ in ()).throw(AssertionError("no debió llamarse")))
        monkeypatch.setattr(bs, "restart_bot", lambda: (_ for _ in ()).throw(AssertionError("no debió llamarse")))

        resp = getattr(client, method)(path)
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Allow-list / extra="forbid"
# ---------------------------------------------------------------------------

class TestAllowList:
    def test_campo_desconocido_422(self, client, temp_env, admin_token):
        resp = client.patch(
            "/admin/config",
            json={"admin_api_token": "hackeado"},
            headers=_headers(admin_token),
        )
        assert resp.status_code == 422
        # Confirma que ni siquiera llegó a tocar el archivo
        assert "no_debe_cambiar" in temp_env.read_text(encoding="utf-8")

    def test_sin_campos_400(self, client, temp_env, admin_token):
        resp = client.patch("/admin/config", json={}, headers=_headers(admin_token))
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Validación de valores
# ---------------------------------------------------------------------------

class TestValidacion:
    def test_valor_con_salto_de_linea_400(self, client, temp_env, admin_token):
        resp = client.patch(
            "/admin/config",
            json={"notifier_email_to": "a@b.com\nADMIN_API_TOKEN=hackeado"},
            headers=_headers(admin_token),
        )
        assert resp.status_code == 400
        assert "no_debe_cambiar" in temp_env.read_text(encoding="utf-8")

    @pytest.mark.parametrize("value", [0, -5])
    def test_intervalo_no_positivo_422(self, client, temp_env, admin_token, value):
        resp = client.patch(
            "/admin/config",
            json={"status_check_interval_low_seconds": value},
            headers=_headers(admin_token),
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Aplicación en caliente + persistencia
# ---------------------------------------------------------------------------

class TestAplicacion:
    def test_actualiza_settings_en_memoria(self, client, temp_env, admin_token):
        resp = client.patch(
            "/admin/config",
            json={"notifier_email_to": "cambiado@correo.com"},
            headers=_headers(admin_token),
        )
        assert resp.status_code == 200
        assert settings.notifier_email_to == "cambiado@correo.com"

    def test_persiste_en_env_sin_tocar_otras_claves(self, client, temp_env, admin_token):
        client.patch(
            "/admin/config",
            json={"notifier_email_to": "cambiado@correo.com"},
            headers=_headers(admin_token),
        )
        values = dotenv.dotenv_values(str(temp_env))
        assert values["NOTIFIER_EMAIL_TO"] == "cambiado@correo.com"
        assert values["SOME_OTHER_VAR"] == "no_tocar"
        assert values["ADMIN_API_TOKEN"] == "no_debe_cambiar"

    def test_groq_api_key_resetea_los_tres_clientes(self, client, temp_env, admin_token, monkeypatch):
        calls = []
        monkeypatch.setattr("src.rag.chain.reset_client", lambda: calls.append("chain"))
        monkeypatch.setattr("src.classifiers.intent_classifier.reset_client", lambda: calls.append("intent"))
        monkeypatch.setattr("src.classifiers.risk_detector.reset_client", lambda: calls.append("risk"))

        resp = client.patch(
            "/admin/config",
            json={"groq_api_key": "gsk_nuevo"},
            headers=_headers(admin_token),
        )
        assert resp.status_code == 200
        assert set(calls) == {"chain", "intent", "risk"}

    def test_bot_scoped_field_marca_requires_restart(self, client, temp_env, admin_token):
        resp = client.patch(
            "/admin/config",
            json={"telegram_bot_token": "nuevo-token"},
            headers=_headers(admin_token),
        )
        assert resp.status_code == 200
        assert resp.json()["requires_bot_restart"] is True

    def test_campo_no_bot_scoped_no_marca_requires_restart(self, client, temp_env, admin_token):
        resp = client.patch(
            "/admin/config",
            json={"groq_model": "llama-nuevo"},
            headers=_headers(admin_token),
        )
        assert resp.status_code == 200
        assert resp.json()["requires_bot_restart"] is False


# ---------------------------------------------------------------------------
# Nunca se exponen secretos
# ---------------------------------------------------------------------------

class TestNoExponeSecretos:
    def test_get_config_no_incluye_secretos_en_editable(self, client, admin_token):
        resp = client.get("/admin/config", headers=_headers(admin_token))
        assert resp.status_code == 200
        editable = resp.json()["editable"]
        assert "groq_api_key" not in editable
        assert "notifier_smtp_password" not in editable
        assert "telegram_bot_token" not in editable

    def test_patch_response_no_incluye_secretos(self, client, temp_env, admin_token):
        resp = client.patch(
            "/admin/config",
            json={"groq_api_key": "gsk_secreto", "notifier_smtp_password": "clave-secreta"},
            headers=_headers(admin_token),
        )
        body_text = resp.text
        assert "gsk_secreto" not in body_text
        assert "clave-secreta" not in body_text
