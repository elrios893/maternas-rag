from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

import src.bot.active_users as au


@pytest.fixture(autouse=True)
def isolated_registry(tmp_path, monkeypatch):
    """Redirige el registro a un archivo temporal — nunca tocar active_users.json real."""
    monkeypatch.setattr(au, "_REGISTRY_PATH", tmp_path / "active_users.json")
    yield


class TestRegister:
    def test_register_new_user_low(self):
        au.register(111, risk_level="low")
        users = au.get_all()
        assert users["111"]["risk_points"] == 0
        assert users["111"]["latest_risk_level"] == "low"
        assert "latest_risk_flags" not in users["111"]
        assert users["111"]["last_activity"]

    def test_register_new_user_high_does_not_store_flags(self):
        au.register(222, risk_level="high")
        users = au.get_all()
        assert users["222"]["risk_points"] == 10
        assert "latest_risk_flags" not in users["222"]

    def test_register_accumulates_points(self):
        au.register(333, risk_level="medium")
        au.register(333, risk_level="medium")
        users = au.get_all()
        assert users["333"]["risk_points"] == 6

    def test_register_caps_at_max_points(self):
        for _ in range(10):
            au.register(444, risk_level="high")
        users = au.get_all()
        assert users["444"]["risk_points"] == au._RISK_MAX_POINTS

    def test_register_updates_latest_level(self):
        au.register(555, risk_level="high")
        au.register(555, risk_level="low")
        users = au.get_all()
        assert users["555"]["latest_risk_level"] == "low"



class TestDecay:
    def test_decay_reduces_points_after_inactivity(self):
        au.register(777, risk_level="high")
        users = au.get_all()
        five_hours_ago = datetime.now(timezone.utc) - timedelta(hours=5)
        users["777"]["last_activity"] = five_hours_ago.isoformat(timespec="seconds")
        au._save(users)

        au.register(777, risk_level="low")
        users = au.get_all()
        assert users["777"]["risk_points"] == 5

    def test_decay_does_not_go_negative(self):
        au.register(888, risk_level="medium")
        users = au.get_all()
        far_past = datetime.now(timezone.utc) - timedelta(hours=100)
        users["888"]["last_activity"] = far_past.isoformat(timespec="seconds")
        au._save(users)

        au.register(888, risk_level="low")
        users = au.get_all()
        assert users["888"]["risk_points"] == 0


class TestMigration:
    def test_migrates_legacy_list_format(self):
        au._REGISTRY_PATH.write_text(json.dumps([111, 222]), encoding="utf-8")
        users = au.get_all()
        assert set(users.keys()) == {"111", "222"}
        assert users["111"]["risk_points"] == 0
        assert users["111"]["latest_risk_level"] == "low"

    def test_migration_persists_to_disk_encrypted(self):
        au._REGISTRY_PATH.write_text(json.dumps([333]), encoding="utf-8")
        au.get_all()
        raw = au._REGISTRY_PATH.read_bytes()
        on_disk = json.loads(au._get_fernet().decrypt(raw))
        assert isinstance(on_disk, dict)
        assert "333" in on_disk


class TestRemoveAndClear:
    def test_remove_existing_user(self):
        au.register(999, risk_level="low")
        au.remove(999)
        assert "999" not in au.get_all()

    def test_remove_nonexistent_user_is_noop(self):
        au.remove(123456)
        assert au.get_all() == {}

    def test_clear_empties_registry(self):
        au.register(1, risk_level="low")
        au.register(2, risk_level="medium")
        au.clear()
        assert au.get_all() == {}


class TestUpdateCheckSent:
    def test_update_check_sent_sets_timestamp(self):
        au.register(42, risk_level="low")
        assert au.get_all()["42"]["last_check_sent"] == ""
        au.update_check_sent(42)
        assert au.get_all()["42"]["last_check_sent"] != ""

    def test_update_check_sent_unknown_user_is_noop(self):
        au.update_check_sent(9999)
        assert au.get_all() == {}


class TestEncryptionAtRest:
    def test_file_on_disk_is_not_plaintext_json(self):
        au.register(1010, risk_level="high")
        raw = au._REGISTRY_PATH.read_bytes()
        with pytest.raises(json.JSONDecodeError):
            json.loads(raw.decode("utf-8"))

    def test_decrypts_correctly_on_read(self):
        au.register(2020, risk_level="medium")
        users = au.get_all()
        assert users["2020"]["risk_points"] == 3

    def test_legacy_plaintext_flags_are_purged_on_read(self):
        legacy = {"3030": {"risk_points": 10, "latest_risk_level": "high", "latest_risk_flags": ["hemorragia"]}}
        au._REGISTRY_PATH.write_text(json.dumps(legacy), encoding="utf-8")
        users = au.get_all()
        assert "latest_risk_flags" not in users["3030"]


class TestGetAll:
    def test_no_file_returns_empty_dict(self):
        assert au.get_all() == {}

    def test_corrupt_json_returns_empty_dict(self):
        au._REGISTRY_PATH.write_text("not valid json", encoding="utf-8")
        assert au.get_all() == {}
