import pytest

from router.db_config import password_options


@pytest.fixture(autouse=True)
def clean_database_environment(monkeypatch):
    import os

    for name in os.environ:
        if name.startswith("PG"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NOVA_DB_PASSWORD", "test-secret")


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "::1"])
def test_local_passwordless_connection_uses_app_secret(host):
    assert password_options(f"host={host} dbname=nova user=cardin") == {"password": "test-secret"}


@pytest.mark.parametrize("connection", [
    "host=localhost dbname=nova user=cardin password=explicit",
    "host=example.com dbname=nova user=cardin",
    "host=localhost hostaddr=192.0.2.1 dbname=nova user=cardin",
    "host=localhost port=5433 dbname=nova user=cardin",
    "host=localhost dbname=other user=cardin",
    "host=localhost dbname=nova user=postgres",
    "service=other",
])
def test_explicit_credentials_and_other_targets_are_preserved(connection):
    assert password_options(connection) == {}


def test_pgpassword_keeps_priority(monkeypatch):
    monkeypatch.setenv("PGPASSWORD", "explicit")
    assert password_options("host=localhost dbname=nova user=cardin") == {}


def test_missing_app_secret_leaves_password_file_available(monkeypatch):
    monkeypatch.delenv("NOVA_DB_PASSWORD")
    assert password_options("host=localhost dbname=nova user=cardin") == {}
