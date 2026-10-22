"""The DB fixture's environment handling: skip locally, fail in CI."""

import pytest
from conftest import start_container


class DockerDown:
    def start(self):
        raise ConnectionError("Error while fetching server API version")


def missing_testcontainers():
    raise ImportError("No module named 'testcontainers'")


def test_docker_unavailable_skips_locally(monkeypatch):
    monkeypatch.delenv("CI", raising=False)

    with pytest.raises(pytest.skip.Exception, match="Docker unavailable"):
        start_container(DockerDown)


def test_docker_unavailable_fails_in_ci(monkeypatch):
    monkeypatch.setenv("CI", "true")

    with pytest.raises(ConnectionError):
        start_container(DockerDown)


def test_missing_testcontainers_skips_locally_and_fails_in_ci(monkeypatch):
    monkeypatch.delenv("CI", raising=False)
    with pytest.raises(pytest.skip.Exception, match="testcontainers not installed"):
        start_container(missing_testcontainers)

    monkeypatch.setenv("CI", "true")
    with pytest.raises(ImportError):
        start_container(missing_testcontainers)


def test_a_started_container_is_returned(monkeypatch):
    monkeypatch.delenv("CI", raising=False)

    class Up:
        started = False

        def start(self):
            self.started = True

    container = start_container(Up)

    assert container.started
