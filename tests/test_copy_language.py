"""Hook/title language can be set apart from the audio (anime for an English
audience): the Japanese hooks burned in as boxes on 2-oct-2026."""
import pytest

main = pytest.importorskip("main")


def test_follows_the_transcript_by_default(monkeypatch):
    monkeypatch.delenv("COPY_LANGUAGE", raising=False)
    assert main.copy_language("ja") == "ja"


def test_job_override_wins(monkeypatch):
    monkeypatch.setenv("COPY_LANGUAGE", "English")
    assert main.copy_language("ja") == "English"


def test_blank_override_is_ignored(monkeypatch):
    monkeypatch.setenv("COPY_LANGUAGE", "  ")
    assert main.copy_language("ja") == "ja"
