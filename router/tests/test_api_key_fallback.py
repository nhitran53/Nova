"""STT/TTS resolve local then repo-root credentials, using temporary files only."""
import pytest

from router import stt, tts


@pytest.mark.parametrize("module", [stt, tts], ids=["stt", "tts"])
@pytest.mark.parametrize("source", ["root", "local", "environment"])
def test_api_key_file_precedence(tmp_path, monkeypatch, module, source):
    local = tmp_path / "router" / ".env"
    local.parent.mkdir()
    (tmp_path / ".env").write_text("ELEVENLABS_API_KEY=root-test-key\n")
    if source == "local":
        local.write_text("ELEVENLABS_API_KEY=local-test-key\n")
    monkeypatch.setattr(module, "ENV_PATH", local)
    monkeypatch.delenv(module.KEY_VAR, raising=False)
    if source == "environment":
        monkeypatch.setenv(module.KEY_VAR, "environment-test-key")
    expected = source + "-test-key"
    if module is stt:
        assert stt._api_key() == expected
    else:
        module.reset()
        monkeypatch.setattr(tts, "_client_factory", lambda key: {"key": key})
        try:
            assert tts.init() == {"key": expected}
        finally:
            tts.reset()
    # load_dotenv mutates os.environ; monkeypatch restores the original value at teardown.
