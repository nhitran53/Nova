import main


def parse(*argv):
    return main.build_parser(main.WakeConfig.from_env()).parse_args(list(argv))


def test_missing_model_error_tells_user_to_drop_hey_nova_onnx(tmp_path):
    models, error = main.resolve_models(parse("--models-dir", str(tmp_path)))
    assert models is None and "drop hey_nova.onnx into" in error and str(tmp_path) in error


def test_models_dir_onnx_files_are_found(tmp_path):
    (tmp_path / "hey_nova.onnx").write_bytes(b"x")
    models, error = main.resolve_models(parse("--models-dir", str(tmp_path)))
    assert error is None and models == [str(tmp_path / "hey_nova.onnx")]


def test_explicit_missing_model_path_is_an_error(tmp_path):
    models, error = main.resolve_models(parse("--model", str(tmp_path / "nope.onnx")))
    assert models is None and "not found" in error


def test_defaults_and_mode_flags():
    args = parse()
    assert args.threshold == 0.3 and args.voice_url == "http://127.0.0.1:8765" and not args.handoff and not args.dry_run
    assert parse("--no-handoff").dry_run and parse("--handoff").handoff


def test_dry_run_with_no_model_exits_2_without_api_key(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    assert main.main(["--dry-run", "--models-dir", str(tmp_path)]) == 2
    assert "drop hey_nova.onnx" in capsys.readouterr().err
