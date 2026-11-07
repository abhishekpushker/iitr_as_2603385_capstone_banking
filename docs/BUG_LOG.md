# Bug log

Protocol: docs/IMPLEMENTATION_PLAN.md §1.2 — reproduce with a failing test (bug id in its
docstring), fix the root cause, run the full gate. Library surprises also get a contract test.

| Id | Phase found | Symptom | Root cause | Fix commit | Regression test |
|---|---|---|---|---|---|
| BUG-001 | 2 | `SentenceTransformer('/models/all-MiniLM-L6-v2')` as uid 10001 fails with `FileNotFoundError: No such file or directory: …/model.safetensors` although the file exists | The `model` build stage runs as root and `save()` writes `model.safetensors` with mode 0600; `COPY --from=model` keeps the mode, so the runtime user cannot read the weights (transformers reports the `PermissionError` as file-not-found). Phase 0 only checked that the file exists. | `chmod -R a+rX,go-w /models` in the `model` stage 41d840a | `tests/contract/test_image.py::test_every_model_file_is_readable_by_the_runtime_user` |
| BUG-002 | 4 | On a fresh `make gate`, step 3 (unit + contract) failed: `tests/unit/test_transcript_kit.py::test_cli_writes_every_registered_transcript` raised `ConnectionError: chroma at chroma:8000 unreachable after 10 attempts`. It had passed in earlier runs. | Hidden coupling to environment state. The test runs every transcript generator, and since Phase 3 transcripts 03–06 query Chroma. In the unit layer it passed only while a previous make target (`bless`, `calibrate`) had left `chroma` running; the gate starts services only from step 4. | Moved to `tests/integration/test_transcripts_cli.py`, which runs `indexer.main` first so a cold volume is indexed. b7d5213 | `tests/integration/test_transcripts_cli.py::test_cli_writes_every_registered_transcript` |
