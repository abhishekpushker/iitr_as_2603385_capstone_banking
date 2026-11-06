# Bug log

Protocol: docs/IMPLEMENTATION_PLAN.md §1.2 — reproduce with a failing test (bug id in its
docstring), fix the root cause, run the full gate. Library surprises also get a contract test.

| Id | Phase found | Symptom | Root cause | Fix commit | Regression test |
|---|---|---|---|---|---|
| BUG-001 | 2 | `SentenceTransformer('/models/all-MiniLM-L6-v2')` as uid 10001 fails with `FileNotFoundError: No such file or directory: …/model.safetensors` although the file exists | The `model` build stage runs as root and `save()` writes `model.safetensors` with mode 0600; `COPY --from=model` keeps the mode, so the runtime user cannot read the weights (transformers reports the `PermissionError` as file-not-found). Phase 0 only checked that the file exists. | `chmod -R a+rX,go-w /models` in the `model` stage (fix commit: see `fix(image)` in git log) | `tests/contract/test_image.py::test_every_model_file_is_readable_by_the_runtime_user` |
