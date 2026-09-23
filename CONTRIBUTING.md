# Contributing

Small, measured changes welcome. This repo is a runtime, not a research log: keep patches focused and the numbers honest.

## Scope

In scope: the MLX encoder and Metal kernels (`laya_mlx.py`), the ANE export/runtime and router (`ane/`, `laya_fast.py`), the request API (`laya_api.py`), benchmarks, and docs. Out of scope: retraining the model, support for non-Apple-Silicon platforms, and anything that needs the weights checked in.

## The gate

`bash verify.sh` is the fail-closed gate. It runs the decision-parity check first, then times the deterministic fixtures. If parity fails, the run exits non-zero and no timing is reported — a faster wrong answer does not count.

- Run it before and after your change, on the same machine and config.
- `LAYA_MODEL`, `LAYA_BENCH_DTYPE`, and `LAYA_AGENT` (`fast` router vs `mlx` GPU-only) select the configuration under test. Report which you used.
- Correctness before speed: a kernel or routing change that shifts any decision is a regression, not a trade-off, unless the PR explicitly argues otherwise and shows the parity diff.

## Submitting a kernel or router change

1. Fork, set up per the README (venv, `hf download`, `convert.py`).
2. Make the change. Keep it small enough to review in one sitting.
3. Run `bash verify.sh` and paste the METRIC lines for before/after.
4. In the PR, state: device (chip, GPU cores, RAM), dtype, model directory, and which fixture moved. "Faster on my machine" without the fixture table is not evidence.
5. A standalone microbenchmark is not sufficient — only an in-context A/B through the gate counts. [`docs/benchmarks.md`](docs/benchmarks.md) lists approaches already measured; check it before proposing one of them.

## Never commit

- Model weights (`source/`, `converted*/`, `*.safetensors`) — download and convert locally.
- Compiled ANE bodies (`ane/body*/model.mlmodelc`, `*.mlpackage`) — ~700 MB each, rebuilt by `ane/export.py`.
- Anything with local absolute paths, tokens, or credentials.

`.gitignore` already covers these; if a generated artifact slips past it, fix the ignore rather than committing the file.

## Setup

See the README for environment setup, the fixture/benchmark table, and the map of which file owns which seam.
