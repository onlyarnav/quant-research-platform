# MASTER PROMPT 2: Clean, Verify, Fix and Make the Platform Deployable on AWS

Attach the repository `quant-research-platform` as the workspace (only this repository; `quant-ai-platform` is an older superseded copy and must not be touched). **Do not ask questions.** Every decision is made below. Anything you cannot execute in your environment (no Docker, no network, no AWS, no Terraform) is reported as **NOT EXECUTED (reason)**, and the matching steps go into the user's own checklist (Part H). Never claim a check passed unless you ran it and saw it pass.

---

# PART A. MISSION, RULES AND SAFETY

## A.1 Mission

The platform (data pipeline → features → ML → signals → backtesting → portfolio → analytics → FastAPI + Streamlit dashboard → orchestration) is functionally built. Make it **clean, proven and deployable for real users on AWS**:

1. Clean the entire codebase from start to end: remove redundant, dead and duplicated code and files, without changing behaviour.
2. Test everything that needs testing before deployment: run **every script, every documented command**, the full test suite, real services, containers, security, performance and failure scenarios.
3. Fix every issue found, with a regression test for each fix.
4. Make the changes needed for smooth, secure deployment on AWS and produce the deployment artifacts (Docker, compose, reverse proxy, Terraform, scripts).
5. Produce a step-by-step A-to-Z checklist so the user can verify everything personally, plus a one-command local verifier and a remote smoke test.

## A.2 Scope

Everything is in scope: `src/`, `scripts/`, `tests/`, `config/`, `alembic/`, `docs/`, root files, Docker and CI files. You may edit any module in this task (earlier role boundaries are lifted), but changes must be **behaviour-preserving** unless a defect is proven by a test.

## A.3 Non-negotiable safety rules

1. **Protect user data.** Before touching anything, copy `data/`, `models/`, `mlflow.db` (and `mlruns/` if present) and `.env` to a timestamped folder **outside the repository** (for example `..\_backup_<timestamp>\`). Never delete, overwrite or commit `data/`, `models/`, `.env`, `mlflow.db`. Run destructive or write-heavy tests only against temp folders (`tmp_path`, a temp data root) or a scratch database, never against the user's real data or database.
2. **Git discipline.** Create branch `release/pre-deploy`. Commit after every phase and after each batch of cleanup with clear messages. Never rewrite history, never force push, never run `git clean -fdx` or `git reset --hard` on the user's working tree. Tag the starting commit `pre-cleanup-baseline` first.
3. **Secrets.** Never print, log, commit or paste secret values (passwords, API keys, tokens). When scanning for secrets, report only file, line and secret type. If a real secret is found in tracked files or history, report it as CRITICAL and tell the user to rotate it; do not try to rewrite history.
4. **Behaviour preservation.** Do not change financial formulas, schemas (`Data_Schema.md` is authoritative), parquet column names, file naming, database tables or public function signatures unless a failing test proves a defect. Existing data files and trained models must keep working. If a breaking change is truly unavoidable, document a migration note.
5. **Remove only when proven unused** (Part D rules). When in doubt, keep and list it under "Deferred".
6. **Fix loop discipline.** For every failure: find the root cause, apply the smallest correct fix, add a regression test, re-run the affected checks. Do not retry the same failing approach more than three times; document the blocker and continue.
7. **No fabricated results.** Every number in your reports comes from a command you ran. Label real-data vs synthetic-data runs.

## A.4 Engineering rules for any code you write or touch

Python 3.12; `from __future__ import annotations`; type hints; Google docstrings; `get_logger(__name__)` with lazy `%s` (no f-strings in logger calls, no `import logging` for loggers); settings only through `config.settings.settings`; no `os.getenv`, hard-coded paths, URLs or rates; files ≤ 300 lines; no `TODO` or placeholder code; no `print` outside `scripts/`; black (88) and isort (profile black); one-directional dependencies `data_pipeline → data_storage → feature_engineering → models → signals → backtesting → portfolio → analytics → dashboard` (`infra` and `scripts` may import everything; nothing imports `scripts` or `tests`).

---

# PART B. ENVIRONMENT BOOTSTRAP AND BASELINE

1. **Python:** try `.\.venv\Scripts\python.exe`, `.\venv\Scripts\python.exe`, `py -3.12`, `python`. Record `python --version`, OS, `pip list` (to a file in a temp folder, not the repo). If Python cannot be executed, stop and report that clearly.
2. **Tooling:** install into the project venv (dev only): `black isort flake8 mypy bandit pip-audit vulture deptry pytest-cov pytest-timeout pytest-randomly pytest-xdist interrogate pip-licenses pre-commit playwright httpx` (skip any that cannot install; note it). Do not add dev tools to production dependencies.
3. **Services:** detect Docker (`docker version`). If available, start the repository's existing compose services (PostgreSQL and MLflow) on a **scratch database** (for example `quantdb_verify`); never run tests or migrations against the user's working database `quantdb`. If Docker is unavailable, DB-marked tests are reported NOT EXECUTED.
4. **Baseline:** run `python -m pytest tests -q` (non-DB first, then everything if the scratch DB is up) and record pass/fail/skip counts, runtime, slowest 25 tests (`--durations=25`), and coverage (`--cov=src --cov-report=term-missing`). Record `git status` and `git log -5`. This baseline is the reference; no previously passing test may regress (except deliberate removal of tests for code you proved dead).
5. Print a short inventory table: top-level folders, number of Python files and lines per package, number of tests per folder, list of scripts, Docker/CI/Terraform files present.

---

# PART C. PHASE 1: AUDIT AND STATIC ANALYSIS (read-only; report findings, do not fix yet)

Run each tool and save raw outputs to a temp folder; summarise in a findings table (tool, finding, file:line, severity, action).

| Tool | Command (adapt paths) | What to extract |
|---|---|---|
| black/isort | `black --check`, `isort --check --profile black` | formatting drift |
| flake8 | `flake8 src tests scripts config` | unused imports/variables, undefined names |
| vulture | `vulture src scripts --min-confidence 80` plus a 60% pass for review | dead functions/classes/variables |
| deptry | `deptry .` | unused, missing and transitive dependencies |
| pip check / pipdeptree | `pip check` | broken requirements |
| pip-audit | `pip-audit` | known vulnerable dependencies (CVE ids) |
| bandit | `bandit -r src scripts -ll` | security smells (SQL strings, subprocess, hard-coded secrets, assert use, weak crypto, `yaml.load`, `pickle`, `joblib.load` on untrusted data) |
| mypy | `mypy src` (non-strict) | type errors; fix real ones, ignore noise with justification |
| interrogate | `interrogate -v src` | docstring coverage |
| pylint duplicate-code | `pylint --disable=all --enable=duplicate-code src` (or `jscpd` if available) | copy-paste blocks |
| radon | `radon cc src -nc` and `radon mi src -nb` | very complex or unmaintainable functions |
| secret scan | `gitleaks` or `detect-secrets` if available; else regex scan for key-like strings, `password=`, `AKIA[0-9A-Z]{16}`, private keys, in the working tree **and** `git log -p` (report counts and locations only) | leaked secrets |
| licences | `pip-licenses --format=markdown` | third-party licence list for `docs/THIRD_PARTY_NOTICES.md` |
| git hygiene | `git ls-files` | tracked files that should not be: `.env`, `data/`, `models/`, `*.db`, `__pycache__`, `.pytest_cache`, logs, notebooks with outputs, large files (> 1 MB), duplicate or backup files (`*_old.py`, `*_copy.py`, `*.bak`, `*.orig`) |

Also manually audit (read the code) for: `logging.getLogger` use instead of `get_logger`, f-string logging, `print` in `src/`, bare `except:` or `except Exception: pass`, mutable default arguments, `datetime.utcnow()` (use timezone-aware), `==`-comparisons on floats in production code, hard-coded absolute or Windows-specific paths (`D:\`, backslashes), `os.getcwd()`-dependent paths, `assert` used for runtime validation in `src/`, `time.sleep` in non-test code, unclosed files/sessions, SQL built with string concatenation, hard-coded ports/URLs, disabled TLS verification, `debug=True`/`reload=True` defaults, and wildcard CORS with credentials.

---

# PART D. PHASE 2: CLEAN-UP (behaviour-preserving)

Apply in batches (one theme per batch); after each batch run `black`, `isort`, `flake8` and the non-DB test suite, then commit.

## D.1 Removal rules

Delete code only when **all** hold: (a) no reference in `src/`, `scripts/`, `tests/` (other than tests that exist solely for that code), `docs/`, `README`, Dockerfiles, compose files, CI, Terraform, `pyproject.toml` entry points; (b) not reachable dynamically: check string-based use (model registry keys, allocator names, `getattr`, `importlib`, Streamlit view dispatch, API route registration, settings names used from `.env`); (c) not an advertised extension point in docs (Protocols, registries); (d) vulture confidence ≥ 80% **and** a manual grep confirms. Anything else goes to "Deferred (needs owner decision)".

## D.2 What to clean (everything, start to end)

1. **Files:** remove stray scratch scripts, backups/duplicates, empty modules, unused notebooks outputs, committed caches, logs, local DB files, generated artifacts; add them to `.gitignore`. Ensure tracked: source, tests, docs, configs, migrations, Docker/Terraform files only. If a large or private file is tracked (data, models), `git rm --cached` it and add it to `.gitignore` (do not delete from disk).
2. **Code:** unused imports/variables/functions/classes/parameters; commented-out code; dead branches; unreachable code; duplicate helpers (consolidate into one shared function with tests; for example the two backtesting simulators must share fill/validation helpers instead of copy-pasting; keep both simulators and the legacy engine because tests and parity checks depend on them, but remove any duplicated block ≥ 10 lines); inconsistent patterns (`logging.getLogger` → `get_logger`; f-string logs → lazy; `print` in `src/` → logger); broad exception swallowing (log and re-raise or handle specifically); magic numbers → settings; inconsistent naming; stale docstrings; stale TODOs (resolve or delete).
3. **Size and structure:** split files > 300 lines; remove empty `__init__` re-exports that cause circular imports; ensure every package has a sane `__init__.py`; verify no circular imports by importing every module in a fresh interpreter (`python -c "import pkgutil, importlib; ..."` over `src`).
4. **Dependencies:** make `pyproject.toml` accurate: remove unused dependencies (confirmed by deptry and grep), add missing ones, move test/dev tools to `[project.optional-dependencies] dev`, set sensible lower bounds, pin Python `>=3.12,<3.13`. Generate a **lock/constraints file** from the verified environment (`requirements/constraints.txt` via `pip freeze` filtered to the production dependency closure, or `pip-compile` if available) used by the Dockerfile for reproducible builds. Remove optional packages never imported in production code (for example report-only or experimental libraries) from the production set.
5. **Config:** one place for settings; delete settings that nothing reads; make sure `.env.example` lists exactly the settings that exist (every key, comment, safe placeholder, no real values) and add `.env.production.example` (Part G).
6. **Tests:** remove duplicated or obsolete tests of deleted code; fix tests that depend on test order, time, network, real `data/`, or the user's DB; replace `sleep` with deterministic waits; add `pytest-timeout` (default 120 s, `timeout_method = "thread"` on Windows) to prevent hangs; register markers (`db`, `integration`, `slow`, `e2e`); document marker usage in `pyproject.toml`.
7. **Git hygiene files:** complete `.gitignore` (venvs, caches, data, models, `*.db`, `mlruns`, logs, `.env*` except examples, IDE files, Terraform state `*.tfstate*`, `.terraform/`, `terraform.tfvars`); `.gitattributes` with `* text=auto` and `*.sh text eol=lf`, `Dockerfile text eol=lf`, `*.ps1 text eol=crlf` (Linux containers must never see CRLF shell scripts); `.dockerignore`; `.editorconfig`.
8. **Docs:** delete obsolete docs, fix wrong ones; README must describe the final state only.
9. **Report:** produce `docs/CLEANUP_REPORT.md`: before/after counts (files, lines, dependencies, flake8/vulture findings), every removal with its evidence (grep result summary), every consolidation, every deferred item with the reason.

---

# PART E. PHASE 3: PRODUCTION-READINESS CODE CHANGES

Implement these (each with unit tests); they are needed for a public, multi-user, containerised deployment.

1. **Configuration for containers:** every path and URL from settings/env; `DATABASE_URL` supports `?sslmode=require`; add settings `APP_ENV` (`dev|staging|prod`, default `dev`), `LOG_LEVEL` (default `INFO`), `LOG_FORMAT` (`text|json`, default `text`), `API_ROOT_PATH` (default empty), `API_WORKERS` (default 1), `API_CORS_ORIGINS` (comma list, default empty = same-origin), `DASHBOARD_READ_ONLY` (default `false`), `DASHBOARD_PASSWORD` (optional), `TZ` handling documented. In `prod`, refuse to start if `DASHBOARD_API_KEY` is empty while `DASHBOARD_READ_ONLY=false` (clear error), and never enable `reload`/debug.
2. **Logging:** `LOG_FORMAT=json` produces one JSON object per line (timestamp UTC ISO, level, logger, message, request_id when available) so CloudWatch can filter; stdout only (containers). Add API request-ID middleware (`X-Request-ID` generated or propagated, included in logs and response headers) and an access log line per request (method, path, status, duration). Mask secrets and passwords in any logged URL or exception text. (If `src/utils/logger.py` must change, keep `get_logger` API unchanged.)
3. **API hardening:** `root_path` support (`API_ROOT_PATH`), `--proxy-headers` and forwarded-IP trust configurable; explicit CORS from `API_CORS_ORIGINS`; request body size limit for `POST /backtests`; consistent JSON errors; security headers middleware (`X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer`, `X-Frame-Options: DENY`); `/health` (liveness, cheap) and `/ready` (readiness: DB reachable, data folders readable, migrations at head); `docs`/`openapi` can be disabled with `API_DOCS_ENABLED` (default true in dev, false in prod).
4. **Job registry persistence:** backtest jobs are persisted atomically as JSON under `settings.BACKTEST_RESULTS_PATH/_jobs/` so they survive restarts and work with more than one worker; stale `running` jobs older than a timeout are marked `failed` on startup; cap concurrent jobs (setting `BACKTEST_MAX_CONCURRENT_JOBS`, default 1) and return 429 when saturated.
5. **Public-safe mode:** `DASHBOARD_READ_ONLY=true` hides the "Run Backtest" view and makes `POST /backtests` return 403; no endpoint accepts file paths from clients; run ids and names are validated everywhere against `^[A-Za-z0-9_.-]+$`.
6. **Dashboard access control:** optional password gate in Streamlit (`DASHBOARD_PASSWORD`; constant-time compare; session state; no password in logs or URLs); the API key stays server-side (the dashboard container calls the API with it; browsers never see it). Add `.streamlit/config.toml`: `server.headless = true`, `server.enableXsrfProtection = true`, `server.enableCORS = true`, `browser.gatherUsageStats = false`, `server.maxUploadSize` small, `client.toolbarMode = "minimal"`.
7. **Legal and risk disclaimer:** a persistent banner on every dashboard page and an API `/disclaimer` response: "Research and educational use only. Not investment advice. Not a SEBI-registered investment adviser or research analyst. Backtests are simulations with simplifying assumptions; past performance does not indicate future results. Market data is obtained from Yahoo Finance via yfinance and may be delayed, incomplete or inaccurate." Add `DISCLAIMER.md`. Add a documented note about third-party data terms (see Part G.10).
8. **Graceful shutdown and robustness:** scheduler and API handle SIGTERM (finish current step or stop cleanly within the container stop timeout); orchestrator and scheduler never exit on a step failure; database engine uses `pool_pre_ping`; API returns 503 (not 500) when the database or data is unavailable; every external call has a timeout.
9. **Data contract validator** (`src/infra/contract_validator.py` and `scripts/validate_data_contracts.py`): validates every Parquet file in every data-lake tier against `Data_Schema.md` (required columns, dtypes, no duplicate `(symbol, date)`, sorted dates, consistent timezone, no negative volume, `high ≥ low`, signals in {−1, 0, 1}, predictions columns, features without target leakage columns in prediction inputs) and the trade log / portfolio history schemas of stored backtest runs; outputs a PASS/FAIL table and exit code (0 ok, 1 violations); `--fix` is not offered. Include tests with deliberately corrupted fixtures.
10. **`scripts/verify_all.py`** (one-command local verifier): runs, in order and with a clear PASS/FAIL/SKIP table and exit code, the gates of Part I that are runnable locally: format/lint, unit tests (non-DB), DB tests (if reachable), migrations on a scratch DB, contract validation on a synthetic lake, API smoke (in-process), dashboard AppTest smoke, script `--help` checks, deploy-artifact tests, optional `--docker` (build and compose smoke) and `--e2e` (Playwright). Options: `--fast`, `--docker`, `--e2e`, `--json report.json`. It must never touch the user's real data or database (it builds its own temp lake and scratch DB name).
11. **`scripts/smoke_test.py --base-url https://host --api-key ... --user ... --password ...`** (post-deployment test, uses `httpx`, no repository imports): checks `/api/health`, `/api/ready`, `/api/runs`, one run's metrics and equity, `/api/signals/latest` (accepts 503 only if explicitly allowed), the dashboard root returns 200 and `_stcore/health` returns `ok`, HTTPS certificate validity and expiry (> 14 days), security headers, that unauthenticated access is rejected when auth is configured, that `docs` are disabled in prod, response-time budget (p95 under 2 s over 20 requests). Prints a PASS/FAIL table and exits non-zero on failure.
12. **Time zone:** all timestamps stored/logged in UTC or explicit offsets; the scheduler uses `PIPELINE_TIMEZONE` (default `Asia/Kolkata`) regardless of the host zone; the container image installs `tzdata`.

---

# PART F. PHASE 4: VERIFICATION MATRIX (run everything yourself)

Record each check as: ID, command, expected, actual, PASS/FAIL/NOT EXECUTED, evidence file. Fix failures (Part I) and re-run until green or documented.

## F.1 Fresh install and imports
1. Fresh clone: `git clone . <temp>` (or `git worktree`) into a temp folder; create a new venv; `pip install -e ".[dev]"` using the constraints file; `pip check`; import every module under `src` and `scripts` helper modules in a clean interpreter; no import errors, no circular imports, no side effects at import time (importing must not create folders, open DB connections or read `.env`-dependent state beyond settings).
2. Production-only install (`pip install .` without dev extras) in another temp venv; start the API app factory and import the Streamlit app module; confirm no dev-only imports are required.

## F.2 Static, security and dependency gates
3. black, isort, flake8 clean; mypy has no new errors (record the count); bandit has no high/medium findings left unexplained; pip-audit has no critical/high vulnerabilities (upgrade or document each); deptry clean; secret scan clean; `git ls-files` hygiene clean.

## F.3 Automated tests
4. Full suite with coverage: report total and per-package coverage; add tests for any core module below 80% line coverage that holds business logic (backtesting, portfolio, analytics, models, signals, repositories, API services). Target overall ≥ 85%; if not reached, list the gaps.
5. Reliability: run the suite **three times** (twice in random order with `pytest-randomly`, once with `-n auto` if xdist is available); any flaky or order-dependent test is fixed or quarantined with a marker and documented.
6. Run the suite **inside a Linux container** (`docker run --rm -v <repo>:/app -w /app python:3.12-slim ...` with the same install) to catch Windows-only assumptions (paths, case sensitivity, line endings, `\` separators, missing system libs such as `libgomp1` for LightGBM/XGBoost). Fix every difference.

## F.4 Database
7. On a scratch database: `alembic upgrade head` from empty; exactly one head; `alembic downgrade base` then `upgrade head` again (reversibility); `alembic check` reports no pending model changes; verify via SQL that all six tables, unique constraints and indexes exist as documented; run repository smoke operations; run the `db` tests.
8. Backup/restore drill (see G.7): `pg_dump` the scratch DB, restore into a second scratch DB, compare row counts and a checksum of key tables.

## F.5 Every script, every command
9. For **every file in `scripts/`** run `--help` (exit 0, usage text, no traceback), then a functional run with realistic arguments, then at least one invalid-argument run (exit code 2, no traceback). Cover: ingestion, features, training (each model type), walk-forward, prediction, signals, retraining, backtest, analytics, daily pipeline (`--dry-run` and a real run), scheduler (build and start/stop in a subprocess with a short grace period), API runner, dashboard runner, health check, contract validator, verify_all, smoke_test (against the local API). Use real data if the repository has it and the network is reachable (ingestion limited to 2–3 symbols and a short range); otherwise a synthetic lake built in a temp folder (keep a reusable generator in `tests/` helpers or `scripts/make_synthetic_lake.py`). Produce a table: script, arguments, exit code, duration, key output line, result.
10. **Run every command written in `README.md`, `docs/runbook.md`, `docs/user_guide.md` and `.env.example` comments** verbatim (adapting only machine-specific paths). Fix the document or the code until each command works as written.

## F.6 Domain correctness audits (run on real or synthetic data)
11. **Pipeline integrity:** raw → processed → features → predictions → signals → backtest: row counts and date ranges are consistent; no lookahead (features at row t unchanged when future rows are removed; targets excluded from model inputs); signal domain {−1, 0, 1}; predictions only for the intended out-of-sample window; timezone-aware dates preserved end to end (compare a symbol's dates in processed vs predictions vs signals vs trades).
12. **Backtest invariants** over all allocation methods × rebalance frequencies × {long-only, long/short where allowed} × lags {0, 1} × stops on/off (grid on a synthetic panel): `equity = cash + long − short` every row; cash never negative; drawdown ≤ 0 and equals `(value − running max)/running max`; cumulative return consistent with first and last value; no trade with non-positive prices; trade count equals closed positions; costs strictly lower final equity versus zero-cost runs; allocator weights respect caps; legacy-vs-weighted parity on legacy-compatible configs; results deterministic across two runs.
13. **Analytics sanity:** metrics finite or documented NaN; Sharpe/Sortino/Calmar/VaR/CVaR recomputed independently with plain NumPy in the test and compared; benchmark identities (identical series → beta 1, alpha 0).
14. **ML sanity:** every model type trains on a small dataset, predicts the right shape, saves/loads identically, logs to MLflow (SQLite backend) with the expected parameters, metrics and artifacts; walk-forward folds do not overlap and respect the gap.

## F.7 API (real server)
15. Start a real Uvicorn process on a free port (subprocess, `sys.executable`), wait for `/health`, then exercise **every endpoint** with `httpx`: happy paths, 404, 422, pagination limits, NaN→null, auth (401 without key, 200 with key when configured), read-only mode (403 on `POST /backtests`), CORS behaviour, security headers, request-id header, `API_ROOT_PATH`, docs disabled in prod mode, OpenAPI JSON valid. Concurrency: 50 parallel requests across endpoints (no 5xx, p95 latency recorded). Kill the DB or point to a bad `DATABASE_URL` and verify `/ready` fails and data endpoints return 503 cleanly. Stop with SIGTERM and verify a clean exit.

## F.8 Frontend (real browser)
16. `AppTest` for every page (empty state, normal state, error state), plus a **real Streamlit process**: start headless, verify `/_stcore/health` returns `ok`, and drive it with **Playwright (Chromium)**: open each view through the sidebar, wait for content, assert no visible "Traceback", "Error" or "Exception" text and no browser console errors, assert the disclaimer banner is present, test the password gate (wrong password rejected, correct accepted), test the read-only mode hides "Run Backtest", test responsive widths 1920, 1366 and 390 px (no horizontal overflow, content visible), and save screenshots of every page to `docs/screenshots/` (useful for the project report). If Playwright cannot run, say NOT EXECUTED and keep the AppTest results.

## F.9 Orchestration and resilience
17. Real `run_daily_pipeline` on the synthetic lake: all steps succeed, report saved, second run is idempotent. Failure injection: make one step fail at each position (raise from a stub, remove a data folder, corrupt a Parquet file, stop the DB, make MLflow unreachable): the orchestrator never crashes, later steps are skipped with the documented message, the exit code is 1, health check reports degraded/down accurately, and the next healthy run recovers. Scheduler: build, verify the next fire time equals the configured IST time, start/stop cleanly.
18. Resource and scale test: generate a **NIFTY-500-scale synthetic lake** (≈ 500 symbols × 2,500 business days) in a temp folder; measure wall time and peak memory (`psutil`) for features, one model fit (LightGBM), prediction, signals, a weighted backtest, analytics, report build, API latency on the large run, dashboard Overview render. Targets: features < 10 min, backtest < 10 min, report build < 60 s, API responses p95 < 1 s once cached, dashboard first paint < 8 s, peak memory < 6 GB for any single step. If a target is missed, profile (`cProfile`/`line_profiler`), optimise the hot spot (vectorise, avoid per-row Python loops, cache), re-measure, and keep tests proving unchanged results. Record the numbers in `docs/VERIFICATION_REPORT.md`; state the recommended EC2 size from the measurements.

## F.10 Containers (Linux)
19. Build the image; check: runs as non-root, no secrets in layers (`docker history`), image size recorded, `python -c "import lightgbm, xgboost, pyarrow, psycopg2"` works, `tzdata` present, healthcheck works. Run `hadolint` on the Dockerfile if available.
20. `docker compose -f docker-compose.prod.yml config` validates; bring up the **whole production stack locally** (postgres, mlflow, migrate, api, dashboard, scheduler, proxy) with a test `.env` (fake secrets) and a self-signed or HTTP-only proxy profile; wait until all healthchecks are healthy; run `scripts/smoke_test.py` against it; confirm migrations ran once and are idempotent; PostgreSQL and MLflow ports are **not** published to the host; restart all services (`down` then `up` without `-v`) and confirm data persisted; `docker compose logs` contains no tracebacks; `docker stats` snapshot recorded; `docker compose down -v` cleanup of the test stack only.

## F.11 Security checks
21. Verify and record: no secret in the image or repository; `.env*` ignored; API rejects missing/invalid key when configured; run id and path parameters reject traversal (`../`, absolute paths, encoded variants); SQL access uses bound parameters only (grep for string-built SQL); no `eval/exec/pickle.loads` on user input; `joblib.load` only on models from the trusted model store path; error bodies never include stack traces or secrets; security headers present at the proxy; dashboard auth gate cannot be bypassed by direct URL; TLS only (HTTP redirects to HTTPS in prod proxy config); dependency audit clean; containers drop capabilities and use `no-new-privileges`; Postgres only on the internal network; file permissions of generated `.env` are 600.

## F.12 Deployment artifact tests
22. `tests/deploy/test_deploy_artifacts.py` (runs without Docker/AWS): parse `docker-compose.prod.yml` (no `latest` image tags, no host-published DB/MLflow ports, healthchecks on every service, restart policies, resource limits, `no-new-privileges`, secrets only via env_file or Docker secrets, logging driver/rotation set); the Caddyfile or Nginx config exists and contains the required routes and security headers; shell scripts start with `#!/usr/bin/env bash` and `set -euo pipefail` and are LF-only (run `shellcheck` if available); Terraform files contain no hard-coded secrets (regex) and every variable has a description; `.env.production.example` contains every key the prod compose requires and no real values; Dockerfile has a non-root `USER`, no `ADD` from URLs, pinned base image; PowerShell scripts parse (`powershell -NoProfile -Command "[scriptblock]::Create((Get-Content ...))"` where available).

---

# PART G. PHASE 5: AWS DEPLOYMENT ENGINEERING

Goal: a safe, low-cost, reproducible production deployment a student can operate: **one EC2 instance running Docker Compose behind an HTTPS reverse proxy, with EBS for state, S3 for backups, SSM Parameter Store for secrets, CloudWatch for logs and alarms, images in ECR.** Document, but do not build, the alternative (ECS Fargate + RDS + EFS + ALB) in `docs/DEPLOYMENT_AWS.md` ("Scaling up").

## G.1 Production runtime (files to create)

`Dockerfile` (multi-stage, `python:3.12-slim-bookworm` pinned by digest or exact tag): builder stage installs from the constraints file; runtime stage installs `libgomp1 tzdata curl` only, copies the venv/site-packages and the app, creates user `quant` (uid 10001), `ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TZ=Asia/Kolkata`, `HEALTHCHECK` using `curl -fsS http://127.0.0.1:${PORT}/health` for the API (override per service in compose), `ENTRYPOINT` via a small `docker/entrypoint.sh` (LF) that supports commands `api`, `dashboard`, `scheduler`, `migrate`, `shell`. API command: `uvicorn src.dashboard.api.main:app --host 0.0.0.0 --port 8000 --workers ${API_WORKERS} --proxy-headers --forwarded-allow-ips='*'` (the proxy is the only thing that can reach it on the internal network). Dashboard command: `streamlit run src/dashboard/app.py --server.port 8501 --server.address 0.0.0.0`. Scheduler: `python scripts/run_scheduler.py`. Migrate: `alembic upgrade head`.

`docker-compose.prod.yml` (standalone; independent of the dev compose files):
- services: `postgres` (image pinned, volume `/opt/quant/pgdata`, `POSTGRES_PASSWORD` from env_file, healthcheck `pg_isready`, creates both `quantdb` and `mlflowdb` via an init script), `mlflow` (pinned image or same app image with `mlflow server`, Postgres backend `mlflowdb`, artifacts on `/opt/quant/mlflow-artifacts`, internal only), `migrate` (one-shot, `depends_on postgres healthy`, `restart: "no"`), `api`, `dashboard` (`DASHBOARD_API_URL=http://api:8000`), `scheduler`, `proxy` (Caddy; only service publishing ports 80/443).
- Every app service: `image: ${ECR_REPO}:${IMAGE_TAG}` (never `latest`), `env_file: /opt/quant/.env`, `read_only: true` with `tmpfs: /tmp` and writable volumes only for `/app/data`, `/app/models`, `/app/logs`, `user: "10001:10001"`, `cap_drop: [ALL]`, `security_opt: [no-new-privileges:true]`, `restart: unless-stopped`, `healthcheck`, `depends_on` with `condition: service_healthy` (and `service_completed_successfully` for `migrate`), `mem_limit`/`cpus` sized from F.9 measurements, logging `driver: awslogs` (options `awslogs-region`, `awslogs-group`, `tag`) with a fallback `json-file` profile (`max-size 10m`, `max-file 5`) for non-AWS runs.
- Internal network only; volumes bind-mounted under `/opt/quant/{data,models,pgdata,mlflow-artifacts,logs,caddy}`.

`deploy/aws/Caddyfile`: `{$DOMAIN}` site (HTTPS automatic via Let's Encrypt; support `<elastic-ip-with-dashes>.sslip.io` as a domain when the user has none), `encode zstd gzip`, security headers (HSTS, `X-Content-Type-Options`, `X-Frame-Options DENY`, `Referrer-Policy no-referrer`, `Permissions-Policy`), `handle_path /api/*` → `reverse_proxy api:8000`, everything else → `reverse_proxy dashboard:8501` (WebSocket works by default), optional `basicauth` block enabled by `BASIC_AUTH_USER` and `BASIC_AUTH_HASH`, request size limit, access log to stdout (JSON), HTTP→HTTPS redirect.

`.env.production.example`: every variable the prod stack needs (`APP_ENV=prod`, `DOMAIN`, `ECR_REPO`, `IMAGE_TAG`, `DATABASE_URL` with `sslmode` note, `POSTGRES_PASSWORD`, `DASHBOARD_API_KEY`, `DASHBOARD_PASSWORD`, `DASHBOARD_READ_ONLY`, `API_ROOT_PATH=/api`, `API_DOCS_ENABLED=false`, `LOG_FORMAT=json`, `PIPELINE_RUN_TIME`, `PIPELINE_TIMEZONE`, `MLFLOW_TRACKING_URI=http://mlflow:5000`, AWS region and log group names...) with placeholders only and comments on how each is generated (for example `openssl rand -base64 32`).

## G.2 Terraform (`deploy/aws/terraform/`)

Files: `versions.tf` (Terraform and AWS provider constraints), `variables.tf`, `main.tf`, `iam.tf`, `network.tf`, `storage.tf`, `monitoring.tf`, `outputs.tf`, `userdata.sh.tftpl`, `terraform.tfvars.example`, `README.md`. Use the **default VPC** via data sources unless `var.vpc_id` is given. Resources:
- **Security group:** inbound 80 and 443 from `0.0.0.0/0`; **no SSH** by default (access through SSM Session Manager); optional `var.admin_cidr` SSH rule disabled by default; all outbound allowed. No database ports open.
- **IAM:** instance role + profile with least privilege: `AmazonSSMManagedInstanceCore`, read on `ssm:GetParameter*` for `/quant/<env>/*` only, read/write on the backup bucket and read on the release bundle prefix, `ecr:GetAuthorizationToken` and pull on the one repository, CloudWatch Logs write to the one log group. No wildcard resources where a specific ARN is possible.
- **ECR repository** (scan on push, immutable tags, lifecycle policy keeping the last 10 images).
- **S3 backup bucket** (versioning, SSE-S3 or KMS, block all public access, lifecycle: transition and expire per retention variables, bucket policy denying insecure transport) and a `releases/` prefix for the deploy bundle.
- **EC2:** Ubuntu 24.04 LTS (latest AMI via data source, pinned by variable), default `t3.large` (variable; set from F.9 measurements), IMDSv2 required, encrypted gp3 root (30 GB) and a separate encrypted gp3 data volume (`var.data_volume_gb`, default 60) mounted at `/opt/quant` with `prevent_destroy`, Elastic IP, detailed monitoring optional, tags (`Project`, `Env`, `ManagedBy=terraform`), `user_data` rendered from the template.
- **SSM Parameters:** `SecureString` placeholders named `/quant/<env>/<NAME>` created with `lifecycle { ignore_changes = [value] }` and a dummy value; the real values are put by the user with the AWS CLI (never in Terraform state or `.tfvars` in git).
- **Monitoring:** CloudWatch log group (retention variable), alarms: EC2 status check failed, CPU > 80% for 15 min, disk used > 80% (CloudWatch agent config in user data), memory > 85%, plus an **SNS topic with email subscription** (`var.alert_email`); an **AWS Budget** with monthly limit variable and 80%/100% email notifications; optional Route 53 record when `var.hosted_zone_id` and `var.domain` are set.
- Outputs: public IP, instance id, ECR URL, bucket names, SSM command to open a shell, the DNS name or sslip.io hostname, the exact next-step commands.
- Run `terraform fmt -check` and `terraform validate` (and `tflint` if present) if Terraform is installed; else NOT EXECUTED and the user runs them (checklist).

## G.3 Server scripts (`deploy/aws/scripts/`, bash, LF, `set -euo pipefail`, shellcheck-clean)

- `bootstrap.sh` (called by user data; idempotent): install Docker Engine + compose plugin, AWS CLI v2, CloudWatch agent; create user/group `quant` (uid 10001), format and mount the data volume (UUID in `/etc/fstab`, only if unformatted), create `/opt/quant/{data,models,pgdata,mlflow-artifacts,logs,caddy,releases}` with correct ownership, set `vm.swappiness`, unattended security upgrades, 2 GB swapfile, time sync, Docker log rotation, install the systemd units.
- `fetch_secrets.sh`: pull `/quant/<env>/*` from SSM into `/opt/quant/.env` (mode 600, owner root), **never echo values**, fail if a required key is missing.
- `deploy.sh <image-tag> [bundle-key]`: download and extract the release bundle from S3, ECR login, `docker compose pull`, run `migrate`, `up -d`, wait for health (timeout), run the on-box smoke test (`scripts/smoke_test.py` inside the app image against `https://${DOMAIN}`), on failure **automatic rollback** to the previously recorded good tag (stored in `/opt/quant/releases/current` and `previous`), keep the last 3 bundles, write a deploy log.
- `rollback.sh [tag]`, `status.sh` (compose ps, disk, memory, last pipeline run, certificate expiry, last backup), `backup.sh` (nightly: `pg_dump -Fc` of `quantdb` and `mlflowdb`, tar of `/opt/quant/data` and `models` with `--exclude` for caches, upload to `s3://<bucket>/backups/<date>/`, verify upload with `aws s3api head-object` and a checksum, write a local success marker used by monitoring; fail loudly otherwise), `restore.sh <backup-date> [--target-db name]` (restore into a scratch DB by default and only replace production with an explicit `--confirm-production` flag), `prune.sh` (old images, logs).
- systemd: `quant.service` (runs compose, `Restart=on-failure`), `quant-backup.service` + `quant-backup.timer` (daily 02:30 IST), `quant-healthcheck.timer` (every 5 min, writes status, optional SNS notification on failure).
- From the user's Windows machine: `deploy/aws/scripts/push_release.ps1` (build the image, tag with the git short SHA or a version, `aws ecr get-login-password | docker login`, push, create the release bundle zip containing `docker-compose.prod.yml`, `Caddyfile`, `deploy.sh`, scripts and the image tag file, upload to S3, print the SSM command that runs `deploy.sh` on the instance) plus a bash equivalent `push_release.sh`.

## G.4 Application changes required for AWS (verify they exist from Part E)

Env-driven config, JSON logging to stdout for the `awslogs` driver, `/health` and `/ready`, proxy headers, `API_ROOT_PATH=/api`, read-only/auth modes, persisted job registry, graceful SIGTERM, `tzdata`, no writes outside the mounted volumes (run the stack with `read_only: true` and fix any module that writes elsewhere, for example `matplotlib`/`joblib` caches or `__pycache__`: set `MPLCONFIGDIR=/tmp`, `PYTHONDONTWRITEBYTECODE=1`, `JOBLIB_TEMP_FOLDER=/tmp`).

## G.5 Data persistence and growth

State lives in `/opt/quant/{data,models,pgdata,mlflow-artifacts}` on the encrypted EBS volume. Provide in `docs/DEPLOYMENT_AWS.md` the procedure to enlarge the volume, the expected growth rate (computed from the F.9 NIFTY-500 synthetic data sizes), and the disk-usage alarm. Do not change the storage backend to S3 in this task (note it as a future enhancement).

## G.6 Initial data and first run on the server

Document and script the first-boot sequence: apply migrations, initial historical ingestion (long-running; run as a one-off container command with progress logging), feature generation, first model training, prediction, signals, first backtest, report generation, so the dashboard is not empty. Provide `deploy/aws/scripts/first_run.sh` that executes these steps in order inside the app image, stops on the first failure and prints what to check. Ingestion volume and timing come from your F.5/F.9 measurements.

## G.7 Backup, restore and disaster recovery

State RPO (24 h) and RTO (≤ 2 h) targets; the nightly backup (G.3); S3 lifecycle (14 daily, 8 weekly); a **restore drill** that you execute locally against scratch databases (F.4 item 8) and that the user repeats on AWS before launch (checklist); a procedure to rebuild the instance from Terraform plus the latest backup.

## G.8 Operations (Day-2) runbook inside `docs/DEPLOYMENT_AWS.md`

Update procedure (build, push, deploy), rollback, rotate secrets (SSM → `fetch_secrets.sh` → restart), renew certificates (Caddy automatic; how to check), scale the instance, view logs (CloudWatch Insights example queries for errors and for pipeline failures), what the alarms mean and what to do, how to stop the stack to save cost, how to destroy everything (`terraform destroy`, with the `prevent_destroy` data volume caveat), and common failures (certificate not issued because DNS is not pointing, port 80 blocked, out-of-memory during training, disk full, database not healthy).

## G.9 Cost control

Document an estimated monthly cost table for the default `t3.large` + 60 GB gp3 + S3 + CloudWatch + data transfer (tell the user to confirm on the AWS pricing page), cheaper options (`t3.medium` for read-only hosting after training elsewhere, Spot not recommended for the database), the Budget alarm, and how to stop the instance without losing data. Do not state prices as facts; label them estimates.

## G.10 Legal and data-terms flags (put in `docs/DEPLOYMENT_AWS.md`, `README.md` and `DISCLAIMER.md`)

State clearly, without legal advice, that (1) publishing predictions or trading signals to the public can fall under Indian securities regulation (SEBI rules for investment advisers and research analysts), so the deployment is framed and labelled as an educational research tool, read-only and with the disclaimer; (2) Yahoo Finance data accessed through yfinance is intended for personal use and its terms may restrict redistribution, so a public deployment should show derived analytics rather than raw redistributed price data, or use a licensed data source; (3) the NIFTY 50/500 index names and data may carry exchange licensing terms. Recommend the user confirm these points with their university or a professional before opening the site to the public, and offer the safer default: authentication in front of the dashboard (basic auth or password gate) during the demonstration period.

---

# PART H. PHASE 6: DOCUMENTATION AND THE USER'S A-TO-Z CHECKLIST

Create or update these files (accurate, tested commands only):

1. `docs/PRE_DEPLOYMENT_CHECKLIST.md`: **the user's own A-to-Z verification**, ordered, each step with: purpose, exact command(s) (PowerShell for the local machine, bash for the server), expected output or pass criterion, and what to do on failure. Sections:
   - **A. Prepare the machine:** tool versions, Git state, clean working tree, backups of `data/`, `models/`, `.env`.
   - **B. Fresh install and one-command verification:** fresh clone, venv, install, `python scripts/verify_all.py --fast`, then `--docker`, then `--e2e`.
   - **C. Database:** start scratch Postgres, migrate up/down/up, run DB tests.
   - **D. Data and ML pipeline by hand:** ingestion, validation, features, training, prediction, signals, contract validation (with expected counts to compare against).
   - **E. Backtest and analytics by hand:** each allocation method, sensitivity run, report.
   - **F. API by hand:** start, open `/docs` (dev), curl every endpoint, auth tests.
   - **G. Dashboard by hand:** open each page, what you should see on each, screenshots reference from `docs/screenshots/`, password gate, read-only mode.
   - **H. Orchestration:** dry run, real run, failure drill, scheduler time check.
   - **I. Container build and local production stack:** build, up, smoke, restart persistence, logs.
   - **J. Security review:** commands for the secret scan, dependency audit, headers check, port exposure check (`nmap` or `Test-NetConnection` against the deployed host).
   - **K. AWS account preparation:** budget alarm first, MFA on the root user, a non-root admin identity, AWS CLI profile, region choice (default `ap-south-1`), service quotas, domain or sslip.io decision.
   - **L. Terraform:** init, fmt, validate, plan (what resources are expected), apply, outputs.
   - **M. Secrets and first deploy:** put SSM parameters with the CLI, push release, deploy, watch logs, first-run script, smoke test from the laptop.
   - **N. Post-deployment verification:** open the HTTPS site, certificate check, auth works, disclaimer visible, data present, API through `/api`, CloudWatch logs flowing, alarms in OK state, force a test alarm, nightly backup runs, **restore drill on AWS**, reboot the instance and confirm everything returns by itself, simulate a failed deploy and confirm automatic rollback.
   - **O. Go/No-Go table** (Part J gates) with a tick box for each.
   - **P. Day-2:** update, rollback, rotate secrets, destroy.
2. `docs/DEPLOYMENT_AWS.md` (architecture diagram in Mermaid, cost estimate, every command, troubleshooting, scaling-up options).
3. `docs/VERIFICATION_REPORT.md` (full matrix with real results, performance numbers, coverage, known gaps) and `docs/CLEANUP_REPORT.md` (Part D.9) and `docs/ISSUES_LOG.md` (every issue found: id, severity Critical/High/Medium/Low, area, symptom, root cause, fix, regression test, status).
4. `README.md`, `docs/runbook.md`, `docs/architecture.md`, `docs/api_reference.md`, `docs/user_guide.md` updated to the final state; `docs/THIRD_PARTY_NOTICES.md`; `DISCLAIMER.md`; `SECURITY.md` (how to report issues, supported versions); `CHANGELOG.md` (version `1.0.0`, summary of changes); bump the version in `pyproject.toml`; tag `v1.0.0-rc1` on the final commit (do not push).
5. Update `.github/workflows/ci.yml` to run the same gates as `verify_all.py --fast` plus a job that builds the image and runs the container smoke test; optionally a manual-dispatch workflow `release.yml` that builds, pushes to ECR with GitHub OIDC (role ARN as a variable) and uploads the release bundle (documented, not required).

---

# PART I. ISSUE HANDLING PROTOCOL

Severity: **Critical** (data loss, secret exposure, security hole, wrong financial result, crash on start), **High** (feature broken, failing test, deployment blocker), **Medium** (degraded behaviour, poor error handling, performance miss), **Low** (cosmetic, docs). Fix all Critical, High and Medium issues in this run; Low issues if cheap. For each: reproduce → root cause → minimal fix → regression test → re-run affected checks → log in `docs/ISSUES_LOG.md`. If an issue cannot be fixed (external dependency, environment limitation), mark **Open** with a workaround and the exact user action needed.

---

# PART J. GO / NO-GO GATES (all must be PASS or explicitly accepted by a documented, justified exception)

1. Fresh install works on Windows and inside a Linux container; `pip check` clean.
2. Full test suite green (non-DB, DB, integration, e2e where runnable), three consecutive runs stable; coverage reported, target met or gaps listed.
3. Linters, formatters clean; bandit/pip-audit/secret scan show no unresolved Critical or High.
4. Migrations reversible; single head; backup and restore drill passed.
5. Every script and every documented command executed successfully (table complete).
6. Real API and real Streamlit verified over HTTP and in a browser; no console errors; disclaimer visible; auth and read-only modes work.
7. Orchestrator failure injection passed; scheduler time verified.
8. Performance targets met or the recommended instance size justified from measurements.
9. Production compose stack runs locally with healthy services, persistence across restarts, no tracebacks in logs, DB/MLflow ports not exposed.
10. Deployment artifacts validated (compose config, shellcheck, terraform validate, artifact tests) or marked NOT EXECUTED with the user's step in the checklist.
11. Documentation complete and commands verified.
12. No secrets in repository or images; `.env*` untracked.

Verdict options: **GO**, **GO WITH CONDITIONS** (list them), **NO-GO** (list blockers).

---

# PART K. FINAL REPORT (print when finished)

1. **Verdict** and the gate table with PASS / FAIL / NOT EXECUTED and evidence paths.
2. **What changed:** counts of files/lines removed, consolidated modules, dependencies removed/added, new files (grouped), commits made, tags.
3. **Issues:** totals by severity, the Critical and High ones in detail, and open items with workarounds.
4. **Test results:** baseline vs final, per folder, coverage, flakiness runs, runtime.
5. **Script and command execution table** (every script, every documented command).
6. **Performance table** and the recommended EC2 size.
7. **Security summary:** findings fixed, accepted risks.
8. **What you could NOT execute** and exactly which checklist steps cover it.
9. **The exact next steps for the user** (a numbered list of at most 15 steps leading from "now" to "live on AWS"), referencing `docs/PRE_DEPLOYMENT_CHECKLIST.md`.
10. **Cross-check list:** any statements in the project report or presentation that are now outdated (test counts, scope, ports, versions) so the user can update them.

Begin with Part A.3 (backups and branch), then Part B, and work through the phases in order. Commit after each phase.
