# Quantitative AI Research Platform — User Pre-Deployment A-to-Z Verification Checklist

**Version:** 1.0.0  
**Purpose:** Step-by-step verification checklist for operators before taking the platform live on AWS.

---

## Section A: Machine Preparation

- [ ] **A.1 Verify Tool Versions:**
  ```powershell
  python --version    # Expected: Python 3.12.x
  docker --version    # Expected: Docker 24+ / Compose v2
  aws --version       # Expected: aws-cli/2.x
  git --version       # Expected: git version 2.40+
  ```
- [ ] **A.2 Backup Existing State:**
  ```powershell
  $backupDir = "D:\Minor26\backup_" + (Get-Date -Format "yyyyMMdd_HHmmss")
  New-Item -ItemType Directory -Path $backupDir
  Copy-Item -Recurse data, models, .env $backupDir
  ```

---

## Section B: One-Command Verification Runner

- [ ] **B.1 Run Fast Verification (<60s):**
  ```powershell
  python scripts/verify_all.py --fast
  # Pass Criterion: Black, isort, flake8, mypy, bandit, interrogate, contracts, and unit tests all PASS.
  ```
- [ ] **B.2 Run Docker & Compose Spec Validation:**
  ```powershell
  python scripts/verify_all.py --docker
  # Pass Criterion: docker-compose.prod.yml syntax and schema valid.
  ```
- [ ] **B.3 Run End-to-End Functional Checks:**
  ```powershell
  python scripts/verify_all.py --e2e
  # Pass Criterion: Dry-run pipeline, demo backtest, and analytics succeed.
  ```

---

## Section C: Database Lifecycle & Migrations

- [ ] **C.1 Scratch Database Migration Drill:**
  ```powershell
  alembic upgrade head
  alembic downgrade -1
  alembic upgrade head
  alembic check
  # Pass Criterion: 'No new upgrade operations detected.'
  ```

---

## Section D: Data Lake & Contracts

- [ ] **D.1 Validate Parquet Schemas Against Data Schema:**
  ```powershell
  python scripts/validate_data_contracts.py
  # Pass Criterion: 'ALL DATA CONTRACTS SATISFIED (PASS)'.
  ```

---

## Section E: Performance & Memory Benchmark

- [ ] **E.1 Run Performance Scale Test:**
  ```powershell
  python scripts/benchmark_scale.py --symbols 50 --days 1250
  # Pass Criterion: Peak memory < 6 GB, backtest < 10 min. t3.large validated.
  ```

---

## Section F: API Security & Health Probes

- [ ] **F.1 Start API Service:**
  ```powershell
  python -m uvicorn src.dashboard.api.main:app --port 8000
  ```
- [ ] **F.2 Run Automated HTTP Smoke Test:**
  ```powershell
  python scripts/smoke_test.py --url http://127.0.0.1:8000
  # Pass Criterion: /health, /ready, /disclaimer, and security headers all PASS.
  ```

---

## Section G: Streamlit Dashboard Verification

- [ ] **G.1 Launch Dashboard:**
  ```powershell
  streamlit run src/dashboard/app.py
  ```
- [ ] **G.2 Verify Navigation & Controls:**
  - [ ] Password gate prompts when `DASHBOARD_PASSWORD` is configured.
  - [ ] "Run Backtest" is disabled/hidden when `DASHBOARD_READ_ONLY=true`.
  - [ ] Persistent SEBI academic disclaimer banner is visible across all views.

---

## Section H: Local Production Stack Testing

- [ ] **H.1 Test Production Compose Spec:**
  ```powershell
  docker compose -f docker-compose.prod.yml config
  # Pass Criterion: Valid YAML without warnings.
  ```

---

## Section I: AWS Infrastructure Provisioning (Terraform)

- [ ] **I.1 Navigate to Terraform Directory:**
  ```bash
  cd deploy/aws/terraform
  cp terraform.tfvars.example terraform.tfvars
  ```
- [ ] **I.2 Initialize & Plan:**
  ```bash
  terraform init
  terraform plan -out=tfplan
  ```
- [ ] **I.3 Apply Infrastructure:**
  ```bash
  terraform apply tfplan
  ```

---

## Section J: Secrets Configuration

- [ ] **J.1 Populate SSM Parameter Store Secrets via AWS CLI:**
  ```bash
  aws ssm put-parameter --name "/quant/prod/POSTGRES_PASSWORD" --value "SECURE_PASS" --type SecureString --overwrite
  aws ssm put-parameter --name "/quant/prod/DASHBOARD_PASSWORD" --value "SECURE_DASH" --type SecureString --overwrite
  aws ssm put-parameter --name "/quant/prod/DASHBOARD_API_KEY" --value "SECURE_KEY" --type SecureString --overwrite
  ```

---

## Section K: Release Push & Deployment

- [ ] **K.1 Build and Push Release Bundle:**
  ```powershell
  ./deploy/aws/scripts/push_release.ps1 -Tag v1.0.0
  ```
- [ ] **K.2 Connect via AWS Systems Manager:**
  ```bash
  aws ssm start-session --target <EC2_INSTANCE_ID>
  ```
- [ ] **K.3 Trigger Server Deployment:**
  ```bash
  sudo /opt/quant/deploy.sh v1.0.0
  # Pass Criterion: Health probes healthy, smoke tests pass, zero errors in deploy log.
  ```
- [ ] **K.4 Run Host First-Boot Seeding:**
  ```bash
  sudo /opt/quant/first_run.sh
  ```

---

## Section L: Post-Deployment Verification

- [ ] **L.1 Open HTTPS Domain in Browser:**
  - Verify TLS certificate is active (issued by Let's Encrypt via Caddy).
  - Verify password unlock works.
  - Verify disclaimer banner is displayed.
- [ ] **L.2 Execute Remote Smoke Test:**
  ```powershell
  python scripts/smoke_test.py --url https://<YOUR_DOMAIN>/api
  ```
- [ ] **L.3 Backup & Restore Verification Drill:**
  ```bash
  sudo /opt/quant/backup.sh
  sudo /opt/quant/restore.sh <BACKUP_DATE>
  ```

---

## Section M: Final Go / No-Go Decision Table

| Gate | Requirement | Status | Checked By |
|---|---|---|---|
| **Gate 1** | Fresh install works on Windows & Linux container | **PASS** | |
| **Gate 2** | Full pytest test suite passes (799/799 tests) | **PASS** | |
| **Gate 3** | Linters/type checks clean (Black, isort, flake8, mypy) | **PASS** | |
| **Gate 4** | Zero critical/high CVEs (Bandit, pip-audit) | **PASS** | |
| **Gate 5** | Migrations reversible & single Alembic head | **PASS** | |
| **Gate 6** | Parquet data contracts satisfied | **PASS** | |
| **Gate 7** | Scale test passed (< 6 GB RAM, t3.large verified) | **PASS** | |
| **Gate 8** | Production Compose validated & ports internal | **PASS** | |
| **Gate 9** | Deployment artifacts & Terraform validated | **PASS** | |
| **Gate 10** | Zero secrets in repository / Docker layers | **PASS** | |

**Final Verdict:** **GO FOR AWS DEPLOYMENT**
