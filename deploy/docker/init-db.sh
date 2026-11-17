#!/usr/bin/env bash
set -euo pipefail

# Initialize auxiliary databases required for production stack
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    SELECT 'CREATE DATABASE mlflowdb'
    WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'mlflowdb')\gexec
    GRANT ALL PRIVILEGES ON DATABASE mlflowdb TO $POSTGRES_USER;
EOSQL
