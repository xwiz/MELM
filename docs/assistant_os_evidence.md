# Local Assistant OS — Evidence Ledger

## Status: auto-generated

This file is **never hand-edited**. A CI job renders it from the JSON
artifacts of `pi-smoke`, `eval`, `run-transcript-replay`, etc., with
timestamps and git SHA. Stale-by-construction claims are impossible.

## Generation

```bash
# Generate evidence report from the latest gate artifacts:
python scripts/local_assistant_os_cli.py pi-smoke --reset --json > pi-smoke.json
python scripts/local_assistant_os_cli.py shortcut-audit --json > shortcut-audit.json
python scripts/local_assistant_os_cli.py eval --json > eval.json
# Then render this file from the artifact directory
```

## Current evidence (will be auto-populated)

| Gate | Passed | Timestamp | Git SHA |
|------|--------|-----------|---------|
| pi-smoke | — | — | — |
| shortcut-audit | — | — | — |
| eval | — | — | — |
| pytest | — | — | — |
