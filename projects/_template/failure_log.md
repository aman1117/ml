# Failure log

Every failure found in an eval run, a load test or production logs goes here, and
**every entry becomes at least one regression eval case**. Use the taxonomy from
`../../serving/monitor.py` and `../../agents/agent_eval.py` where it fits.

| # | First seen | Source (eval / load test / prod log) | Category | Example (request id or input) | Frequency | Root cause | Fix | Eval case added | Status |
|---|------------|--------------------------------------|----------|-------------------------------|-----------|------------|-----|-----------------|--------|
| 1 | YYYY-MM-DD | | | | | | | | open |

**Categories to start from:** retrieval miss · wrong chunk · unfaithful answer ·
should have abstained · wrong tool · bad tool args · agent loop · prompt injection
followed · format/template error · truncation · degenerate repetition · timeout ·
overload (429) · input drift · regression after change.
