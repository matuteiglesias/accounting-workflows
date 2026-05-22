# 91 doc freshness and drift checks

Status: draft
Last reviewed: 2026-05-22

## Drift checks

Run periodically:

```bash
make help
make doctor
rg -n "python -m accounting\.views|run-views|publish-latest|build-all" notes README.md Makefile
```

## Freshness markers

Each page should include:
- status
- last reviewed date
- source anchors (for authority pages)

## Drift policy

- If docs and Makefile disagree, Makefile wins until docs are patched.
- If docs reference removed compatibility paths, mark as historical or update immediately.
