---
id: notes/repository_tree_contract
title: "Repository Tree Contract"
sidebar_label: "Repository Tree Contract"
---

# Repository Tree Contract

Status: current authority
Last reviewed: 2026-08-25

## Purpose

Keep source ownership legible from the repository root and prevent migration-era material from leaking back into the runtime package.

The current import root is deliberately the top-level `accounting/` package. There is no parallel `src/accounting/` tree. Moving to a `src/` layout would be a separate packaging migration that must update imports, tests, scripts, `PYTHONPATH`, and deployment wiring atomically; it is not a cleanup synonym.

## Governed top-level roots

| Root | Ownership |
|---|---|
| `.github/` | Repository CI/workflow metadata. |
| `accounting/` | Importable production Python only. No notebooks, generated reports, copied documentation, or alternate app/runtime stacks. |
| `diagnostics/` | Frozen dated audit/census evidence. Not runtime input and not current command authority. |
| `docs/` | Contract/reference and historical source documents. Mixed historical content may be consolidated later; runtime code must not depend on it. |
| `fixtures/` | Deterministic offline test/smoke inputs. |
| `notes/` | Current operational/governance documentation plus dated migration evidence. Status/date determine authority. |
| `reference/` | Versioned runtime reference policies such as governed FX policy files. |
| `scripts/` | Repository checks and bounded operator/developer utilities. No alternate pipeline orchestration. |
| `tests/` | Regression, contract, and architecture tests. Tests must protect supported behavior, not preserve unused compatibility surfaces. |

Generated roots such as `out/`, `public/`, local `private/`, caches, and virtual environments are not source-tree authorities and must remain untracked as governed by `.gitignore` and publication rules.

## `accounting/` path rules

1. A path under `accounting/` must be importable runtime code or package metadata used by the supported pipeline.
2. Presentation notebooks and exploratory reports do not belong under the runtime namespace. The retired `accounting/notebooks/` tree must stay absent.
3. Removed alternate stacks such as `accounting.human` and `accounting.viz` must stay absent unless a new independently justified product boundary is approved; they are not compatibility destinations.
4. A `*_legacy.py` module is temporary migration debt, not a naming convention. It must have a current caller census and a concrete removal frontier.
5. Public compatibility exports are permitted only for concrete callers that cannot yet migrate. Test-only consumers do not justify a public alias.
6. New Python code should import supported modules directly rather than routing through facade aliases for convenience.

## Current compatibility remainder

The remaining physical legacy delegates are intentionally narrow:

- `accounting/metrics/annual_legacy.py`: still supplies baseline annual builder/helpers and schema constants reached through the current annual facade/contract validation;
- `accounting/professional/annual_dashboard_tables_legacy.py`: still supplies historical-shape fallback helpers for current cash/funding companion builders, but no longer has a public compatibility-export surface in the facade;
- `accounting/professional/drilldown_legacy.py`: still supplies the bounded orchestration/rendering remainder and a small caller-backed export seam.

Their presence is not precedent for new legacy modules. The next deletion wave should migrate callers/helpers out of these delegates before deleting each file; it should not add more forwarding layers.

## Root cleanup policy

Dated audit evidence may remain in `diagnostics/`, `docs/`, or `notes/` when it explains historical decisions. It must be clearly non-authoritative and must not be used as justification to preserve dead executable paths.

A root/path cleanup is safe when all of the following hold:

- no production or automation caller consumes the path;
- current docs do not advertise it as supported;
- tests cover the underlying supported invariant elsewhere;
- removal does not erase the only evidence for an unresolved accounting decision.

## Enforcement

Architecture tests assert the high-value negative boundaries: no `src/accounting`, no `accounting/notebooks`, no retired alternate runtime packages, and no notebook files inside `accounting/`. The Make command-surface test separately asserts the exact supported public target set so compatibility names cannot silently grow back.
