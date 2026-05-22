---
id: notes/library/90-governance/90-doc-ownership-and-review
title: "90 doc ownership and review"
sidebar_label: "90 doc ownership and review"
sidebar_position: 90
---

# 90 doc ownership and review

Status: draft
Last reviewed: 2026-05-22

## Ownership model

- Operations docs owner: operator lead
- Automation docs owner: automation steward
- Consumer docs owner: reporting/product lead
- Dev/governance docs owner: engineering lead

## Review cadence

- Light review: weekly for active refactors
- Full review: monthly for command/contract drift

## Required reviewer checks

- commands still match `make help`
- paths still exist or are correctly marked legacy
- consumer-safe vs internal-only boundary is respected
