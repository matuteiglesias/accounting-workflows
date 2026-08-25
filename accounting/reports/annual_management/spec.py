from __future__ import annotations

YEARS = ["2022", "2023", "2024", "2025", "2026"]

REPORT_META = {
    "title": "FAMILY BUSINESS / PROPERTY MANAGEMENT",
    "subtitle": "Informe patrimonial y de gestión",
    "scope": "Family Business + Property Management",
    "currency_note": "Moneda nativa; ARS y USD nunca se suman entre sí.",
}

# Exact selectors only. missing_policy=zero is reserved for sparse additive
# flow breakdowns where absence of a dimension row means no activity. Source
# rows explicitly marked unavailable remain unavailable.
SUMMARY_ROWS = [
    {"label": "Renta", "metric_id": "IS.RENT.TOTAL", "currency": "ARS", "role": "major"},
    {"label": "OPEX de propiedad", "metric_id": "IS.OPEX.PROPERTY", "currency": "ARS"},
    {"label": "Resultado operativo", "metric_id": "IS.NET.OPERATING", "currency": "ARS", "role": "major"},
    {"label": "Funding / aportes", "metric_id": "FUND.CONTRIB.TOTAL", "currency": "ARS"},
    {"label": "Retiros personales", "metric_id": "DIST.DRAWS.PERSONAL", "currency": "ARS", "role": "major"},
    {"label": "Dividendos", "metric_id": "DIST.DIVIDENDS", "currency": "ARS"},
    {"label": "Resultado post-retiros", "metric_id": "COV.NET.AFTER_DRAWS", "currency": "ARS", "role": "major"},
    {"label": "Tasa de ahorro / retención", "metric_id": "COV.SAVINGS_RATE", "currency": "ARS", "format": "ratio"},
    {"label": "Caja validada ARS", "metric_id": "BS.CASH.TOTAL", "currency": "ARS"},
    {"label": "Caja validada USD", "metric_id": "BS.CASH.TOTAL", "currency": "USD", "format": "usd"},
    {"label": "Posición neta PM USD", "metric_id": "ID.DEBT.NET_PM_POSITION", "currency": "USD", "format": "usd"},
]

KPI_SPECS = [
    {"label": "RENTA", "metric_id": "IS.RENT.TOTAL", "currency": "ARS"},
    {"label": "RESULTADO OPERATIVO", "metric_id": "IS.NET.OPERATING", "currency": "ARS"},
    {"label": "RETIROS", "metric_id": "DIST.DRAWS.PERSONAL", "currency": "ARS"},
    {"label": "DIVIDENDOS", "metric_id": "DIST.DIVIDENDS", "currency": "ARS"},
    {"label": "POST-RETIROS", "metric_id": "COV.NET.AFTER_DRAWS", "currency": "ARS"},
]

OPERATING_ROWS = [
    {"label": "RENTAS", "metric_id": "IS.RENT.TOTAL", "currency": "ARS", "role": "major"},
    {"label": "CABA", "metric_id": "IS.RENT.BY_PROPERTY", "currency": "ARS", "dimension_name": "Lugar", "dimension_value": "CABA", "indent": 1, "missing_policy": "zero"},
    {"label": "Tigre 01", "metric_id": "IS.RENT.BY_PROPERTY", "currency": "ARS", "dimension_name": "Lugar", "dimension_value": "Tigre 01", "indent": 1, "missing_policy": "zero"},
    {"label": "Tigre 28", "metric_id": "IS.RENT.BY_PROPERTY", "currency": "ARS", "dimension_name": "Lugar", "dimension_value": "Tigre 28", "indent": 1, "missing_policy": "zero"},
    {"label": "Tigre 32", "metric_id": "IS.RENT.BY_PROPERTY", "currency": "ARS", "dimension_name": "Lugar", "dimension_value": "Tigre 32", "indent": 1, "missing_policy": "zero"},
    {"label": "OPEX DE PROPIEDAD", "metric_id": "IS.OPEX.PROPERTY", "currency": "ARS", "role": "major"},
    {"label": "Impuestos", "metric_id": "IS.OPEX.BY_CATEGORY", "currency": "ARS", "dimension_name": "semantic_subbucket", "dimension_value": "taxes", "indent": 1, "missing_policy": "zero"},
    {"label": "Servicios", "metric_id": "IS.OPEX.BY_CATEGORY", "currency": "ARS", "dimension_name": "semantic_subbucket", "dimension_value": "services", "indent": 1, "missing_policy": "zero"},
    {"label": "Mantenimiento", "metric_id": "IS.OPEX.BY_CATEGORY", "currency": "ARS", "dimension_name": "semantic_subbucket", "dimension_value": "maintenance", "indent": 1, "missing_policy": "zero"},
    {"label": "Legal", "metric_id": "IS.OPEX.BY_CATEGORY", "currency": "ARS", "dimension_name": "semantic_subbucket", "dimension_value": "legal", "indent": 1, "missing_policy": "zero"},
    {"label": "RESULTADO OPERATIVO", "metric_id": "IS.NET.OPERATING", "currency": "ARS", "role": "major"},
    {"label": "Margen operativo", "derived": "operating_margin", "format": "ratio", "indent": 1},
    {"label": "OPEX / renta", "derived": "opex_to_rent", "format": "ratio", "indent": 1},
]

OPERATING_USD_ROWS = [
    {"label": "Renta CABA", "metric_id": "IS.RENT.BY_PROPERTY", "currency": "USD", "dimension_name": "Lugar", "dimension_value": "CABA", "format": "usd", "missing_policy": "zero"},
    {"label": "Renta total USD", "metric_id": "IS.RENT.TOTAL", "currency": "USD", "format": "usd", "missing_policy": "zero", "role": "major"},
    {"label": "OPEX propiedad USD", "metric_id": "IS.OPEX.PROPERTY", "currency": "USD", "format": "usd", "missing_policy": "zero"},
    {"label": "Resultado operativo USD", "metric_id": "IS.NET.OPERATING", "currency": "USD", "format": "usd", "missing_policy": "zero", "role": "major"},
]

FUNDING_ACTORS = ["Matías", "Inquilino", "Household", "Alejandro", "Primos", "Héctor"]
DEBT_RELATIONS = ["PM -> MI", "PM -> Primos", "Alejandro -> PM", "Alejandro -> MI", "Hector -> MI"]

QUALITY_METRICS = [
    {"label": "Coverage clasificación", "metric_id": "DQ.CLASSIFICATION.COVERAGE"},
    {"label": "Monto unknown / review", "metric_id": "DQ.UNKNOWN.AMOUNT"},
    {"label": "Cash frontend-safe", "metric_id": "DQ.CASH.FRONTEND_SAFE"},
    {"label": "Reconciliación actividad deuda", "metric_id": "DQ.DEBT.ACTIVITY.RECONCILIATION"},
    {"label": "FX missing-rate", "metric_id": "DQ.FX.MISSING_RATE.AMOUNT"},
    {"label": "FX one-sided", "metric_id": "DQ.FX.ONE_SIDED.AMOUNT"},
    {"label": "FX rows review-required", "metric_id": "DQ.FX.ROWS.REVIEW_REQUIRED"},
]
