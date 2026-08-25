from __future__ import annotations

START_PERIOD = "2022-01"

LABELS = {
    "opening_control": "Apertura control",
    "rent_in": "Rentas",
    "funding_cash_in": "Aportes",
    "debt_principal_in": "Principal deuda",
    "debt_repayment_in": "Repagos recib.",
    "debt_interest_in": "Interés deuda",
    "internal_transfer_in": "Transf. internas",
    "fx_in": "FX entrada",
    "other_cash_in": "Otras entradas",
    "unknown_cash_in": "Sin clasificar",
    "taxes_out": "Impuestos",
    "services_out": "Servicios",
    "maintenance_out": "Mantenimiento",
    "legal_out": "Legal",
    "personal_draws_out": "Retiros personales",
    "dividends_out": "Dividendos",
    "debt_principal_out": "Principal deuda",
    "debt_repayments_out": "Repagos",
    "debt_interest_out": "Interés deuda",
    "internal_transfer_out": "Transf. internas",
    "fx_out": "FX salida",
    "fx_cost_out": "Costo FX",
    "other_cash_out": "Otras salidas",
    "unknown_cash_out": "Sin clasificar",
    "total_cash_in": "Total entradas",
    "total_cash_out": "Total salidas",
    "net_cash_flow": "Flujo neto",
    "closing_control": "Cierre control",
    "direct_tax_support_non_cash": "Impuestos",
    "direct_service_support_non_cash": "Servicios",
    "other_non_cash_support": "Otro apoyo",
}

GROUPS = [
    ("Posición", ["opening_control"]),
    ("Entradas", [
        "rent_in", "funding_cash_in", "debt_principal_in", "debt_repayment_in",
        "debt_interest_in", "internal_transfer_in", "fx_in", "other_cash_in",
        "unknown_cash_in",
    ]),
    ("Salidas", [
        "taxes_out", "services_out", "maintenance_out", "legal_out",
        "personal_draws_out", "dividends_out", "debt_principal_out",
        "debt_repayments_out", "debt_interest_out", "internal_transfer_out",
        "fx_out", "fx_cost_out", "other_cash_out", "unknown_cash_out",
    ]),
    ("Resultado", ["total_cash_in", "total_cash_out", "net_cash_flow", "closing_control"]),
    ("Apoyo directo sin movimiento de caja", [
        "direct_tax_support_non_cash", "direct_service_support_non_cash",
        "other_non_cash_support",
    ]),
]

ALWAYS_SHOW = {
    "opening_control", "rent_in", "total_cash_in", "total_cash_out",
    "net_cash_flow", "closing_control",
}

BOX_ORDER = ["Family Business", "Property Management"]
CURRENCY_ORDER = ["ARS", "USD"]
