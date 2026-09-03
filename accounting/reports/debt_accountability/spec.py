from __future__ import annotations

REPORT_ID = "debt_accountability"
TITLE = "Posición y movimientos de deuda"
SUBTITLE = "Obligaciones registradas, actividad, repagos y trazabilidad"
TOLERANCE = 0.01

REQUIRED_SOURCES = (
    "monthly_debt_position.csv",
    "monthly_debt_position_qa.csv",
    "monthly_debt_activity.csv",
    "monthly_debt_activity_qa.csv",
    "monthly_debt_repayment_detail.csv",
    "cost_allocation_gaps.csv",
    "cost_allocation_gaps_qa.csv",
)
