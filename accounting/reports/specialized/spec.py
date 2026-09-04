from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpecializedReportSpec:
    report_id: str
    title: str
    description: str
    family: str
    question: str
    caveat: str


REPORT_SPECS = (
    SpecializedReportSpec("pm_tax_accountability", "Impuestos de Property Management", "Impuestos PM por actor y período.", "costs", "¿Qué impuestos PM fueron pagados o aplicados por cada actor?", "Los importes reconocidos no implican que hayan ingresado a caja PM ni determinan responsabilidad jurídica."),
    SpecializedReportSpec("pm_services_accountability", "Servicios de Property Management", "Servicios PM por actor y período.", "costs", "¿Qué servicios PM fueron pagados o aplicados por cada actor?", "Los importes reconocidos no implican que hayan ingresado a caja PM ni determinan responsabilidad jurídica."),
    SpecializedReportSpec("stakeholder_support", "Aportes aplicados a Property Management", "Aportes reconocidos por actor y período.", "support", "¿Quién aportó o aplicó recursos a PM?", "El apoyo se informa por Box objetivo y no constituye una liquidación o neteo jurídico."),
    SpecializedReportSpec("distributions_by_recipient", "Distribuciones registradas por receptor", "Distribuciones gobernadas por receptor y período.", "distributions", "¿Quién recibió distribuciones registradas?", "La pertenencia a este universo proviene de la autoridad de distribuciones; no prueba custodia final."),
)
