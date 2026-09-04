from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpecializedReportSpec:
    report_id: str
    title: str
    description: str
    family: str
    audience: str
    view_key: str
    scope: str
    period_policy: str
    currency_policy: str
    question: str
    establishes: str
    caveat: str
    section_plan: tuple[str, ...]


REPORT_SPECS = (
    SpecializedReportSpec(
        "pm_tax_accountability",
        "Impuestos de Property Management",
        "Impuestos PM por actor y período.",
        "costos",
        "stakeholder",
        "pm_tax_by_actor",
        "FBPM",
        "latest_year",
        "separate_native",
        "¿Qué impuestos PM fueron pagados o aplicados por cada actor?",
        "Muestra importes reconocidos dentro de la población gobernada de impuestos y su fuente de cobertura.",
        "Los importes reconocidos no implican que hayan ingresado a caja PM ni determinan responsabilidad jurídica.",
        ("summary", "pie", "table", "method"),
    ),
    SpecializedReportSpec(
        "pm_services_accountability",
        "Servicios de Property Management",
        "Servicios PM por actor y período.",
        "costos",
        "stakeholder",
        "pm_services_by_actor",
        "FBPM",
        "latest_year",
        "separate_native",
        "¿Qué servicios PM fueron pagados o aplicados por cada actor?",
        "Muestra importes reconocidos dentro de la población gobernada de servicios y su fuente de cobertura.",
        "Los importes reconocidos no implican que hayan ingresado a caja PM ni determinan responsabilidad jurídica.",
        ("summary", "pie", "table", "method"),
    ),
    SpecializedReportSpec(
        "stakeholder_support",
        "Aportes aplicados a Property Management",
        "Aportes reconocidos por actor y período.",
        "stakeholders",
        "stakeholder",
        "pm_support_by_actor",
        "FBPM",
        "latest_year",
        "separate_native",
        "¿Quién aportó o aplicó recursos a PM?",
        "Muestra apoyo reconocido por actor y Box objetivo a partir del mart gobernado de stakeholder support.",
        "El apoyo no constituye por sí solo caja física, deuda jurídica ni una liquidación o neteo entre actores.",
        ("summary", "pie", "table", "method"),
    ),
    SpecializedReportSpec(
        "distributions_by_recipient",
        "Distribuciones registradas por receptor",
        "Distribuciones gobernadas por receptor y período.",
        "distribucion",
        "stakeholder",
        "distributions_by_recipient",
        "FBPM",
        "latest_year",
        "separate_native",
        "¿Quién recibió distribuciones registradas?",
        "Muestra la distribución registrada por receptor dentro de la membresía gobernada de distribuciones.",
        "La pertenencia a este universo no prueba custodia final, derecho jurídico ni saldo neto definitivo entre actores.",
        ("summary", "pie", "table", "method"),
    ),
    SpecializedReportSpec(
        "rent_by_property",
        "Renta por inmueble / fuente",
        "Renta reconocida por inmueble o ubicación de origen.",
        "produccion",
        "stakeholder",
        "rent_by_property",
        "FBPM",
        "latest_year",
        "separate_native",
        "¿Qué inmuebles o fuentes explican la renta reconocida?",
        "Muestra renta operativa gobernada agrupada por ubicación para el período y moneda seleccionados.",
        "No determina titularidad, derecho a frutos ni distribución jurídica de la renta.",
        ("summary", "pie", "table", "method"),
    ),
    SpecializedReportSpec(
        "rent_monthly_evolution",
        "Evolución mensual de renta",
        "Serie mensual de renta reconocida durante el año de corte.",
        "produccion",
        "stakeholder",
        "rent_monthly_evolution",
        "FBPM",
        "latest_year_months",
        "separate_native",
        "¿Cómo evolucionó mes a mes la renta reconocida?",
        "Muestra la serie mensual gobernada de renta operativa en moneda nativa.",
        "La serie no convierte moneda ni interpreta por sí sola vacancia, cobrabilidad o derechos sobre los frutos.",
        ("summary", "bars", "table", "method"),
    ),
    SpecializedReportSpec(
        "opex_by_category",
        "Costos operativos por categoría",
        "OPEX de propiedad agrupado por categoría gobernada.",
        "costos",
        "stakeholder",
        "opex_by_category",
        "FBPM",
        "latest_year",
        "separate_native",
        "¿En qué categorías se concentraron los costos operativos de propiedad?",
        "Muestra OPEX reconocido por categoría semántica gobernada y moneda.",
        "No asigna responsabilidad jurídica por el costo ni transforma funding, deuda o distribuciones en OPEX.",
        ("summary", "pie", "table", "method"),
    ),
    SpecializedReportSpec(
        "taxes_by_property",
        "Impuestos por inmueble",
        "Impuestos reconocidos agrupados por inmueble o ubicación.",
        "costos",
        "stakeholder",
        "taxes_by_property",
        "FBPM",
        "latest_year",
        "separate_native",
        "¿Qué inmuebles explican los impuestos reconocidos?",
        "Muestra impuestos de propiedad reconocidos y agrupados por ubicación.",
        "No determina quién debía soportarlos jurídicamente ni implica que cada pago haya pasado por caja del Box.",
        ("summary", "pie", "table", "method"),
    ),
    SpecializedReportSpec(
        "services_by_property",
        "Servicios por inmueble",
        "Servicios reconocidos agrupados por inmueble o ubicación.",
        "costos",
        "stakeholder",
        "services_by_property",
        "FBPM",
        "latest_year",
        "separate_native",
        "¿Qué inmuebles explican los servicios reconocidos?",
        "Muestra servicios de propiedad reconocidos y agrupados por ubicación.",
        "No determina quién debía soportarlos jurídicamente ni implica que cada pago haya pasado por caja del Box.",
        ("summary", "pie", "table", "method"),
    ),
    SpecializedReportSpec(
        "distributions_vs_rent",
        "Distribuciones y renta reconocida",
        "Comparación anual entre renta reconocida y distribuciones registradas.",
        "distribucion",
        "stakeholder",
        "distributions_vs_rent",
        "FBPM",
        "latest_year",
        "separate_native",
        "¿Cómo se comparan la renta reconocida y las distribuciones registradas del período?",
        "Presenta ambas magnitudes gobernadas lado a lado para facilitar la lectura administrativa.",
        "No supone que toda distribución provenga de la renta del mismo período ni constituye una liquidación jurídica o saldo final por actor.",
        ("comparison", "table", "method"),
    ),
)
