---
id: notes/journal
title: "Journal"
sidebar_label: "Journal"
---

Bien. Hagamos el lock-in como “Decision Record” corto y operativo, y dejemos claro qué se de-prioriza.

## Lock-in: decisión de arquitectura para reporting

### Objetivo

Generar reportes distribuibles (HTML y luego PDF) de forma reproducible y automatizable, sin duplicar lógica contable ni reintroducir hardening en notebooks.

### Decisión principal

**Opción A**: el reporte NO ejecuta notebooks.
El reporte consume artefactos ya canónicos de D.materialize y F.views, genera figs, y ensambla HTML con template.

### Qué queda de-priorizado (y por qué)

1. **Storypack v0 como pipeline paralelo**
   Se considera deprecated. Se rescatan solo patrones de ensamblado (manifest, paths relativos, galería de figs/tablas).
   Motivo: mezcla hardening, paths y lógica nueva, y compite con el contrato de stages.

2. **Notebook como producto final (nbconvert a HTML)**
   Queda como herramienta de dev/EDA, no como salida distribuible.
   Motivo: reproducibilidad y contrato débil.

---

## Contratos e invariantes

### Invariante 1: fuente de verdad

El reporte se alimenta de un `run_id` específico, leyendo desde:

* `out/run/accounting/<run_id>/views/...`
* y si hace falta: `out/run/accounting/<run_id>/...` (marts canónicos)

### Invariante 2: separación dev vs producto

Tu idea de `out/reports` sirve, pero con una regla estricta para evitar pisar cosas:

* `out/reports/dev/`
  para notebooks y exploración manual

* `out/reports/run/<run_id>/`
  para outputs publicables de ese run:

  * `figs/`
  * `tables/`
  * `html/`
  * `story_manifest.json`
  * `storypack_governance.html`
  * `storypack_operational.html`

* `out/reports/latest/`
  puntero al último `out/reports/run/<run_id>/`

Con esto mantenés tu preferencia de “carpeta out/reports” sin caer en un output global que se pisa solo.

### Invariante 3: inputs mínimos del governance pack

El governance pack v1 consume solo:

* `v_cashflow_monthly`
* `v_contributions_monthly`
* `v_opex_category_monthly`
* warnings/sanity (si existe)

Nada más. Cualquier cosa fuera de eso va al operativo o queda fuera.

---

## Estructura del nuevo componente (sin bloat)

### Nuevo stage: Report builder

Un módulo/script en `src/accounting/` (por ejemplo `accounting/report_builder.py`) que hace:

1. Resolver `run_id` y paths
2. Cargar views
3. Generar figs (PNG o SVG)
4. Exportar tablas a HTML (pandas Styler simple)
5. Ensamblar `storypack_governance.html` y `storypack_operational.html` usando templates y CSS fijo
6. Escribir `story_manifest.json` con:

   * `run_id`
   * inputs usados (paths)
   * warnings
   * lista de outputs generados

### CSS y templates versionados

* `templates/storypack_governance.html.j2`
* `templates/storypack_operational.html.j2`
* `templates/storypack.css`

Nada de CSS generado por notebook.

---

## Próximas sesiones de dev (claras y acotadas)

### Sesión 1: esqueleto reproducible

* Crear `report_builder.py` que genere:

  * `story_manifest.json`
  * HTML final con secciones vacías (placeholders)
  * copia de CSS al output
* Crear `Makefile` target:

  * `make report RUN_ID=...`

Resultado: pipeline existe, aunque aún con contenido mínimo.

### Sesión 2: llenar governance pack v1

* Conectar inputs reales: `v_cashflow_monthly`, `v_contributions_monthly`, `v_opex_category_monthly`
* Generar 3 a 5 figs mínimas y 3 tablas cortas
* Portada con 3 bullets auto calculados (sin texto largo)

Resultado: ya podés mandar un HTML prolijo a familia.

### Sesión 3: operativo y packaging PDF

* Agregar drilldown operativo (tablas largas, anomalías)
* Agregar conversión HTML a PDF como paso separado (sin mezclar con generación)

Resultado: distribución por PDF, sin romper reproducibilidad.

---

## Nota de guardrail (para no reintroducir caos)

Si algo requiere “hardening” para que el reporte funcione, eso es señal de que:

* o falta un output en views/materialize
* o hay drift en el contrato de esos stages

En ese caso se corrige en D/F, no en el report builder.

Si querés, el próximo paso práctico es que peguemos el contenido actual de tu `scripts/run_storypack.py` (o el entrypoint que pensabas usar) y lo rediseñamos como `make report` con esa estructura `out/reports/run/<run_id>` y `latest`, sin tocar tu out/run existente.

