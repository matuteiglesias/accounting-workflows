# Report Director — misión de mejora continua de artefactos

**Estado:** misión persistida, **no activada** como servicio o tarea programada. `accounting-workflows` es **público**: no incluir nombres, cifras sensibles, correspondencia, documentos privados ni datos de ledger en archivos versionados.

## Inicio de un nuevo hilo

1. Abrir [START_HERE.md](START_HERE.md) y pegar su prompt completo en un nuevo hilo de ChatGPT.
2. El agente debe leer [MISSION.md](MISSION.md), [ARTIFACT_MAP.md](ARTIFACT_MAP.md), [TASKS.yaml](TASKS.yaml) y [CODEX_TASK.md](CODEX_TASK.md).
3. Si está disponible un entorno autorizado, consultar los repositorios y los artefactos **existentes**, no reconstruirlos desde cero.
4. Mantener el trabajo real y los resultados de sesiones en `private/` (excluido de Git) o en un almacenamiento privado autorizado. No inventar sesiones Codex ni despliegues.

La misión se centra en **la calidad comunicacional y epistemológica de los informes y gráficos**: qué pretende decir cada artefacto, qué evidencia lo sustenta, qué interpreta realmente un lector, qué falta para una decisión y cuál es el cambio mínimo que mejora esa relación.

**Bucle:** elegir una decisión/audiencia → inspeccionar el artefacto renderizado y su fuente → formular una crítica concreta → corregir explicación, visualización o código según corresponda → regenerar/validar → comparar antes/después → aceptar, revertir o detenerse.

No es un proyecto de dashboard nuevo, no necesita duplicar un motor contable ni introducir otro orquestador.

## Responsabilidad por repositorio

| Autoridad | Dónde | Qué puede hacer el Director |
| --- | --- | --- |
| Cálculos contables y reportes HTML/PDF | Este repo, `accounting/reports/`, `accounting/reports/time_series.py`, `accounting/reports/build.py` | Revisar; pedir mejoras **de presentación** y tests. Semántica contable sólo con decisión explícita del dueño. |
| Manifest, provenance y publicación | `notes/report_bundle_contract.md`, report manifests y catálogo | Validar cortes, hashes, enlaces y contenido; **no** publicar o mover `latest` sin autorización. |
| Navegación de informes | `accounting-viewer` (repositorio separado) | Revisar la secuencia editorial y solicitar cambios específicos en la base de código del Viewer; no implementar un segundo motor. |
| Comprobantes | `pm_payment_evidence` y snapshots privados aprobados | Evaluar explicación, descubrimiento y enlaces; las relaciones documentales sólo pueden ser aprobadas en su proceso propio. |
| Gobernanza / stakeholders | Material A0–A17 y Strategy Lab **privados** | Seleccionar preguntas, probar interpretaciones y preparar materiales; no convertir hipótesis en decisiones familiares ni difundir dossiers legales. |

Para contexto reservado, el operador puede mantener `private/family_context.md` dentro de esta carpeta. **Está ignorado por Git mediante `.gitignore`.** Los archivos de trabajo nunca deben pasar accidentalmente a PRs públicos.

## Condición de progreso

Una iteración sólo se cuenta como mejora cuando identifica **una falla de lectura o decisión** y entrega un cambio que la corrige **sin degradar la verdad, el alcance, la seguridad o la trazabilidad**. Los cambios de código son medios, no objetivos.

WIP: 1 revisión de artefacto + 1 intervención técnica pequeña como máximo. Si no hay evidencia nueva, devolver `NO_OP`.

**Entradas canónicas:** antes de actuar, leer `AGENTS.md` y `SYSTEM.yaml` de este repo; no relajar sus restricciones. Consultar la rama de trabajo/PRs y preservar cualquier edición previa del operador. Este director no debe sobreescribir los cambios simultáneos del Viewer.
