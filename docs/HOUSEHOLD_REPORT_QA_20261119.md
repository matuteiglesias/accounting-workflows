# Report Director — Household: ciclo de informes y revisión del viewer

**Fecha:** 19/11/2026 · **Ámbito:** HH1, casa de Balbín · **Estado:** una iteración funcional, semántica y visual completada; elegibilidad de costos pendiente.
**Universo:** `Box=Household` solamente · **run validado:** `20261119T222524Z_HH` · **corte:** `2026-10-31`.
**Producción FB/PM:** deliberadamente fuera de alcance. **Saldo de una cuenta personal y deuda exigible:** deliberadamente fuera de alcance.

## Propósito comunicacional

El público de HH1 es la familia que participa o debe comprender la administración de Balbín. Los informes deben contestar, en este orden:

1. **Cuál es la posición contable de la casa** y por qué no representa automáticamente una deuda de Matías.
2. **Quién contribuyó** y cuál es el carácter real/virtual del aporte.
3. **A qué se destinaron los recursos** según el ledger, sin atribuir sin prueba la obligación final.
4. **Cuándo ocurrieron los gastos**, distinguiendo fecha de pago y período facturado.
5. **Cómo fluyeron los fondos** en cada mes y cómo se conecta esto con el saldo.

Este es el contrato de narrativa. Los reportes que no puedan responderlo con datos trazables quedan `SKIPPED_INCOMPATIBLE`, no se reemplazan con datos inventados.

## QA factual del backend (datos cerrados y reproducibles)

| Aserción | Resultado | Control |
|---|---:|---|
| Movimientos canónicos HH | 265 | Exclusivamente `Household` |
| Ingresos virtuales HH enero–octubre 2026 | ARS 1.327.500,00 | Cierre y `monthly_cash_accountability` |
| Egresos enero–octubre 2026 | ARS 2.897.813,43 | Igual a gasto clasificado en este run |
| Resultado enero–octubre | ARS −1.570.313,43 | Entradas menos salidas |
| Saldo al 31/12/2025 | ARS −439.080,58 | Apertura 2026 |
| Saldo al 31/10/2026 | ARS −2.009.394,01 | Apertura + resultado |
| Aportes virtuales Matías en 2025 | ARS 1.800.000,00 | Ledger + regla narrada |
| Aporte Claudia de agosto, dentro de corte | ARS 150.000,00 | CA→HH |
| Aporte Claudia de noviembre | Fuera de corte | No sumar a octubre |
| Moneda representada por los cinco informes | ARS | USD excluido sin conversión |
| Elementos incompatibles del registry patrimonial | 32 definiciones omitidas | No corresponden a 32 informes desplegados |

**No se adjudica saldo negativo como acreencia individual.** La fuente debe seguir siendo el único ámbito donde se imputan pagos; la visualización consume exclusivamente agregados aprobados y reportes preconstruidos.

## Auditoría de producto: hallazgos → correcciones

| Severidad | Hallazgo visto en navegador | Acción implementada | Evidencia y resultado |
|---|---|---|---|
| P0 | Cinco gráficos sin reportes que los stakeholders pudieran abrir | El backend ahora produce cinco **HTML** independientes con KPI, tabla de valores, lectura, cautela y fuente | Se abrieron los cinco enlaces en Chrome |
| P0 | Empezaba por «entradas y salidas»; escondía el déficit en última pestaña | Recorrido reorganizado: posición virtual → aportes → destino → gastos → flujo | Orden y visibilidad verificados |
| P0 | El protagonista de la portada era «POC» en vez de Balbín | Portada «La casa de Balbín», breve pregunta de gestión y resumen factual de cuatro KPIs | Capturas desktop y mobile |
| P0 | Posible confusión de saldo negativo con deuda personal | Advertencia explícita en portada e informe de caja; datos de fuentes contables | Revisión textual + prueba HTML |
| P1 | Gráficos sin tabla ni importes exactos | Cada reporte incluye tabla mensual/por actor y texto de lectura | HTML reportes, 5 tablas |
| P1 | Reportes incompatibles con Household podrían crear datos ficticios o provocar falla global | No se ejecutan: registry de reportes FBPM se registra como `SKIPPED_INCOMPATIBLE` | Catalog HH scope y pruebas |
| P1 | Primeras imágenes demoraban al cambiar de tab por carga `lazy` | Cinco SVG pequeños se precargan (`eager`) | Browser audit 0 errores de imágenes tras 16 vistas |
| P1 | Eje de ARS en inglés y déficit graficado en azul neutro | Unidades en formato local y déficit en naranja | Captura real del SVG final |
| P2 | Móvil mostraba solo 2–3 tabs sin instrucción | Texto «Deslizá las pestañas…» en móvil | Capturas a 390×844 y navegación de cinco tabs |
| P2 | Repetir la construcción exigía comandos manuales sin gate | Targets `make build-hh1` y `make accept-hh1`; bundle explícito | Prueba de aceptación sobre raíz aislada |

No se alteraron las reglas patrimoniales, ni el Sheet de origen, ni los 26 informes FBPM, ni la publicación de `accounting-viewer.vercel.app`.

## Contrato de artefactos

**Backend** `accounting.reports.household_poc` produce catálogo `household_poc_catalog.v2`:
- Cinco SVG de gráficos.
- Cinco reportes HTML (cada uno con fuente, tabla, KPIs y caveat).
- Un `household_catalog.json` con `scope=HH`, cutoff, checks, rutas, hashes SHA-256 para SVG y HTML, lista de skips.
- Ningún ledger CSV dentro del bundle publicado.

**Frontend** `scripts/build_hh1.py` solo acepta el catálogo v2, valida todos los hashes y rutas de las 10 piezas, rechaza SVG activo y HTML con scripts/iframes/external refs; crea `/HH1/`, subruta `/HH1/reports/`, y deja `/reports/` tal cual. El paquete completo aceptado tiene **66 archivos**: 55 de FBPM aceptados más 11 de HH1 (index + cinco SVG + cinco HTML). Nada de hojas o archivos secretos de usuario.

**Pruebas:** 8 pruebas focalizadas de backend/scope, 29 del viewer, controles de catalog y hashes, 16 verificaciones efectivas con Puppeteer sobre cinco tabs + cinco reportes + móvil. Auditoría del navegador final: **0 errores**, sin overflow horizontal en móvil ni reportes sin gráfico.

**Evidencia visual privada:** `ops/report-director/private/20261119_HH1_visual_QA/` en el repo local (no subir capturas a Vercel). Tests y capturas se reproducen con el script del Report Director, no reemplazan lectura humana.

## Loop de aprobación (repetible)

1. **Problema observable:** identificar tab/informe y registrar `DATA`, `REPORT`, `NARRATIVE`, `VISUAL` o `SECURITY`; definir el criterio de aceptación.
2. **Revisión de origen:** localizar el movimiento/mes en `20261119T222524Z_HH` y confirmar el scope `Household`; no inferir de otros Box.
3. **Corregir en la capa más temprana válida:** clasificación fuente si afecta la cifra, productor de reportes si afecta sentido, frontend solo si es disposición o legibilidad.
4. **Regenerar run-owned bundle nuevo** con un destino vacío; ejecutar pruebas del productor; no mover `latest_FBPM` por accidente.
5. **Build del viewer aislado** con `make accept-hh1 HH1_BUNDLE=<ruta> DIST=<dir-limpio>`; validar allowlist de archivos, hashes y ausencia de material privado.
6. **Recorrer navegando realmente** todas las pestañas y abriendo todos los informes, desktop y móvil, registrar screenshots y fallas por severidad.
7. **Publicar solo preview protegido** si `SSO=all` y no hay P0/P1; no promover a dominio principal sin revisión explícita del contenido.
8. **Registrar conclusión, defectos remanentes y la próxima prioridad**, evitando trabajar simultáneamente en cambios cosméticos que dependan de datos inestables.

## Dos frentes de trabajo siguientes (no abrir más por ahora)

**A. Elegibilidad de costos comunes (DATA):** revisar movimientos `Servicios` ARS 2.562.982,86, `Legales` ARS 200.000, `Mantenimiento` ARS 134.830,57; separar consumos comunes, perro, Macro, plantas, obligaciones PM/ABL y reparaciones estructurales. Confirmar que algunos asientos de agua fueron imputados por fecha de pago aunque correspondan al mes siguiente. **No** calcular tercios antes de depurar el denominador.

**B. Estado de relaciones y aportes (REPORT):** conciliación entre aportes, pagos personales directos y compensaciones; incorporar los montos del perro pagados por Candela si están sustentados, evitando doble registro. La fecha efectiva de salida de Alén sigue incierta; cuotas anuales y prorrateos de Claudia siguen siendo escenarios explícitos, no deudas.

**Pruning:** no forzar 26 informes, no instrumentar login adicional (ya existe Vercel Authentication), no crear hojas paralelas, no convertir la caja virtual a conciliación bancaria ni inventar una cuenta a cobrar a Matías. El éxito de esta iteración es que una persona puede leer cinco reportes correctos y entender exactamente qué está confirmado y qué requiere una decisión.

## Release candidate probado (preview, sin promoción)

- **URL de revisión:** https://accounting-viewer-407ef7c8r-matias-projects-5c20d82c.vercel.app/HH1/
- **Deployment:** `dpl_GKLnMY5fKaxFYiH7nvCZn1N3cioC` (`READY`, target preview, sin alias de producción).
- **Integridad de archivos desplegados:** Vercel devolvió exactamente 66 paths, 11 de ellos debajo de `/HH1/`. Ningún path ajeno a los dos catálogos allowlisted.
- **Acceso:** protección `Vercel Authentication` activa `deploymentType=all`. Un visitante sin credenciales redirige a `vercel.com` y no recibe el contenido Household.
- **Prueba real:** peticiones autenticadas a `/HH1/`, `/HH1/reports/virtual_balance.html`, `/HH1/reports/contributions.html` devolvieron contenido correcto con Claudia ARS 150.000 al corte y aportes virtuales MI 2025 ARS 1.800.000.
- **Estado:** sin cambios en el dominio principal de producción ni en los `latest` de backend.

El próximo agente no necesita volver a implementar la navegación ni el productor: revisar primero los dos pendientes de datos enumerados arriba, generar nuevo run HH si se corrige el ledger, reproducir `make accept-hh1` y pasar el recorrido de navegador antes de sugerir promoción.
