# HH Balbín — Rendición documental y reglas de convivencia (discovery)

**Corte de discovery:** 19 de noviembre de 2026 (hora local Buenos Aires).
**Origen:** relato del administrador Matías, fuentes HH de contabilidad y archivos operativos.
**Estado:** diseño y revisión de cobertura documental; **no es acuerdo familiar, deuda emitida, auditoría bancaria ni afirmación de consentimiento de Claudia**.

## 1. Qué requiere la contraparte que pide rendición

Según lo que Matías relata de sus conversaciones recientes, Claudia quiere **abrir las boletas** y ver el respaldo de las cifras. No se localizó una minuta original de esas conversaciones con una posición textual firmada por Claudia. No atribuirle declaraciones más allá de lo relatado.

**Requerimiento de interfaz (HH):** desde un gasto visible en un informe, un integrante autorizado debe poder ver:
- inmueble/universo, concepto/proveedor, mes de consumo o cuota, vencimiento y monto de la boleta;
- asiento de pago, fecha, caja, ejecutor/custodio, quién financió, estado (pagado, planeado, refinanciado);
- **boleta** vinculada (documento que origina obligación, no prueba pago);
- **comprobante** vinculado y estado de revisión (pagado no implica comprobante aprobado);
- clasificación del gasto como HH, propiedad, mantenimiento, perro, etc.;
- criterio de compartibilidad y su **estado de aceptación o controversia**;
- código del ledger y trazabilidad para que una corrección no borre el original.

El visual debería mostrar por separado **importe gastado** y **cuota imputable a cada persona**. Hacer accesibles las facturas no obliga automáticamente a alguien a cubrir todas las partidas.

## 2. Cobertura documental REAL del HH actual

En `out/run/accounting/20261119T222524Z_HH`, corte 31/10/2026:
- 265 transacciones HH canónicas, de las cuales **40 salidas ARS** son gastos clasificados `property_opex` en 2026 (ARS 2.897.813,43).
- Entre esas 40, **cero** tienen `evidence_ref` o `evidence_status` informados. No significa que las boletas no existan en Drive/correo/carpeta; **significa que no están vinculadas, aprobadas y verificadas por la corrida HH**.
- Varias filas HH identifican el destino simplemente como `Servicios`; **el proveedor y el período de factura todavía no están bien expuestos en la tabla**. Deben resolverse por `Detalle`, `tag`, datos de cuenta, período de obligación y documentación, sin deducirlos de la caja o del pagador.\n- Las 40 tienen identidad de asiento original verificable. Ojo: **`source_row` es el índice del origen, no necesariamente el ID de la columna A de la hoja**. Validamos cada `source_row+1` contra el `transaction_id` leído de la hoja real: 40/40 coinciden. No generar enlaces a `A{transaction_id}` porque puede enlazar a la operación equivocada.
- La prueba interna `ops/report-director/private/20261119_HH_transparency_review/HH_documentation_internal_review.html` muestra las 40, el faltante documental y la URL al **asiento** solo para el administrador. **No confundir el enlace al asiento con un enlace a la boleta**, ni exponer la hoja completa a familiares: contiene FB/PM y otros datos fuera de HH.
- El backend ya tiene un contrato reaprovechable `accounting/reports/payment_evidence.py`, usado para PM, que distingue `liability_source`/boleta, `payment_proof` y `transfer_proof`, `statement_context`, e integridad criptográfica de anexos. Su título, `report_id` y `scope_tag` actuales son específicos de PM, así que HH requiere **un adaptador de alcance y parámetros de presentación**, no copiar datos PM ni duplicar el motor de evidencia.
- **NO publicar** los documentos ni la vista de fuente completa hasta comprobar control de acceso y autorización por documento. La protección Vercel del viewer no implica que Claudia tenga ya permisos de lectura a Drive.

### MVP de links a boletas — interfaz solicitada

Prototipo con una vista *Documentos y comprobantes*, accesible desde informes HH de salidas:
1. Elegir período y servicio (Edenor, Metrogas, AySA, Internet, ABL u otros respaldados).
2. Tabla: cuota/mes, importe, vencimiento, fecha de pago, responsable/ejecutor, estado documental.
3. Dos enlaces distintos con estados independientes: **Ver boleta**; **Ver comprobante**.
4. Si falta cualquiera, mostrar **Sin documento vinculado** y opción de solicitarlo/añadir evidencia para el administrador; nunca generar URL supuesta ni marcar pagado como documento aprobado.
5. Un contador de cobertura (`N con boleta / N elegibles`, `N con comprobante aprobado / N pagos`), con denominadores explícitos. En el corte actual 0/40 vinculados, antes de hacer la búsqueda de documentos.
6. Matizar pagos HH→PM: el comprobante de transferencia entre cajas no demuestra por sí mismo pago del servicio al emisor; se necesitan ambos tramos si se quiere probar aplicación final.

**Decisión técnica:** anexar una vista de evidencia al recorrido HH1 sin convertir las cinco pestañas en 26 informes patrimoniales. Iniciar con un piloto de 10 facturas de electricidad/gas/agua; completar, verificar que las URLs se puedan abrir con el rol familiar y después escalar. No convertir este piloto en obligación de digitalizar *todo* desde 2024 antes de que la primera boleta sea consultable.

## 3. Candela y gastos del perro — no duplicar los ARS 550.000 nuevos

El propietario de la administración reporta que **Candela pagó ARS 550.000 el 19/11/2026** por el perro y que asumió otros pagos de días anteriores. Esta declaración **no es todavía una fila adicional validada**. El ledger tiene un paquete para el perro en noviembre:

| Concepto | ARS | Registro |
|---|---:|---|
| Operación veterinaria | 650.000 | pagado, imputado a Candela; fecha real de operación/documental a confirmar |
| Estudios cardiológicos | 135.000 | pagado, fecha 04/11 provisional |
| Consulta/medicación | 71.000 | pagado, fecha 04/11 provisional |
| Medicación | 50.000 | pagado; pagador específico a confirmar |
| **Gasto pagado total** | **906.000** | cuatro conceptos, sin incluir futuro |
| Adicional operativo | 80.000 | **planeado**, Matías pagador previsto, NO pagado |
| Contribución económica Candela atribuida | 706.000 | neto de 856.000 que habría pagado y 150.000 reintegrados por Alén |
| Contribución de Alén vía Candela | 150.000 | no depósito separado a HH; su posible compensación de deuda histórica sigue pendiente |

**Ambigüedad obligatoria de conciliar:** ¿los ARS 550.000 del 19/11 forman parte de los ARS 650.000 de operación o de los ARS 856.000 ya atribuidos a Candela, o son un nuevo pago? Sin un identificador de transferencia/boleta no se puede elegir. También queda identificar quién pagó exactamente los ARS 50.000 de medicación. Un costo del perro y su aporte de financiación son dos patas de una operación, **no dos gastos**.

El costo se reconoce en HH como gasto **mantenimiento/bienestar compartido** según política del administrador; **la obligación de Claudia de soportar su fracción está controvertida**, no aceptada/emitida. Cuota fiscal/distributiva pendiente. Registrar escenario *incluye perro como gasto común* versus *excluye perro del reparto a Claudia*; mostrar sensibilidad, sin alterar lo efectivamente pagado.

## 4. Alén: aportó hasta mayo de 2025, pero son dos hechos distintos

El último aporte nominal de `Alen` en 2025 fue de **ARS 150.000 el 12/05/2025**; también realizó pagos directos de Metrogas y mantenimiento en febrero/marzo. No se registran nuevos aportes HH de Alén durante el resto de 2025. Esto respalda la interrupción **de aportes**, no prueba automáticamente la fecha de desocupación ni libera de una cuota histórica.

El usuario propone tercios 2024 y 2025 entre Matías, Candela, Alén. El clearing 2025 permanece **escenario de reparto** hasta aprobar costos elegibles y conciliación de los repagos históricos. En noviembre de 2026 Alén transfirió ARS 150.000 a Candela por el perro; el usuario evalúa aplicar esa suma a deudas viejas. **La misma transferencia no puede ser aporte corriente HH y cancelación de deuda antigua a la vez**.

## 5. Diferencia de criterios con Claudia: reportar sin confundir con acuerdo

El administrador sostiene la regla normativa doméstica: quien aprovecha los beneficios de compartir hogar participa proporcionalmente en sus gastos comunes, incluidos ciertos gastos de bienestar (mascota, plantas, insumos comunes). Claudia, según relato de Matías, **no desea participar del gasto del perro**. Por tanto:
- marcar **`shareability_status=DISPUTED`** para el perro respecto de Claudia;
- conservar **`household_expense_status=RECORDED`** para pagos efectivamente realizados;
- preservar **`proposed_policy=40/40/20 (2026)`**, **`approved_policy=NOT_DOCUMENTED`**;
- representar escenarios de reparto completos e incremental *con/sin perro* para evitar que la diferencia normativa o de convivencia contamine el costo observado;
- no etiquetar “morosa”, “deudora” o “obligada jurídicamente” por un escenario no aprobado;
- ofrecer información neutral, legible y comprobable que permita discutir la aceptación sin convertir el viewer en una herramienta de presión.

No confundir “desentendimiento”, “dejó de aportar” y “no residía”; son hipótesis con cronologías distintas. Si surgen notas originales de la conversación, adjuntarlas como evidencia de **postura**, no como boletas ni como aprobación de cuotas.

## 6. Orden de trabajo / puertas de control

**P0 — Evidencia que abre:** vincular diez facturas HH (boleta real y comprobante real) por `tx_id` a un inventario aprobado, comprobar acceso externo autorizado y que el PDF/imagen corresponde al mismo período e importe. Usar la infraestructura de evidencia PM sin abrir su corpus al viewer HH. Corregir aquí antes de agregar más UI.

**P1 — Paquete mascota:** conciliar los ARS 550k reportados el 19/11, costos pagados ARS 906k, aportes atribuidos Candela ARS 706k + Alén ARS 150k, y ARS 80k planeados. En particular, auditar evidencia para cirugía y medicación de ARS 50k.

**P1 — Convivencia y reparto:** publicar *hechos* + *posturas registradas* + *simulaciones* como secciones separadas. Para 2026 los porcentajes 40/40/20 son un escenario definido por el administrador, no una regla aceptada por las partes.

**No hacer ahora:** emitir pagarés USD, modificar aportes por el reporte de ARS 550k sin identidad de pago, reconstruir cuentas bancarias personales, dar acceso a la hoja contable integral, copiar todos los PDFs de PM a HH o desplegar una lista sin boletas con el rótulo “comprobantes disponibles”.
