# Household Balbín — discovery y contrato de informes HH1 v2

**Estado:** requisitos recogidos y conciliación de aportes de Claudia actualizada el 19/11/2026. **No** equivale a liquidación aprobada, deuda exigible, cambio del ledger ni determinación legal sobre propiedad. **Scope de datos:** `Box=Household` (etiqueta `HH`), distinto de los `Box=Family Business/Property Management`.

## 1. Unidad administrada y límites

La unidad económica es **la casa de Balbín**, no la fortuna, salario ni cuenta personal completa de Matías. Los recursos los administra Matías, que recibe aportes y ejecuta muchos pagos. La administración usa una **caja virtual**: registra contribuciones y gastos como una cuenta corriente teórica, aunque las facturas puedan pagarse directamente desde una cuenta individual o tarjeta. **No representar `closing_control` como efectivo en banco.**

Separar cuatro ejes que nunca deben colapsarse:
1. **Quién es el responsable del gasto / beneficiario** (HH, propiedad compartida, PM/FB, individuo; todavía discutible en ABL y reparaciones estructurales).
2. **Quién pagó y desde dónde** (cuenta o tarjeta, ejecutor, caja intermediaria).
3. **Qué contribución se reconoció a HH** (aporte físico, aporte contable/virtual, dinero de terceros; una transferencia interna no es ingreso nuevo).
4. **Quién podría tener una acreencia o compensación** (solo si hay una regla/documento aprobado; el déficit HH es un indicador, no un pagaré automático).

**Transferencias:** HH -> PM para la participación HH en agua y eventualmente ABL, sin reintegro automático. Héctor -> FB -> servicios y Chino -> PM, de otros universos, solo como contrapartida de conciliación, jamás duplicar en HH. La participación de cada propietario y la estrategia de ABL se mostrarán como alternativas disputadas, no como resolución jurídica. La contribución de Alejandro a HH viene de distribuciones recibidas por él: registrar **una sola entrada HH** (no computar además la renta/fondo de origen como segunda entrada de HH).

## 2. Actores, ocupación y evidencia

| Actor | Identificador existente/propuesto | Hecho / período | Estado |
|---|---|---|---|
| Matías | `MI` | Administrador; 12 × ARS 150.000 contabilizados en 2025 como aportes **virtuales**; en 2026 paga facturas directamente, sin nuevas contribuciones nominales de MI | Explicación del usuario + 12 registros |
| Candela | `Cande` (no cambiar sin migración) | Aportante irregular; algunos gastos del perro pagados directamente | Registros de aportes + pagos del perro aún por conciliar |
| Alén / Valén | `Alen` en ledger | Último aporte registrado en mayo 2025. Fecha de salida de residencia **aprox. marzo–mayo 2026**, incierta; no inferir ausencia en 2025 por fin de aportes. | Usuario + registros |
| Alejandro | `Alejandro` | No residente; cuatro × ARS 200.000 a HH durante mayo, junio, julio y octubre 2026, desde distribuciones suyas | Usuario + registros |
| Claudia Catalina Acuna Ponce | `CA` (**creado y verificado**) | Residente desde 01/08/2026; dos aportes recibidos y aprobados de ARS 150.000: **29/08/2026** y **04/11/2026**. El primero entra en el corte 31/10; el segundo se registra en origen para un corte posterior. | Usuario y dos asientos fuente verificados, IDs 1668/1669 |
| Chino / Héctor | Externos a HH | Contrapartes en pago de servicios/ABL de propiedad, no miembros HH | No incluir como residentes ni aportantes HH |

**Identificador de actor:** el usuario mencionó inicialmente `SA` y después propuso `CA` para Claudia. El identificador definitivo `CA` ya está disponible en el directorio de actores y en ambos asientos de `C. Long Ledger`. No renombrar ni crear más actores duplicados.

No asignar automáticamente prorrateos mensuales por ocupación mientras las fechas de Alén sean inciertas. Para 2025 el cliente propone considerar tres hermanos como unidad de reparto, aunque el pago de Alén se interrumpe en mayo. Para 2026, madre desde agosto y Alén no contribuye en este año.

## 3. Corte de verdad observado

**Fuente reproducible actualizada:** `out/run/accounting/20261119T222524Z_HH/`, cutoff `2026-10-31`. Sustituye para saldos a la corrida previa `20261119T211126Z_HH`. Los valores siguientes son de **contabilidad virtual** de HH en ARS, no certificados bancarios:

| Indicador | Resultado |
|---|---:|
| Saldo virtual a 2025-12-31 | -439.080,58 |
| Ingresos contables de HH ene–oct 2026 | 1.327.500,00 |
| Egresos contables de HH ene–oct 2026 | 2.897.813,43 |
| Resultado contable ene–oct 2026 | -1.570.313,43 |
| Saldo virtual a 2026-10-31 | -2.009.394,01 |

Aportes `funding_contribution` según el ledger:

| Persona | 2025 contabilizado | 2026 a octubre contabilizado |
|---|---:|---:|
| Matías (MI) | 1.800.000,00 (virtuales según usuario) | 0 |
| Candela (Cande) | 760.000,00 | 377.500,00 |
| Alén | 450.000,00 | 0 |
| Alejandro | 0 | 800.000,00 |
| Claudia (CA) | 0 | 150.000,00 |

**Diferencias remanentes:** el usuario estimó contribuciones de Candela 2026 alrededor de ARS 300.000, mientras el ledger registra ARS 377.500 hasta octubre; requiere revisar si hubo redondeo o pagos con otra finalidad. Los dos aportes de Claudia están conciliados **con fechas 29 de agosto y 4 de noviembre**, no septiembre/noviembre, y se registraron una sola vez en la caja virtual HH. Solo el primero afecta octubre; el segundo entra en el siguiente corte. Matías 2026 sigue con aportes nominales **0**, aunque sus pagos directos pueden financiar parte del desequilibrio, **sin cuantificación individual verificada**. En 2025 la entrada virtual de MI es 12 × 150.000; no insertar segundo ingreso por las facturas que esos aportes representan.

## 4. Catálogo v2: cinco informes, en este orden

| Prioridad | Informe / pestaña | Pregunta de decisión | Origen mínimo | Presentación v2 |
|---|---|---|---|---|
| P0 | **Estado de caja virtual** | ¿Cuál era la posición a diciembre 2025, cómo evolucionó en 2026 y de cuánto es el déficit virtual? | `monthly_cash_accountability`; mostrar egresos, aportes y `closing_control` por separado | Serie mensual, tarjeta de cierre, advertencia **no saldo bancario / no acreencia automática** |
| P0 | **Aportes por persona** | ¿Quién reconocidamente aportó cuánto? ¿Qué falta conciliar? | `classification_audit` `funding_contribution`; no crear filas por testimonio | Barras 2025/2026 nominal ARS; distinguir aportes virtuales y declarados pendientes |
| P0 | **Costos y gastos comunes** | ¿Qué cuesta mantener la casa? | Gasto clasificado HH y reglas de elegibilidad | Composición por servicios, mantenimiento corriente, mascota, plantas; separar reparaciones estructurales y ABL en discusión |
| P1 | **Distribución y escenarios** | ¿Qué significarían reglas alternativas para cada residente? | Costos elegibles, ocupación y aportes validados | Dos o tres escenarios versionados, nunca “deuda ya generada” |
| P1 | **Obligaciones y cruces entre cajas** | ¿Qué vence, qué se pagó por cuenta de HH y qué fue a PM/FB? | Tx/source IDs, espejo de transferencias, obligaciones documentadas | Tabla de trazabilidad y vencimientos; gráfico agregado opcional |

**No crear 26 espejos HH del catálogo FBPM.** Los informes FBPM incompatibles continúan `SKIPPED_INCOMPATIBLE` sin tumbar la corrida. Priorizar que el primero y segundo informes sean correctos antes de crear interacción rica.

**Reglas de costos pendientes de reconciliar:**
- Servicios: electricidad, gas, internet, agua. Internet abonado por tarjeta personal; comprobar si figura en HH ledger antes de añadirlo.
- Agua: HH participa en una de cada dos facturas; HH -> PM -> servicios. Usar **mes facturado**, no necesariamente fecha de pago.
- ABL: participación teórica de la vivienda en un tercio, pero usuario declara controversia/reciprocidad por incumplimiento de Héctor y posible pago por caja común. Presentar escenarios; **no crear deuda HH a partir de esa explicación**.
- Compras mayoristas Macro: insumos compartidos / staples y limpieza, no mezclarlos con todas las compras personales.
- Mascota y plantas: gastos de bienestar/mantenimiento corriente, con etiqueta propia si está sustentada por línea de gasto; no todo es mantenimiento edilicio.
- Pared con humedad y otras reparaciones estructurales: costo potencial de propiedad con presupuesto pendiente, separado de desembolsos actuales y de gastos HH aprobados.
- Limitar la primera versión a los gastos ya presentes en `Box=Household`; no imputar de oficio gastos mencionados sin comprobante o movimiento.

## 5. Escenarios Claudia: especificación, no deuda

El usuario propuso varias bases que son económicamente diferentes. No seleccionar un único criterio de justicia ni confundir una cuota calculada con un ingreso ya recibido. Los aportes de Claudia están conciliados y se agregan **una sola vez** como contribuciones de HH. Definir `C_y` como **costo anual compartible confirmado**, no necesariamente el total del ledger; `C_{ago-dic}` como costo compartible durante la residencia, y `A_CA` como aportes **efectivamente corroborados**.

- **Escenario A — tercio del año completo:** `cuota_CA = C_2026 / 3`. Es una hipótesis distributiva, no prorrateada por ocupación.
- **Escenario B — tercio de los meses que vivió:** `cuota_CA = C_(ago-dic) / 3`. Requiere noviembre y diciembre efectivamente cerrados o pronosticados por separado.
- **Escenario C — cinco doceavos del tercio anual:** `cuota_CA = (5/12) * (C_2026/3)`. Alternativa explícita de prorrateo temporal que **no equivale** a B si los costos varían por mes.
- **Sensibilidad adicional:** `cuota_CA = 1/2 * (C_2026/3)`, equivalente a un peso relativo 0,5 frente a pesos 1,25/1,25 para las otras dos partes si se normaliza el total en 3.

Para cualquiera: `diferencia = cuota_CA - A_CA`. No llamar a `diferencia` «deuda» sin acuerdo y sin reconciliación de costos y aportes. El usuario verificó dos aportes de ARS 150.000 (**29/08 y 04/11**): ARS **150.000 conciliados con CA en el run cerrado a octubre** y ARS 300.000 acumulados en los asientos de origen hasta el 4/11, pendientes de incorporarse a un futuro run de noviembre. No comparar estimaciones de 2026 completo con un cierre efectivo de octubre sin identificar el pronóstico de noviembre–diciembre y la inflación. Los costos excluidos o controversiales (ABL, humedad y otros) deben estar desglosados antes de aplicar terceras partes.

## 6. Corte de implementación para el equipo

**Trabajo inmediato (máximo 2 unidades en curso):**
1. **Clasificar correctamente los gastos compartidos y sus evidencias** (Macro, perro, plantas) y reconciliar cruces con PM/FB. Los aportes de CA ya están asentados; asegurar que `MI` 2025 siga visible como aporte virtual.
2. **Mejorar los dos primeros informes HH1** (saldo virtual y aportes por persona). Reutilizar `monthly_cash_accountability` y `classification_audit` y no diseñar otro motor de deuda.

**No abrir aún:** cobro exigible a familiares, imputación de compras fuera de ledger, pipeline financiero personal, conciliación bancaria universal, reconstrucción de 2021–2022, UI de presupuestos complejos, reparto automático ABL, comparación de ARS nominales con supuestas cifras reales sin deflactor. Otros agentes pueden trabajar la estética después de validar los contratos de datos.

**Criterios de aceptación de HH1 v2:** solo `Household`, monedas separadas, hashes y source run registrados, ningún CSV privado en web, ningún doble conteo de aportes virtuales/pagos directos, aclarar gasto del hogar vs gasto de propiedad, separar hechos/escenarios, reportar filas faltantes como `REVIEW_REQUIRED`, protección de acceso activa.

## 7. Fuente y trazabilidad

Contexto narrativo: conversación de discovery HH del 19/11/2026. Fuente técnica actual: `out/run/accounting/20261119T222524Z_HH`, sin cambio de los punteros `latest_*`. Proof HH1 anterior documentado en `docs/HOUSEHOLD_HH1_POC.md`. Este documento de diseño es **una especificación de trabajo**, no altera registros ni la publicación actual.
