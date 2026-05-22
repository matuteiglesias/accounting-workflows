---
id: notes/narrative
title: "Narrative"
sidebar_label: "Narrative"
---

Perfecto: tu “negocio” no es solo contabilidad de gastos, es un sistema de renta inmobiliaria familiar con gobernanza frágil, préstamos internos, y riesgo de desvío de caja. Eso cambia por completo qué historias valen la pena y qué figuras son “estándar”.

Abajo te propongo un set de historias que yo intentaría contar, con vistas bastante estándar, pero adaptadas a tus reglas (alquileres como ingreso base, egresos operativos, deuda interna Matías↔familia, contribuciones extraordinarias, y necesidad de transparencia).  

## 1) Historia: “El motor de caja” funciona o no funciona

Pregunta: con alquileres alcanza para operar sin pedir plata afuera

Vistas estándar:

* Estado de resultados cash basis mensual: Ingresos (alquileres + contribuciones) vs egresos (impuestos, servicios, mantenimiento, legales)  
* Serie mensual de flujo de caja (neto) con acumulado (cumsum)
* Waterfall anual: Ingresos -> Operativos -> Legales -> Mantenimiento -> Resultado neto

Por qué importa en tu caso: ya está escrito que el flujo de caja negativo fuerza aportes personales y vuelve el sistema “insostenible”. 

## 2) Historia: “Caja crónica” y quién la financia

Pregunta: si no alcanza, quién cubre el agujero y con qué costo institucional

Vistas estándar:

* “Fuentes de financiamiento del déficit”: alquileres vs contribuciones extraordinarias
* Timeline de contribuciones con conteo de eventos y monto total
* Tabla: contribuciones por actor (Matías, Alen, Candela, otros) y saldo neto

Esto está en el corazón de tus reglas: se describe explícitamente dependencia de contribuciones y el costo en equidad y conflicto.  

## 3) Historia: “Disciplina operativa”: impuestos, servicios, mantenimiento, legales

Pregunta: cuánto cuesta mantener el patrimonio vivo y qué pasa si se posterga

Vistas estándar:

* Top categorías de egresos por año y por mes
* “Aging” de pasivos operativos: cuánto de impuestos y servicios se acumula y desde cuándo
* Semáforo de prioridad de egresos (urgente, necesario, opcional) si lo tenés como tag

Esto calza con tu esquema de egresos y el énfasis en diferenciar operación vs personal, y en visibilizar deuda acumulada. 

## 4) Historia: “Desvíos y mezcla de fondos”: cuánto distorsiona todo

Pregunta: qué porcentaje de la caja se fue a gastos personales o no operativos, y qué impacto tuvo en deuda y tensión

Vistas estándar:

* Serie mensual de “gastos personales” atribuibles al administrador vs egresos operativos 
* “Unmapped / sospechoso”: gasto sin categoría clara, o categoría “personal”, o sin propiedad asignada
* Ratio: gastos personales / ingresos del mes

Esta historia es clave porque tu guía lo marca como error grave que genera desbalance y conflicto.  

## 5) Historia: “Balance real”: liquidez, deudas, y patrimonio neto proxy

Pregunta: cuál es la posición económica consolidada, no solo el flujo

Vistas estándar:

* Balance simplificado: Activos líquidos + cuentas por cobrar internas vs pasivos operativos + deuda interna
* Evolución mensual de Activos líquidos y Pasivos operativos
* Indicadores de alerta:

  * Factor de ahorro (si lo calculás)
  * Equity proxy y su evolución

Tu propio material enfatiza que factor de ahorro cercano a cero y equity negativo son señales de alerta y deben monitorearse. 

## 6) Historia: “Quién le debe a quién” (deuda interna como objeto de gobernanza)

Pregunta: cuánto se le debe a Matías, a primos, etc, y cuál es el plan de repago

Vistas estándar:

* Subledger de préstamos internos: saldo inicial, nuevos adelantos, repagos, saldo final
* “Aging” de deuda interna: cuánto tiempo lleva pendiente
* Escenarios de repago: si destinás X por mes, cuándo se pone en cero

Tu guía lo define como reporte estándar y además lo usa como palanca estratégica: priorizar desendeudamiento interno antes que “invertir”.  

## 7) Historia: “Rentas por propiedad”: performance y riesgo operativo

Pregunta: qué propiedades sostienen el sistema y cuáles son problema

Vistas estándar:

* P&L por propiedad (ingresos atribuibles, egresos atribuibles, neto)
* Vacancia y morosidad: meses sin renta o con retraso
* “Cobranza”: calendario de renta esperada vs cobrada

Esto engancha con lo que marcás: vacancias e incumplimientos afectan liquidez, y hay que descontarlas en proyección.  

## 8) Historia: “Transparencia y control”: el sistema como antídoto de conflicto

Pregunta: el sistema está haciendo su trabajo político: alinear percepciones y evitar discusiones estériles

Vistas estándar:

* “Pack trimestral”: 5 láminas fijas (cashflow, balance proxy, deuda interna, pasivos operativos, anomalías) 
* Métrica de calidad de datos: porcentaje categorizado, número de transacciones sin propiedad, reconciliación lograda
* “Anomalías”: top 10 transacciones más grandes, pagos duplicados, meses con gaps

Tu guía lo plantea como estructura de reportes periódicos y acceso compartido como mecanismo de gobernanza.  

---

## Dos decisiones que te van a destrabar “del reporte a la historia”

1. Definir 2 audiencias y 2 paquetes

* Paquete Operativo (para vos): detalle, anomalías, reconciliación, tags, debugging.
* Paquete Gobernanza (para familia): 5 vistas fijas, cero ruido, foco en obligaciones, saldo, y reglas.

2. Congelar un “Story Pack” con vistas inmutables
   Si cada mes inventás vistas nuevas, nunca se vuelve un instrumento. Elegí 8 historias máximo (las de arriba) y dejalas quietas. Lo que cambia mes a mes es el dato, no el formato.

---

## Qué te pediría de tus reglas para aterrizarlo del todo

Sin que me pases datos todavía, solo estructura:

* Lista de propiedades y cómo las identificás en el ledger
* Lista de “cajas” o cuentas bancarias (FB vs PM si aplica) 
* Cómo marcás: gasto operativo vs personal, préstamo interno, transferencia, y moneda
* Qué querés que sea “verdad oficial” para la familia: cash basis puro o algo más

Con eso, te devuelvo un mapa muy concreto: “de tus tablas agregadas actuales, cuáles alimentan cada historia y qué gráficos exactos van en cada notebook”.



Buenísimo, esto ya tiene forma de sistema “tipo ERP” chico: dimensiones (Sources, Parties, Rent Contracts), un fact table (Ledger), staging de excepciones (Bills), y una capa explícita para reconciliación (Balance Overrides + Dashboard). Con eso se puede armar un set de historias estándar, pero bien adaptadas a renta inmobiliaria familiar y a tus dos cajas (Household vs Property Management).

Voy a proponerte:

1. las historias que yo contaría
2. qué vistas estándar van con cada historia
3. qué datasets mínimos tiene que producir tu capa Report para que esas historias salgan baratas
4. dos problemas de diseño que conviene corregir ya, porque si no te rompen reconciliación y narrativas

## 1) Historias estándar para tu caso (cash basis como verdad oficial)

### Historia A: Salud de caja por caja y total

Pregunta: el sistema está líquido o vive al borde

Vistas:

* Serie mensual de saldo final por `Box` y consolidado
* Serie diaria o semanal si tenés suficiente granularidad
* Mínimo del año, drawdown y tiempo de recuperación

Dataset report mínimo:

* `balances_daily` o `balances_monthly` por `box_id` y por `bank_account_id`

### Historia B: Reconciliación bancaria, la verdad dura

Pregunta: opening + net flows = closing, y dónde falla

Vistas:

* Tabla por mes y cuenta: opening, net, closing esperado, closing statement, diferencia
* Drilldown de diferencias: top 20 líneas candidatas (missing, duplicadas, fecha corrida)

Dataset:

* `recon_monthly` por bank_account_id
* `recon_unmatched_statement_lines`
* `recon_unmatched_ledger_lines` (si importás statement)

Tu tab E. Balance Overrides encaja perfecto acá: sirve para cargar opening y closing “de estado de cuenta” cuando no importás statements completos.

### Historia C: Motor de ingresos de rentas (rent roll real)

Pregunta: quién paga, cuánto, cuándo, y cuánto falta

Vistas:

* Rent roll mensual: esperado vs cobrado por inquilino y por propiedad
* Morosidad: meses con cobro tardío o faltante
* Ocupación: % unidades activas, unidades con ocupante no pagador (como Barrios)

Dataset:

* `rent_expected_monthly` desde D. Rent Contracts
* `rent_collected_monthly` desde Ledger filtrando rent receipts
* `rent_arrears_snapshot` (saldo vencido estimado)

Esto es la historia más “estándar” en property management y la que más rápido te da claridad operacional.

### Historia D: Costos operativos por propiedad y por rubro

Pregunta: cuánto cuesta sostener el patrimonio y dónde se va la plata

Vistas:

* OPEX mensual por propiedad (impuestos, servicios, mantenimiento, legales)
* Top rubros anuales y su volatilidad
* Heatmap rubro x mes

Dataset:

* `opex_by_property_month`
* `opex_by_category_month`
* `vendor_spend_month` (source name o account_id)

Tu tabla de Sources con `Category`, `Source Name`, `Cuenta`, `Partida` es exactamente el backbone para esto. La clave es que quede 100% determinística la asignación ledger line -> category -> property.

### Historia E: Separación negocio vs hogar (mezcla y desvíos)

Pregunta: cuánta caja del negocio termina en hogar o viceversa, y por qué

Vistas:

* Flujo neto entre boxes (Household <-> Property Management) por mes
* Pie o barras: “Contribuciones” vs “Transferencias” vs “Préstamos” vs “Repagos”
* Ratio: aportes extraordinarios / ingresos de rentas

Dataset:

* `interbox_flows_month`
* `capital_contributions_month`
* `owner_draws_month` (si aplica)

Esto te sirve para gobernanza familiar, porque convierte discusiones difusas en objetos contables.

### Historia F: Subledger de préstamos internos (quién le debe a quién)

Pregunta: saldo de préstamos entre partes y su evolución

Vistas:

* Saldos por parte: MI, Alejandro, Candela, etc
* Nuevos préstamos y repagos por mes
* Aging simple: cuánto tiempo promedio queda saldo abierto

Dataset:

* `loans_ledger` o `internal_loans_monthly` con payer, receiver, flujo y saldo acumulado

Acá tu tabla Parties & Units es clave para normalizar HH, PM, MI, CANDE, etc.

### Historia G: Performance por propiedad

Pregunta: qué propiedades sostienen el sistema y cuáles drenan caja

Vistas:

* NOI cash basis por propiedad (rentas cobradas menos opex atribuible)
* Ranking anual de propiedades por neto
* Timeline por propiedad con eventos grandes (mantenimiento, legales)

Dataset:

* `property_pnl_month` con ingresos, opex, neto

Con cash basis no es NOI “contable” perfecto, pero como instrumento de decisión funciona.

### Historia H: FX y exposición (solo si hay multi moneda de verdad)

Pregunta: cuánto ARS vs USD, y qué parte del resultado es FX

Vistas:

* Flujos por moneda
* Balance por moneda
* Si convertís: resultado en moneda base con tipo de cambio de D. FX Rates

Dataset:

* `cashflow_month_currency`
* `balances_month_currency`
* `fx_valuation_notes` (metodología simple)

## 2) El cuello de botella que sentías: mapping “datos -> historias -> figuras”

Tu sensación es correcta. La solución no es “aprender contabilidad general”, es definir un “story pack” fijo y exigir a la capa Report que produzca los datasets para ese pack.

En tu caso, yo congelaría 8 notebooks, uno por historia A a H, y cada notebook solo consume 1 a 3 tablas report estables.

Ejemplo de contrato para cada notebook:

* inputs: `report/*.parquet` o `report/*.csv`
* outputs: 3 a 8 figuras + 1 tabla resumen + un bloque de “hallazgos del mes”

## 3) Qué debería producir tu capa Report para que todo sea fácil

Si hoy tu Report layer es “varias tablas agregadas”, perfecto. Yo las haría explícitas, estilo star schema liviano:

Dimensiones:

* `dim_date`
* `dim_party`
* `dim_property` (Lugar: Tigre 27, Tigre 28, CABA, etc)
* `dim_source` (tus Sources)
* `dim_category`
* `dim_box`
* `dim_currency`

Hechos:

* `fact_ledger_lines` canonizado (cada línea con line_id, txn_id, date, amount_signed, currency, payer_party_id, receiver_party_id, property_id, source_id, category_id, box_id, status)
* `fact_rent_expected` (desde contratos)
* `fact_balance_statement` (desde overrides)

Agregados report listos para plotear:

* `cashflow_monthly`
* `balances_monthly`
* `recon_monthly`
* `property_pnl_month`
* `rent_roll_month`

Con eso, las historias salen sin lógica ad hoc.

## 4) Decisiones de diseño que te van a ahorrar dolor

1. Convención de signo única
   Elegí una sola regla y aplicala siempre en `amount_signed`:

* positivo = entra a la caja
* negativo = sale de la caja

Luego, “payer/receiver” queda para análisis de contrapartes, no para signo.

2. Clasificación inequívoca de movimientos entre cajas
   Cualquier cosa HH <-> PM debe caer en una de estas 4 clases:

* transferencia interna
* contribución de capital
* préstamo
* repago de préstamo

Si esto queda ambiguo, la mitad de las historias se mezclan.

## 5) Próximo paso concreto, sin tocar tu pipeline

Como ya tenés ingestion/materialize/report funcionando, lo más eficiente ahora es:

1. Definir tu “Story Pack 2025” como lista fija A a H
2. Listar cuáles de estos datasets report ya existen y cuáles faltan
3. Ajustar el contrato del report layer para emitir los que faltan, sin reescribir lógica en notebooks

Si me pegás:

* 10 a 20 nombres de outputs actuales de tu Report layer (archivos o tablas)
* y cómo definís hoy `Box`, `payer`, `receiver`, `Flujo`, `Tipo`, `Lugar`

te devuelvo un mapa exacto: por cada historia, qué tabla usás, qué filtros, qué métricas, y qué gráficos específicos (tipo de gráfico, ejes, agrupación, top-N, etc).

