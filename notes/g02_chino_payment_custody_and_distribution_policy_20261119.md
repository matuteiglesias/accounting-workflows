# G02 — Chino: pago personal en ARS, imputación a PM y USD bajo custodia

**Corrección del 19 de noviembre de 2026. Fuente de custodia/pagador físico: declaración del usuario, NO extractos conciliados.**
**Estado:** experimento aislado, no modificación del long ledger, de contratos, del motor de deuda ni de informes publicados.

## Hallazgo y corrección crítica

Los históricos **2026-05, 2026-06, 2026-07, 2026-10** muestran transferencias efectivamente pagadas a primos por **ARS 482.384 / 485.000 / 514.000 / 506.900**, que incluyen ARS 200.000 base **más** ARS 282.384 / 285.000 / 314.000 / 306.900 equivalentes a USD 190 por mes.

- Los **ARS 1.188.284** de equivalencias **salieron del Mercado Pago personal de Matías** y deben **imputarse a PM**. Los asientos pagados PM→Primos ya debitan esos pesos en el control contable PM: **no se debitan otra vez**.
- Los **USD 760** de referencia **siguen físicamente bajo custodia de Matías** (por cuenta de PM según relato), y **no fueron vendidos para obtener esos ARS**. No hay entrada de ARS por FX a PM ni salida USD por FX.
- Los pares de cambio documentales en mayo–julio tenían estatus **X** (excluido); octubre no tiene el par. **No son cambios ejecutados**. La diferencia de julio entre borrador X y pago ARS no debe interpretarse como error de caja; el borrador de FX no es la fuente válida del pago.
- El pago personal, la imputación a PM y la custodia de USD son **ejes distintos**. No se crea automáticamente una cuenta de reembolso a Matías ni se inventa un banco o una liquidación FX.
- Separadamente, la asignación histórica de **ARS 200.000/mes a Alejandro se aplica a HH** y la de **USD 190/mes a Alejandro se compensa con deuda registrada hacia PM**. El efecto USD neto en el control PM de esa asignación/compensación es cero; **el stock de deuda no se recalcula** por este escenario.
- **Agosto/septiembre:** sin pagos y **sin retroactivo**. Julio ya pagado; no se duplica en el escenario.

**Corrección de conclusiones previas:** el puente que sumaba **ARS 1.188.284 y restaba USD 760** a PM era **incorrecto**, porque suponía una venta de dólares que no existió. **Invalidar y no usar** resultados locales G02 etiquetados `forward_v1`–`forward_v4` cuando modelan ese FX. El control contable del mart contiene el débito ARS registrado; **no hay crédito FX para agregar**. El control PM no constituye saldo físico bancario.

## Modelo aislado para meses próximos

Archivos en `accounting-workflows`:

- `scenarios/g02_chino_distribution_schedule.yaml`: calendario explícito de `YYYY-MM` por variante, sin atrasos; introduce la verdad histórica `historical_settlement` y la referencia ARS/USD **sin compraventa de divisas**. Para noviembre–febrero la referencia **ARS 1.500 por USD es ilustrativa, editable**.
- `accounting/scenarios/g02_chino_distribution.py`: seis asientos de **imputación económica**, no caja física confirmada, por cada mes seleccionado: PM→Primos **ARS 200.000 + equivalente ARS de USD 190**; PM→Alejandro ARS 200.000→HH (incluyendo espejo analítico en HH); PM→Alejandro USD 190 y Alejandro→PM USD 190 como compensación de deuda (cero efecto USD neto). **Ninguna entrada de Cambio/FX** ni doble imputación.
- `tests/test_g02_chino_distribution_schedule.py`: guardas de importes, meses no acumulables, hashes de fuente, seis entradas, custodia USD sin salida, pago personal ARS, ausencia de FX, HH y efecto en PM por moneda.
- Los escenarios **no suponen que Matías vuelva a adelantar dinero** en meses futuros. El pagador físico queda **NO ASIGNADO** hasta decisión, y **no se crea obligación con su sueldo**. El experimento calcula la reducción económica PM si se decidiera distribuir.
- El registro real `ledger_canonical_all_status.csv` no es modificado. Los CSV de overlay tienen estado `planeado` y no ingresan a producción.

**Fuentes pinned:** base `20261119T061512Z_FBPM` y `monthly_policy_inputs.csv` del snapshot `20261119T061512Z_FBPM_6m_sustainability`; verificaciones SHA-256 en YAML. Noviembre es actual/parcial, abril de 2027 carece de planes.

## Resultado del control contable PM ARS, tal cual imputado

| Meses seleccionados para distribución adicional | Primer control negativo | Mínimo PM ARS hasta marzo |
| --- | --- | ---: |
| Ninguno | 2027-02 | -254.849,43 |
| 2026-11 y 2026-12 | 2027-01 | -1.624.849,43 |
| 2026-11, 2026-12, 2027-01, 2027-02 | 2027-01 | **-2.994.849,43** |

Cada mes elegido **debilita el control PM en ARS 685.000** con la referencia ilustrativa. No debita USD por un FX inexistente; el offset de deuda de Alejandro queda separado. **Los resultados no certifican insolvencia real ni fuerzan financiación salarial**, pues la custodia física, bancos y sus obligaciones no están conciliados.

## Salidas vigentes y precauciones

Resultado vigente privado:

`~/Desktop/MEMOS/10_accounting_family/08_scenarios/g02_chino_distribution/20261119T061512Z_personal_MP_PM_charge_v5/`

- `summary.json`: resultados, saldos PM y atributos de financiación personal **históricos**.
- `historical_distribution_bridge.csv`: conciliación de piernas registradas, diferencias con borradores FX X explícitamente fuera de caja.
- `historical_personal_payment_and_usd_custody.csv`: trazabilidad de pesos personales imputados a PM y USD retenidos, no venta.
- `<variant>/scenario_overlay_ledger.csv`: seis asientos económicos futuros por mes elegido, sin FX.
- `<variant>/scenario_accounting_movements.csv`, `monthly_controls_by_box_currency.csv`, `hh_receipts_analytic.csv`: rendición y saldos por moneda/Box.
- `comparison.html` y `comparison.pdf`: reporte privado, jamás se publica en Viewer sin validación.

**Límite para G02 y reuniones:** un control PM negativo muestra **brecha de financiación bajo supuestos**, no saldo bancario negativo ni obligación personal de Matías. Para demostrar que debería adelantar salario se necesita probar disponibilidad de caja, vencimientos y ausencia de otras fuentes. No comunicar `forward_v1`–`forward_v4`: incorporan premisas equivocadas.
