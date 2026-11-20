# HH1 v3 — Rediseño 2024–2026, noviembre provisional

**Estado de implementación:** productor nuevo, aditivo, sin reemplazar el
contrato v2, sin cambios en la fuente contable y sin actualizar `latest_*`.
**Informe orientado a Household de Balbín, exclusivamente `Box=Household`.**

## Cinco lecturas

1. **Caja virtual**: ventana enero 2024 – noviembre 2026. Barras mensuales
   verdes positivas para entradas, rojas negativas para salidas, grises
   firmadas para el neto; curva/puntos del saldo virtual acumulado en panel
   separado y con eje de valor **negativo**. El primer saldo al 01/01/2024 es
   **−ARS 4.179**, arrastrado desde 2023; **no se reinicia ficticiamente**.
   Repagos y capital de deuda influyen en la caja si pertenecen a sus flujos,
   aunque no son consumo.
2. **Aportes**: matriz de actores Matías, Candela, Alén, Alejandro y Claudia
   para 2024, 2025 y 2026 hasta el corte. Incluye tanto
   `funding_contribution` como `property_opex / direction=in` cuando se
   representa una contribución que financió directamente el gasto. La fila
   de gasto del perro se contabiliza **una sola vez** en los costos, y su
   contrapartida de apoyo Cande/Alén **una vez** en los aportes.
3. **Destino de recursos**: matriz 2024 / 2025 / 2026*/total por 11 rubros:
   Luz, Gas, Internet, Agua, ABL, Legales, Mantenimiento/Makro,
   Mantenimiento/Perro, Mantenimiento/Otros, Servicios sin identificar y
   Otros gastos. Se clasifica usando `ledger_canonical.issuer`, `Detalle`,
   `tag` y semántica, en una unión `tx_id` uno-a-uno. Sin porcentajes.
   Los repagos/deudas no se suman al cuadro de `property_opex out`.
4. **Evolución del gasto**: 35 meses desde enero de 2024 y promedios de cada
   año como total anual / **12** (incluso 2026 parcial). Son ARS nominales,
   no corregidos por inflación. El gráfico muestra segmentos de promedio.
5. **Boletas y comprobantes**: 134 movimientos de gasto en informe HTML con
   fecha, rubro, monto, condición de boleta/statement, prueba de pago y
   referencia de conciliación. Reconcilia por `tx_id`, fecha e importe con
   inventario privado antiguo: 14 PDFs para 10 gastos, 3 relaciones de prueba
   de pago aprobadas internamente y 11 relaciones candidatas. **Ningún PDF
   se copia a la edición familiar**, porque el manifest declara
   `publication_allowed=false` y `stakeholder_publication_allowed=false`.
   Los botones reales a PDFs son trabajo pendiente de autorización/QA; no se
   sustituyen por enlaces a la planilla compartida con otras cajas.

## Corte semántico (congelado en la fuente)

- **Run fuente:** `20261120T220100Z_HH`, corrida live read-only, Stage A con
  `cutoff_date=2026-11-20` y solo status `pagado`. Fecha última operación
  efectiva **18/11/2026**; no se incluyen futuras, entre ellas el asiento de
  Internet con fecha 23/11 aunque en el long ledger figure como pagado.
- **Horizon de presentación:** `2026-11-30`, **PROVISIONAL_AS_OF** y fecha
  de datos visible 20/11. No se declara que ya terminó noviembre. No se
  incluyen partidas planeadas (Edenor 80.000 ni perro adicional 80.000).
  Las mismas podrán incorporarse tras confirmación y una futura corrida.
- **Aportes 2026 hasta 20/11:** ARS **2.333.500,00**. **Gastos 2026**:
  ARS **3.872.449,43**. **Saldo virtual acumulado**: **−ARS
  1.978.030,01**.
- **Gastos operativos por año:** 2024 **2.101.083,84**, 2025
  **3.579.831,74**, 2026* **3.872.449,43**. Promedios dividiendo por 12:
  2024 **175.090,32**, 2025 **298.319,31**, 2026* **322.704,12**.
- **Perro nov 2026 pagado:** ARS **906.000** de costos operativos;
  contribuciones de contrapartida registradas: Candela **706.000** y Alén
  **150.000**. La declaración de «550.000 de Candela hoy» aún necesita
  conciliación contra estos asientos y no se suma por segunda vez.
- Los datos de HH **no prueban saldo bancario disponible ni adjudican
  individualmente un pagaré o crédito**: cada persona necesita liquidación
  y acuerdo/regla distinta.

## Compilación

```bash
# Desde accounting-workflows con PYTHONPATH apuntando al mismo código:
python3 -m accounting.reports.household_v3 \
  --run-root out/run/accounting/20261120T220100Z_HH \
  --out-dir /tmp/HH1_v3 \
  --data-as-of 2026-11-20 --month-end 2026-11-30 \
  --evidence-snapshot ops/report-director/private/20261119_HH_evidence_leader/evidence_snapshot/HH
```

La salida contiene exactamente `household_catalog.json`, cinco SVG
`charts/*.svg` y cinco HTML `reports/*.html`, todos SHA-256 validados
por el consumidor `accounting-viewer/scripts/build_hh1_v3.py`. El catálogo
usa `household_poc_catalog.v3`; v2 sigue disponible para auditorías previas.
Los informes tienen tabla numérica y enlace local para ampliar SVG en móvil.

**Gates:** scope HH exacto; no fechas posteriores al corte observado; 35
meses ARS continuos y reconciliados; entrada/costos y contribuciones
reconciliados por año; expediente documental privado no se publica;
no sobrescritura; no conversión FBPM. El viewer debe publicar solo desde
directorio limpio con los demás subpaths autorizados y mantener SSO.

## Decisiones para la siguiente edición

1. Determinar con responsables qué boletas y pruebas de pago pueden
   compartirse con cada familia/stakeholder; publicar documentos **solo**
   si el contrato documental cambia de `publication_allowed=false`.
2. Conciliar el «550.000» de Candela con las contrapartidas registradas del
   perro; no duplicar aportes ni reconocer deudas automáticamente.
3. Confirmar el cierre real del 30/11 con nuevas operaciones en diciembre,
   separar asientos programados de pagos ejecutados y regenerar una corrida.
4. Revisar si los gastos de ABL, perro, reparación edilicia y otros son
   distribuibles y quién los debe pagar; ese reparto no es objeto de HH1 v3.
