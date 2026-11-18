# Escenarios contables — contrato humano/máquina (piloto)

La descripción YAML es **la interfaz de revisión humana**. No es un script de Python ni permite expresiones libres, `eval`, callbacks o SQL. Una gramática pequeña es implementada y validada por `accounting/scenarios/runner.py`.

## Primer caso

`chino_juan_no_paga_desde_2026_12.yaml` describe dos variantes de **la misma fuente exacta**:

1. `base_planeada`: la copia del `ledger_canonical_all_status.csv` conserva todas las previsiones.
2. `chino_no_paga`: se excluyen únicamente los 8 cobros `planeado` de Chino Juan entre diciembre 2026 y marzo 2027 (ARS 400.000 + USD 380 por mes).

Los filtros por `Box`, `payer`, `receiver`, `Flujo`, `Tipo`, `Lugar`, `Detalle`, `status`, `tag` y meses deben coincidir. La suma por moneda y cantidad de filas deben concordar con `expect`; si cambió el ledger, la ejecución se interrumpe antes de producir informes.

## Cómo correr

```bash
cd ~/repos/accounting-workflows
python3 -m accounting.scenarios.runner \
  --scenario scenarios/chino_juan_no_paga_desde_2026_12.yaml \
  --outbox ~/Desktop/MEMOS/10_accounting_family/08_scenarios/outbox
```

El corredor valida el SHA256 de la fuente exacta, crea una copia temporal aislada por variante, reemplaza solo los datos autorizados, conserva el ledger `pagado` sin modificar, y ejecuta las cuatro etapas offline existentes: `run-materialize`, `run-debt`, `run-metrics`, `run-reports`. Nunca invoca ingestión desde Google Sheets, `run-full`, `latest` ni publicación.

Después de ambas ejecuciones válidas, exporta de manera conjunta **cuatro archivos y nada más** en el outbox (HTML/PDF de cada variante). Todos los CSVs, métricas, logs de ejecución y demás artefactos intermedios se borran junto a su espacio temporal. El propio YAML y este README se conservan en el repositorio como definiciones, no en el outbox.

## Interpretación importante

El informe solicitado, `treasury_accountability`, todavía termina en octubre 2026. Su contrato de cierre solo incluye operaciones `pagado` y **no incorpora los cobros `planeado` de diciembre 2026-marzo 2027**. Como acordamos, no modificamos ni extendemos ese informe ahora. En consecuencia, los dos documentos pueden mostrar las mismas cifras de tesorería histórica; eso **no significa** que la hipótesis no se aplicó a los datos temporales. Nunca anunciar una pérdida visible de liquidez futura en este informe antes de ampliar su horizonte de presentación.

Al abrir/exportar fuera del visor, los archivos llevan `SIMULACION` en el nombre: **no son informes oficiales ni prueba de impago**. El HTML/PDF interno se genera exactamente con el renderizador oficial, sin ramas de presentación propias para el simulador.

## Qué queda abierto

- Paridad de un pipeline `sin operaciones` contra el correspondiente pipeline oficial (se comprueba la estructura y los artefactos, todavía no equivalencia visual binaria).
- La ampliación futura de los reportes, diseñada de forma independiente del corredor.
- Generalizar los operadores declarativos (distribuciones, reasignaciones, costos imputados), con controles de conservación de recursos y compatibilidad.
- Probar el runner aislado contra un fixture completo de extremo a extremo y revisar la política de cierre documental antes de automatizar decenas de escenarios.
