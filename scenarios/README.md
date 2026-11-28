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

El corredor valida el SHA256 de la fuente exacta, crea una copia temporal aislada por variante, reemplaza solo los datos autorizados, conserva el ledger `pagado` sin modificar, y ejecuta las tres etapas offline estándar de cálculo: `run-materialize`, `run-debt`, `run-metrics`. Después llama únicamente al **renderizador oficial de tesorería** y al **motor PDF oficial**, sin modificar sus definiciones ni generar los otros 27 PDF del paquete general. El bundle completo (`run-reports`) fue descartado en este piloto tras fallar por un PDF especializado ajeno a la tesorería. Nunca invoca ingestión desde Google Sheets, `run-full`, `latest` ni publicación.

Después de ambas ejecuciones válidas, exporta de manera conjunta **cuatro archivos y nada más** en el outbox (HTML/PDF de cada variante). Todos los CSVs, métricas, logs de ejecución y demás artefactos intermedios se borran junto a su espacio temporal. El workspace se crea bajo `~/Desktop/MEMOS/10_accounting_family/08_scenarios/` (directorio contenedor del outbox), **no en `/tmp`**: Chromium como Snap usa otro espacio de `/tmp` y puede terminar con código 0 sin crear el PDF. Esta restricción de infraestructura se verificó con una prueba de renderización en ambas rutas. El propio YAML y este README se conservan en el repositorio como definiciones, no en el outbox.

## Interpretación importante

Por defecto, `make run-reports` y `run-compute` generan `treasury_accountability` con seis meses calendario incluyendo el mes de evaluación como mes 1. Ejemplo: noviembre 2026 → abril 2027 inclusive; el 1 de diciembre → mayo 2027. Stage A conserva `ledger_planning_snapshot.csv` por separado del ledger observado; los planes vigentes desde la evaluación hasta fin del sexto mes se procesan en una ejecución aislada. La fuente histórica, deuda reconocida, métricas anuales y publicación no se adelantan. Los meses sin planes no son una predicción de cero gastos. Para histórico exclusivamente, usar `python -m accounting.reports.build ... --treasury-observed-only`. El renderizador también admite un horizonte futuro **independiente** de las reglas de escenarios mediante los argumentos emparejados `--projection-as-of-date` y `--projection-through-date`, sobre insumos analíticos que ya incluyen los planes seleccionados. No genera planes por sí mismo.

El YAML del piloto incluye `projection.as_of_date=2026-11-19` y `projection.through_date=2027-03-31`. El runner deriva por separado los insumos de cada variante, conserva la historia pagada, y renderiza la misma tesorería con meses prospectivos diferenciados en gris. `base_planeada` es la proyección de referencia sin modificaciones; `chino_no_paga` aplica únicamente la hipótesis declarada. Se pueden revisar de forma independiente o comparar. No se altera el cierre observado ni se mezcla el catálogo oficial con una simulación.

Los archivos llevan `SIMULACION` en el nombre y una etiqueta de horizonte dentro del informe. **No son informes oficiales, saldos realizados ni prueba de impago.** Una futura integración en el visor debe preservar identidad del run, fecha de evaluación, horizonte y variante como metadatos separados y exige autorización de audiencia. El corte histórico y el horizonte prospectivo son parámetros diferentes.

## Qué queda abierto

- Paridad de un pipeline `sin operaciones` contra el correspondiente pipeline oficial (se comprueba la estructura y los artefactos, todavía no equivalencia visual binaria).
- Integración explícita de las proyecciones ya implementadas en la navegación privada del visor; no confundirlas con el catálogo observado.
- Generalizar los operadores declarativos (distribuciones, reasignaciones, costos imputados), con controles de conservación de recursos y compatibilidad.
- Probar el runner aislado contra un fixture completo de extremo a extremo y revisar la política de cierre documental antes de automatizar decenas de escenarios.

## Auditoría de tiempos (opcional, fuera del outbox)

Para diagnosticar el costo de varias corridas sin alterar los HTML/PDF por defecto, agregar
`--audit-file ~/Desktop/MEMOS/10_accounting_family/08_scenarios/diagnostics/<nombre-unico>.json`
al comando del corredor. No repetir el nombre: la auditoría no sobreescribe archivos.

El registro usa `perf_counter` en las etapas de Make y en los renderizadores HTML/PDF.
Cuando está activado, además emplea `accounting.diagnostics.timed_python` como
wrapper optativo de los comandos Python de Make, registrando módulos, tiempos y errores.
No instala profilers adicionales.

Para obtener un informe navegable:

```bash
python3 -m accounting.diagnostics.summarize_scenario_timings \
  ~/Desktop/MEMOS/10_accounting_family/08_scenarios/diagnostics/<nombre-unico>.json \
  --html ~/Desktop/MEMOS/10_accounting_family/08_scenarios/diagnostics/<nombre-unico>.html
```

Las duraciones de los subprocesos están **dentro** de las etapas de Make:
no se suman como si fueran tareas adicionales. La auditoría es optativa;
el outbox sigue conteniendo solamente los cuatro HTML/PDF. Primera medición
y principal ineficiencia hallada en `notes/scenario_performance_audit_20261118.md`.
