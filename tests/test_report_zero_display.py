import pytest
from accounting.reports.annual_management.render import MetricStore
from accounting.reports.debt_accountability.render import _money as debt_money
from accounting.reports.specialized.render import _money as specialized_money
from accounting.reports.treasury_transaction_tape.render import _money as tape_money
from accounting.reports.treasury_accountability.render import _fmt_num

@pytest.mark.parametrize("fmt", ["money", "usd", "count", "ratio", "fraction"])
def test_annual_zero_is_bare_dash_but_missing_is_not(fmt):
    assert MetricStore._format(0, "available", fmt, "USD") == "-"
    assert MetricStore._format(0, "structural_zero", fmt, "ARS") == "-"
    assert MetricStore._format(0, "unavailable", fmt, "USD") == "N/A"
    assert MetricStore._format(None, "available", fmt, "USD") == "N/A"
    assert MetricStore._format(None, "not_applicable", fmt, "USD") == "—"
    assert MetricStore._format(1, "available", fmt, "USD") != "-"

@pytest.mark.parametrize("render", [debt_money, lambda v:specialized_money(v,"USD"), tape_money, _fmt_num])
def test_table_formatters_distinguish_zero_and_missing(render):
    assert render(0) == "-"
    assert render(None) == "N/A"
    assert render(-0.01) != "-"
