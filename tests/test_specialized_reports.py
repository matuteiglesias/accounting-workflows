from accounting.reports.specialized.spec import REPORT_SPECS


def test_specialized_vertical_has_small_explicit_recipe_set():
    assert [spec.report_id for spec in REPORT_SPECS] == [
        "pm_tax_accountability",
        "pm_services_accountability",
        "stakeholder_support",
        "distributions_by_recipient",
    ]
    assert all(spec.question and spec.caveat for spec in REPORT_SPECS)
