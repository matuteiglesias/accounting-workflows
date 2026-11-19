from __future__ import annotations

import pandas as pd

from accounting.box_control import select_box_motor_control_period


def test_native_box_motor_selector_partitions_currency_and_accepts_zero_negative() -> None:
    motor = pd.DataFrame([
        {"TimePeriod": "2026-01", "TimePeriod_end": "2026-01-31", "Box": "PM", "Currency": "ARS", "cum_net": 0},
        {"TimePeriod": "2026-02", "TimePeriod_end": "2026-02-28", "Box": "PM", "Currency": "ARS", "cum_net": -25},
        {"TimePeriod": "2026-02", "TimePeriod_end": "2026-02-28", "Box": "PM", "Currency": "USD", "cum_net": 10},
    ])
    assert select_box_motor_control_period(motor, period="2026-01", currency="ARS", box="PM").value == 0
    assert select_box_motor_control_period(motor, period="2026-02", currency="ARS", box="PM").value == -25
    assert select_box_motor_control_period(motor, period="2026-02", currency="USD", box="PM").value == 10


def test_native_box_motor_selector_rejects_duplicate_malformed_and_cutoff_rows() -> None:
    motor = pd.DataFrame([
        {"TimePeriod": "2026-02", "TimePeriod_end": "2026-02-28", "Box": "PM", "Currency": "ARS", "cum_net": 10},
        {"TimePeriod": "2026-02", "TimePeriod_end": "2026-02-28", "Box": "PM", "Currency": "ARS", "cum_net": 10},
        {"TimePeriod": "2026-03", "TimePeriod_end": "not-a-date", "Box": "PM", "Currency": "ARS", "cum_net": 20},
    ])
    duplicate = select_box_motor_control_period(motor, period="2026-02", currency="ARS", box="PM")
    assert (duplicate.status, duplicate.reason) == ("unavailable", "duplicate_box_motor_snapshot")
    malformed = select_box_motor_control_period(motor, period="2026-03", currency="ARS", box="PM")
    assert (malformed.status, malformed.reason) == ("unsupported", "malformed_box_motor_period")
    cutoff = select_box_motor_control_period(motor, period="2026-02", currency="ARS", box="PM", cutoff_date="2026-01-31")
    assert (cutoff.status, cutoff.reason) == ("unavailable", "period_after_cutoff")
