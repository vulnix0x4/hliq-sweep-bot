from __future__ import annotations

import json
from pathlib import Path

from hliq_bot.config import RuntimeConfig
from hliq_bot.ml.gate import MLGate


def _write_model(path: Path, train_samples: int | None = None) -> None:
    raw = {
        "model_type": "logistic_v1",
        "features": ["x"],
        "means": {"x": 0.0},
        "stds": {"x": 1.0},
        "weights": {"x": 1.0},
        "intercept": 0.0,
    }
    if train_samples is not None:
        raw["train_samples"] = int(train_samples)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(raw), encoding="utf-8")


def test_adaptive_threshold_tightens_after_bad_outcomes(tmp_path: Path) -> None:
    model_path = tmp_path / "model.json"
    _write_model(model_path)
    gate = MLGate(
        RuntimeConfig(
            ml_enabled=True,
            ml_provider="logistic",
            ml_model_path=str(model_path),
            ml_model_min_samples=0,
            ml_min_prob=0.62,
            ml_threshold_floor=0.52,
            ml_threshold_cap=0.85,
            ml_adaptive_threshold=True,
            ml_adaptive_window=20,
            ml_adaptive_min_trades=5,
            ml_adaptive_step=0.03,
        )
    )

    baseline = gate.evaluate({"x": 0.6, "regime_range": 1.0, "session_us": 1.0})
    assert baseline.allowed is True
    assert baseline.threshold == 0.62

    for _ in range(6):
        gate.register_outcome(probability=0.8, r_multiple=-1.0, regime="range", session="us")

    after = gate.evaluate({"x": 0.6, "regime_range": 1.0, "session_us": 1.0})
    assert after.allowed is False
    assert after.threshold > baseline.threshold


def test_ensemble_uses_logistic_when_codex_unavailable(tmp_path: Path) -> None:
    model_path = tmp_path / "model.json"
    _write_model(model_path)
    gate = MLGate(
        RuntimeConfig(
            ml_enabled=True,
            ml_provider="ensemble",
            ml_model_path=str(model_path),
            ml_model_min_samples=0,
            ml_min_prob=0.60,
            ml_adaptive_threshold=False,
            codex_cmd="/definitely/missing-codex-bin",
            ml_fail_open=False,
        )
    )
    decision = gate.evaluate({"x": 0.7, "regime_range": 1.0, "session_us": 1.0})
    assert gate.active is True
    assert decision.allowed is True
    assert "ensemble(" in decision.reason
    assert "logistic=" in decision.reason


def test_ensemble_fail_closed_when_no_models(tmp_path: Path) -> None:
    gate = MLGate(
        RuntimeConfig(
            ml_enabled=True,
            ml_provider="ensemble",
            ml_model_path=str(tmp_path / "missing.json"),
            ml_min_prob=0.60,
            ml_adaptive_threshold=False,
            codex_cmd="/definitely/missing-codex-bin",
            ml_fail_open=False,
        )
    )
    decision = gate.evaluate({"x": 0.7})
    assert decision.allowed is False
    assert "unavailable" in decision.reason


def test_logistic_ignores_undertrained_model_when_min_samples_not_met(tmp_path: Path) -> None:
    model_path = tmp_path / "model.json"
    _write_model(model_path, train_samples=6)
    gate = MLGate(
        RuntimeConfig(
            ml_enabled=True,
            ml_provider="logistic",
            ml_model_path=str(model_path),
            ml_model_min_samples=20,
            ml_fail_open=True,
        )
    )

    decision = gate.evaluate({"x": 0.9})
    assert decision.allowed is True
    assert decision.reason == "ml_model_unavailable"


def test_logistic_ignores_model_without_sample_metadata_when_min_samples_required(tmp_path: Path) -> None:
    model_path = tmp_path / "model.json"
    _write_model(model_path)
    gate = MLGate(
        RuntimeConfig(
            ml_enabled=True,
            ml_provider="logistic",
            ml_model_path=str(model_path),
            ml_model_min_samples=20,
            ml_fail_open=False,
        )
    )

    assert gate.active is False
    decision = gate.evaluate({"x": 0.9})
    assert decision.allowed is False
    assert decision.reason == "ml_model_unavailable"
