from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import json
import math
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import time

from hliq_bot.config import RuntimeConfig


@dataclass(slots=True)
class GateDecision:
    allowed: bool
    probability: float
    reason: str
    threshold: float = 0.0


class LogisticGateModel:
    def __init__(
        self,
        features: list[str],
        means: dict[str, float],
        stds: dict[str, float],
        weights: dict[str, float],
        intercept: float,
        train_samples: int | None = None,
    ) -> None:
        self.features = features
        self.means = means
        self.stds = stds
        self.weights = weights
        self.intercept = intercept
        self.train_samples = train_samples

    @classmethod
    def load(cls, path: str) -> "LogisticGateModel":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if raw.get("model_type") != "logistic_v1":
            raise ValueError(f"Unsupported model_type: {raw.get('model_type')!r}")
        return cls(
            features=list(raw["features"]),
            means=dict(raw.get("means", {})),
            stds=dict(raw.get("stds", {})),
            weights=dict(raw["weights"]),
            intercept=float(raw.get("intercept", 0.0)),
            train_samples=_safe_int(raw.get("train_samples")),
        )

    def predict_prob(self, values: dict[str, float]) -> float:
        z = self.intercept
        for name in self.features:
            v = float(values.get(name, 0.0))
            mu = float(self.means.get(name, 0.0))
            sigma = float(self.stds.get(name, 1.0))
            if sigma <= 1e-9:
                sigma = 1.0
            x = (v - mu) / sigma
            z += float(self.weights.get(name, 0.0)) * x
        # Clamp z for numeric stability.
        z = max(-30.0, min(30.0, z))
        return 1.0 / (1.0 + math.exp(-z))


class CodexCliGateModel:
    def __init__(self, cfg: RuntimeConfig) -> None:
        self.cfg = cfg
        cmd = shlex.split(cfg.codex_cmd.strip()) if cfg.codex_cmd.strip() else ["codex"]
        self._cmd = cmd if cmd else ["codex"]
        self._schema_path = Path(cfg.runtime_dir) / "codex_gate_schema.json"
        self._last_call_ms = 0
        self._calls_ms: deque[int] = deque()
        self._available = shutil.which(self._cmd[0]) is not None
        if self._available:
            self._ensure_schema()

    @property
    def available(self) -> bool:
        return self._available

    def predict_prob(self, features: dict[str, float]) -> tuple[float, str]:
        if not self._available:
            raise RuntimeError("codex_cli_not_found")

        now_ms = int(time.time() * 1000)
        min_interval_ms = max(0, self.cfg.codex_min_interval_sec) * 1000
        if self._last_call_ms > 0 and now_ms - self._last_call_ms < min_interval_ms:
            raise RuntimeError("codex_local_rate_limit")

        self._evict_old_calls(now_ms)
        if len(self._calls_ms) >= max(1, self.cfg.codex_max_calls_hourly):
            raise RuntimeError("codex_hourly_cap")

        prompt = self._build_prompt(features)
        with tempfile.TemporaryDirectory(prefix="codex_gate_") as tmp:
            out_path = Path(tmp) / "out.json"
            cmd = [
                *self._cmd,
                "exec",
                "--skip-git-repo-check",
                "--sandbox",
                "read-only",
                "--ephemeral",
                "--model",
                self.cfg.codex_model,
                "--output-schema",
                str(self._schema_path),
                "-o",
                str(out_path),
                prompt,
            ]
            if self.cfg.codex_profile.strip():
                cmd.extend(["--profile", self.cfg.codex_profile.strip()])

            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=max(1.0, float(self.cfg.codex_timeout_sec)),
            )
            if proc.returncode != 0:
                err = (proc.stderr or proc.stdout or "").strip()
                msg = err.splitlines()[-1] if err else f"exit_{proc.returncode}"
                raise RuntimeError(f"codex_exec_failed:{_clip_reason(msg)}")

            if not out_path.exists():
                raise RuntimeError("codex_no_output")
            raw = out_path.read_text(encoding="utf-8").strip()
            if not raw:
                raise RuntimeError("codex_empty_output")
            data = json.loads(raw)

        prob = float(data.get("probability", 0.5))
        prob = max(0.0, min(1.0, prob))
        reason = _clip_reason(str(data.get("reason", "codex")))
        self._last_call_ms = now_ms
        self._calls_ms.append(now_ms)
        return prob, reason

    def _evict_old_calls(self, now_ms: int) -> None:
        cutoff = now_ms - 3_600_000
        while self._calls_ms and self._calls_ms[0] < cutoff:
            self._calls_ms.popleft()

    def _ensure_schema(self) -> None:
        self._schema_path.parent.mkdir(parents=True, exist_ok=True)
        if self._schema_path.exists():
            return
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "allow": {"type": "boolean"},
                "probability": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "reason": {"type": "string"},
            },
            "required": ["allow", "probability", "reason"],
            "additionalProperties": False,
        }
        self._schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")

    def _build_prompt(self, features: dict[str, float]) -> str:
        clean: dict[str, float] = {}
        for k, v in features.items():
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(fv):
                continue
            clean[k] = round(fv, 6)

        return (
            "You are a conservative trade-quality gate for BTC perpetual sweep-rejection entries.\n"
            "Estimate the probability that this setup will have positive R-multiple within the next 3-30 minutes.\n"
            "Use only the provided features. Penalize high spread, high short-term volatility, low confidence,\n"
            "and weak reward-to-risk. If uncertain, return a lower probability.\n"
            f"features={json.dumps(clean, sort_keys=True)}\n"
            "Return JSON that matches the provided schema."
        )


class MLGate:
    def __init__(self, cfg: RuntimeConfig) -> None:
        self.cfg = cfg
        self._model: LogisticGateModel | None = None
        self._codex: CodexCliGateModel | None = None
        adaptive_window = max(5, cfg.ml_adaptive_window)
        self._recent_outcomes: deque[tuple[float, float]] = deque(maxlen=adaptive_window)
        self._recent_regime: dict[str, deque[tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=adaptive_window)
        )
        self._recent_session: dict[str, deque[tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=adaptive_window)
        )
        self.reload()

    def reload(self) -> None:
        self._model = None
        self._codex = None
        if not self.cfg.ml_enabled:
            return
        provider = self.cfg.ml_provider.strip().lower()
        if provider in {"codex_cli", "ensemble"}:
            self._codex = CodexCliGateModel(self.cfg)
        if provider in {"logistic", "ensemble"}:
            try:
                model = LogisticGateModel.load(self.cfg.ml_model_path)
                min_samples = max(0, int(getattr(self.cfg, "ml_model_min_samples", 0)))
                train_samples = model.train_samples
                if min_samples > 0 and (train_samples is None or train_samples < min_samples):
                    self._model = None
                else:
                    self._model = model
            except Exception:
                self._model = None

    @property
    def active(self) -> bool:
        if not self.cfg.ml_enabled:
            return False
        provider = self.cfg.ml_provider.strip().lower()
        if provider == "codex_cli":
            return self._codex is not None and self._codex.available
        if provider == "ensemble":
            return self._model is not None or (self._codex is not None and self._codex.available)
        return self._model is not None

    def evaluate(self, features: dict[str, float]) -> GateDecision:
        if not self.cfg.ml_enabled:
            return GateDecision(True, 1.0, "ml_disabled", threshold=0.0)
        provider = self.cfg.ml_provider.strip().lower()
        regime, session = self._context_from_features(features)
        threshold = self._effective_threshold(regime, session)
        if provider == "codex_cli":
            return self._evaluate_codex(features, threshold)
        if provider == "ensemble":
            return self._evaluate_ensemble(features, threshold)
        return self._evaluate_logistic(features, threshold)

    def register_outcome(self, probability: float, r_multiple: float, regime: str = "", session: str = "") -> None:
        prob = _clamp_probability(probability)
        try:
            r_val = float(r_multiple)
        except (TypeError, ValueError):
            return
        if not math.isfinite(r_val):
            return
        self._recent_outcomes.append((prob, r_val))
        regime_key = (regime or "").strip().lower()
        if regime_key:
            self._recent_regime[regime_key].append((prob, r_val))
        session_key = (session or "").strip().lower()
        if session_key:
            self._recent_session[session_key].append((prob, r_val))

    def _evaluate_logistic(self, features: dict[str, float], threshold: float) -> GateDecision:
        if self._model is None:
            if self.cfg.ml_fail_open:
                return GateDecision(True, 0.5, "ml_model_unavailable", threshold=threshold)
            return GateDecision(False, 0.0, "ml_model_unavailable", threshold=threshold)
        prob = self._model.predict_prob(features)
        allowed = prob >= threshold
        return GateDecision(allowed, prob, "ml_allow" if allowed else "ml_deny", threshold=threshold)

    def _evaluate_codex(self, features: dict[str, float], threshold: float) -> GateDecision:
        if self._codex is None or not self._codex.available:
            if self.cfg.ml_fail_open:
                return GateDecision(True, 0.5, "codex_unavailable", threshold=threshold)
            return GateDecision(False, 0.0, "codex_unavailable", threshold=threshold)
        try:
            prob, reason = self._codex.predict_prob(features)
            allowed = prob >= threshold
            return GateDecision(allowed, prob, f"codex:{reason}", threshold=threshold)
        except Exception as exc:
            reason = _clip_reason(str(exc))
            if self.cfg.ml_fail_open:
                return GateDecision(True, 0.5, f"codex_error:{reason}", threshold=threshold)
            return GateDecision(False, 0.0, f"codex_error:{reason}", threshold=threshold)

    def _evaluate_ensemble(self, features: dict[str, float], threshold: float) -> GateDecision:
        # In warmup phases a local model may not exist yet. If fail-open is enabled,
        # avoid letting codex-only scores hard-veto every setup before we gather outcomes.
        if self._model is None and self.cfg.ml_fail_open:
            return GateDecision(True, 0.5, "ensemble_logistic_unavailable_fail_open", threshold=threshold)

        probs: dict[str, float] = {}
        notes: list[str] = []

        if self._model is not None:
            probs["logistic"] = _clamp_probability(self._model.predict_prob(features))
        else:
            notes.append("logistic_unavailable")

        if self._codex is None or not self._codex.available:
            notes.append("codex_unavailable")
        else:
            try:
                codex_prob, codex_reason = self._codex.predict_prob(features)
                probs["codex"] = _clamp_probability(codex_prob)
                notes.append(f"codex:{_clip_reason(codex_reason, limit=60)}")
            except Exception as exc:
                notes.append(f"codex_error:{_clip_reason(str(exc), limit=60)}")

        if not probs:
            reason = _clip_reason("ensemble_unavailable;" + ",".join(notes))
            if self.cfg.ml_fail_open:
                return GateDecision(True, 0.5, reason, threshold=threshold)
            return GateDecision(False, 0.0, reason, threshold=threshold)

        if "codex" in probs and "logistic" in probs:
            codex_w = max(0.0, min(1.0, float(self.cfg.ml_ensemble_codex_weight)))
            logit_w = 1.0 - codex_w
            prob = (probs["codex"] * codex_w) + (probs["logistic"] * logit_w)
        elif "codex" in probs:
            prob = probs["codex"]
        else:
            prob = probs["logistic"]

        allowed = prob >= threshold
        parts = [f"{k}={v:.3f}" for k, v in probs.items()]
        reason = _clip_reason("ensemble(" + ",".join(parts + notes) + ")")
        return GateDecision(allowed, prob, reason, threshold=threshold)

    def _effective_threshold(self, regime: str, session: str) -> float:
        base = _clamp_threshold(self.cfg.ml_min_prob, self.cfg.ml_threshold_floor, self.cfg.ml_threshold_cap)
        if not self.cfg.ml_adaptive_threshold:
            return base

        adjust = self._threshold_adjust(self._recent_outcomes)
        if regime:
            adjust += 0.5 * self._threshold_adjust(self._recent_regime.get(regime))
        if session:
            adjust += 0.35 * self._threshold_adjust(self._recent_session.get(session))

        return _clamp_threshold(base + adjust, self.cfg.ml_threshold_floor, self.cfg.ml_threshold_cap)

    def _threshold_adjust(self, samples: deque[tuple[float, float]] | None) -> float:
        if samples is None:
            return 0.0
        min_samples = max(2, self.cfg.ml_adaptive_min_trades)
        if len(samples) < min_samples:
            return 0.0
        n = len(samples)
        avg_prob = sum(p for p, _ in samples) / n
        win_rate = sum(1.0 if r > 0.0 else 0.0 for _, r in samples) / n
        avg_r = sum(r for _, r in samples) / n
        gap = avg_prob - win_rate
        step = max(0.0, float(self.cfg.ml_adaptive_step))
        adjust = 0.0

        if avg_r <= -0.10:
            adjust += step
        elif avg_r >= 0.12:
            adjust -= step * 0.5

        if gap >= 0.08:
            adjust += step
        elif gap <= -0.10 and avg_r >= 0.0:
            adjust -= step * 0.35
        return adjust

    def _context_from_features(self, features: dict[str, float]) -> tuple[str, str]:
        regime = "unknown"
        if _safe_float(features.get("regime_trend", 0.0)) >= 0.5:
            regime = "trend"
        elif _safe_float(features.get("regime_range", 0.0)) >= 0.5:
            regime = "range"

        session = "unknown"
        if _safe_float(features.get("session_us", 0.0)) >= 0.5:
            session = "us"
        elif _safe_float(features.get("session_eu", 0.0)) >= 0.5:
            session = "eu"
        elif _safe_float(features.get("session_asia", 0.0)) >= 0.5:
            session = "asia"
        return regime, session


def _clip_reason(text: str, limit: int = 140) -> str:
    t = " ".join((text or "").strip().split())
    if len(t) <= limit:
        return t
    return t[: limit - 3] + "..."


def _clamp_probability(value: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.5
    if not math.isfinite(v):
        return 0.5
    return max(0.0, min(1.0, v))


def _clamp_threshold(value: float, floor: float, cap: float) -> float:
    lo = max(0.0, min(1.0, float(floor)))
    hi = max(lo, min(1.0, float(cap)))
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = lo
    if not math.isfinite(v):
        v = lo
    return max(lo, min(hi, v))


def _safe_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(out):
        return 0.0
    return out


def _safe_int(value: object) -> int | None:
    try:
        out = int(value)
    except (TypeError, ValueError):
        return None
    return out if out >= 0 else None
