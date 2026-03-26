"""
OpenAI Prediction Service
Generates all vehicle health predictions (KPIs, component health, degradation contributors,
maintenance decision) by sending telemetry features directly to OpenAI GPT.

Returns the EXACT same JSON schema as PredictionService so the frontend needs zero changes.

Usage:
    python app.py --mode openai
"""
import os
import json
from typing import Dict, List

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class OpenAIPredictionService:
    """
    Drop-in replacement for PredictionService.
    Uses OpenAI GPT instead of LSTM + Autoencoder models to generate predictions.
    """

    MODEL = "gpt-4o-mini"

    # Keep feature names identical to PredictionService for context building
    FEATURE_NAMES = [
        "State of Charge",       # 0  — normalized 0-1 (1 = 100% charged)
        "State of Health",       # 1  — normalized 0-1 (1 = brand new battery)
        "Battery Voltage",       # 2  — volts
        "Battery Current",       # 3  — amps (negative = discharging)
        "Battery Temperature",   # 4  — °C
        "Motor Temperature",     # 5  — °C
        "Motor Vibration",       # 6  — g (acceleration units, 0-3 typical)
        "Motor RPM",             # 7  — revolutions per minute
        "Brake Pad Wear",        # 8  — normalized 0-1 (1 = fully worn)
        "Power Stress",          # 9  — normalized 0-1 (1 = maximum stress)
        "Usage Intensity",       # 10 — 0-100 scale
        "Health Trend",          # 11 — normalized 0-1 (1 = improving, 0 = degrading)
    ]

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.client = None

        if not OPENAI_AVAILABLE:
            print("⚠️  openai package not installed. OpenAI prediction mode unavailable.")
            return

        if self.api_key:
            try:
                self.client = OpenAI(api_key=self.api_key)
                print("✅ OpenAI Prediction Service initialized successfully")
            except Exception as e:
                print(f"⚠️  Failed to initialize OpenAI Prediction Service: {e}")
        else:
            print("⚠️  OPENAI_API_KEY not set. OpenAI prediction mode unavailable.")

    # -------------------------------------------------------------------------
    # Public API — identical signature to PredictionService.predict()
    # -------------------------------------------------------------------------

    def predict(self, features: List[float]) -> Dict:
        """
        Generate vehicle health prediction using OpenAI GPT.

        Args:
            features: List of 12 normalized telemetry feature values

        Returns:
            Dictionary matching PredictionService output schema exactly
        """
        if not self.client:
            raise RuntimeError("OpenAI client not initialized. Check OPENAI_API_KEY.")

        if len(features) != 12:
            raise ValueError(f"Expected 12 features, got {len(features)}")

        prompt = self._build_prediction_prompt(features)

        try:
            response = self.client.chat.completions.create(
                model=self.MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert AI vehicle diagnostics system. "
                            "You receive raw telemetry sensor values and return a "
                            "structured JSON health analysis. "
                            "Return ONLY valid JSON — no markdown, no extra text."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                response_format={"type": "json_object"},
            )

            raw = response.choices[0].message.content.strip()
            result = json.loads(raw)

            # Validate and sanitize the response to match expected schema
            return self._sanitize_result(result, features)

        except json.JSONDecodeError as e:
            print(f"⚠️  OpenAI returned invalid JSON: {e}. Falling back to rule-based.")
            return self._rule_based_fallback(features)
        except Exception as e:
            print(f"⚠️  OpenAI prediction error: {e}. Falling back to rule-based.")
            return self._rule_based_fallback(features)

    def load_models(self):
        """No-op — keeps interface compatible with PredictionService."""
        print("ℹ️  OpenAI mode: no local models to load.")

    # -------------------------------------------------------------------------
    # Prompt building
    # -------------------------------------------------------------------------

    def _build_prediction_prompt(self, features: List[float]) -> str:
        feature_lines = "\n".join(
            f"  - {self.FEATURE_NAMES[i]}: {features[i]}"
            for i in range(12)
        )

        return f"""
You are analyzing an electric vehicle's health using the following 12 telemetry sensor readings:

{feature_lines}

Feature value ranges for context:
  - State of Charge: 0.0 (empty) to 1.0 (fully charged)
  - State of Health: 0.0 (end of life) to 1.0 (new battery)
  - Battery Voltage: ~300-420 V typical EV range
  - Battery Current: negative = discharging, positive = charging. Typical: -200 to +100 A
  - Battery Temperature: safe range 15-45°C; >50°C = high risk
  - Motor Temperature: safe range <70°C; >90°C = critical
  - Motor Vibration: 0-3 g; >1.5 g = concerning
  - Motor RPM: 0-6000; >4000 = high stress
  - Brake Pad Wear: 0 = new, 1 = fully worn (replace immediately)
  - Power Stress: 0 = idle, 1 = maximum continuous stress
  - Usage Intensity: 0-100 scale; >70 = heavy usage
  - Health Trend: 0 = rapidly degrading, 1 = stable or improving

Based on this data, return a JSON object with EXACTLY this structure:

{{
  "kpis": {{
    "failure_probability": <float: 0-100, percentage probability of failure soon>,
    "remaining_useful_life": <float: estimated remaining cycles, typically 10-250>,
    "anomaly_score": <float: 0-1, how anomalous this reading is; 0=normal, 1=very anomalous>,
    "overall_health": <float: 0-100, overall vehicle health score>
  }},
  "component_health": {{
    "Battery System": {{
      "score": <float: 0-100>,
      "status": <"healthy" | "degrading" | "critical">,
      "icon": <"🟢" | "🟡" | "🔴">
    }},
    "Thermal System": {{
      "score": <float: 0-100>,
      "status": <"healthy" | "degrading" | "critical">,
      "icon": <"🟢" | "🟡" | "🔴">
    }},
    "Motor System": {{
      "score": <float: 0-100>,
      "status": <"healthy" | "degrading" | "critical">,
      "icon": <"🟢" | "🟡" | "🔴">
    }},
    "Braking System": {{
      "score": <float: 0-100>,
      "status": <"healthy" | "degrading" | "critical">,
      "icon": <"🟢" | "🟡" | "🔴">
    }},
    "Usage Stress": {{
      "score": <float: 0-100>,
      "status": <"healthy" | "degrading" | "critical">,
      "icon": <"🟢" | "🟡" | "🔴">
    }}
  }},
  "degradation_contributors": [
    {{"feature": "<name>", "value": <float>, "importance": <float: 0-1>}},
    {{"feature": "<name>", "value": <float>, "importance": <float: 0-1>}},
    {{"feature": "<name>", "value": <float>, "importance": <float: 0-1>}}
  ],
  "maintenance_decision": {{
    "level": <"normal" | "soon" | "warning" | "immediate">,
    "message": <short human-friendly message string>,
    "description": <1-2 sentence description>,
    "color": <"green" | "yellow" | "orange" | "red">
  }},
  "should_alert": <true | false>,
  "alert_severity": <"normal" | "warning" | "critical">
}}

Rules:
- degradation_contributors must list the top 3 features that are most concerning
- should_alert = true only when failure_probability >= 70 OR remaining_useful_life <= 30
- alert_severity: "critical" if failure_probability>=70 or rul<=30, "warning" if >=40 or rul<=60, else "normal"
- All score values are floats, not strings
- Return ONLY the JSON, no markdown fences or extra text
"""

    # -------------------------------------------------------------------------
    # Response sanitization
    # -------------------------------------------------------------------------

    def _sanitize_result(self, result: Dict, features: List[float]) -> Dict:
        """Ensure the result has all required fields with correct types."""

        # KPIs
        kpis = result.get("kpis", {})
        sanitized_kpis = {
            "failure_probability": round(float(kpis.get("failure_probability", 50.0)), 2),
            "remaining_useful_life": round(float(kpis.get("remaining_useful_life", 100.0)), 1),
            "anomaly_score": round(float(kpis.get("anomaly_score", 0.1)), 4),
            "overall_health": round(float(kpis.get("overall_health", 50.0)), 1),
        }

        # Component health
        default_components = {
            "Battery System": {"score": 75.0, "status": "degrading", "icon": "🟡"},
            "Thermal System": {"score": 75.0, "status": "degrading", "icon": "🟡"},
            "Motor System": {"score": 75.0, "status": "degrading", "icon": "🟡"},
            "Braking System": {"score": 75.0, "status": "degrading", "icon": "🟡"},
            "Usage Stress": {"score": 75.0, "status": "degrading", "icon": "🟡"},
        }
        raw_ch = result.get("component_health", {})
        component_health = {}
        for name, default in default_components.items():
            ch = raw_ch.get(name, default)
            score = round(float(ch.get("score", default["score"])), 1)
            if score >= 80:
                status, icon = "healthy", "🟢"
            elif score >= 50:
                status, icon = "degrading", "🟡"
            else:
                status, icon = "critical", "🔴"
            component_health[name] = {"score": score, "status": status, "icon": icon}

        # Degradation contributors
        raw_contribs = result.get("degradation_contributors", [])
        contributors = []
        for c in raw_contribs[:3]:
            contributors.append({
                "feature": str(c.get("feature", "Unknown")),
                "value": round(float(c.get("value", 0.0)), 3),
                "importance": round(float(c.get("importance", 0.0)), 3),
            })

        # Maintenance decision
        raw_md = result.get("maintenance_decision", {})
        level = raw_md.get("level", "normal")
        if level not in ("normal", "soon", "warning", "immediate"):
            level = "normal"
        maintenance_decision = {
            "level": level,
            "message": str(raw_md.get("message", "Vehicle status: normal")),
            "description": str(raw_md.get("description", "All systems within normal parameters.")),
            "color": raw_md.get("color", "green"),
        }

        # Alert fields
        fp = sanitized_kpis["failure_probability"]
        rul = sanitized_kpis["remaining_useful_life"]
        should_alert = bool(result.get("should_alert", fp >= 70 or rul <= 30))
        alert_severity = str(result.get("alert_severity", "normal"))
        if alert_severity not in ("normal", "warning", "critical"):
            alert_severity = "critical" if fp >= 70 or rul <= 30 else ("warning" if fp >= 40 or rul <= 60 else "normal")

        return {
            "kpis": sanitized_kpis,
            "component_health": component_health,
            "degradation_contributors": contributors,
            "maintenance_decision": maintenance_decision,
            "should_alert": should_alert,
            "alert_severity": alert_severity,
        }

    # -------------------------------------------------------------------------
    # Rule-based fallback (if OpenAI call fails)
    # -------------------------------------------------------------------------

    def _rule_based_fallback(self, features: List[float]) -> Dict:
        """Simple heuristic prediction used when the OpenAI call fails."""
        soc, soh = features[0], features[1]
        bat_temp, mot_temp = features[4], features[5]
        brake_wear = features[8]
        health_trend = features[11]

        overall_health = round((soc * 0.25 + soh * 0.35 + health_trend * 0.40) * 100, 1)
        failure_prob = round(max(5, min(95, (100 - overall_health) * 0.9)), 2)
        rul = round(max(10, overall_health * 2.5), 1)
        anomaly_score = round(max(0, 1 - overall_health / 100), 4)

        def _status(score):
            if score >= 80:
                return "healthy", "🟢"
            elif score >= 50:
                return "degrading", "🟡"
            return "critical", "🔴"

        bat_score = round((soc + soh) / 2 * 100, 1)
        thermal_score = round(max(0, min(100, (1 - max(bat_temp - 45, 0) / 25) * 100)), 1)
        motor_score = round(max(0, min(100, (1 - max(mot_temp - 70, 0) / 40) * 100)), 1)
        brake_score = round((1 - brake_wear) * 100, 1)
        usage_score = round((1 - features[10] / 100) * 100, 1)

        level = "immediate" if failure_prob >= 70 else ("soon" if failure_prob >= 40 else "normal")
        color = {"immediate": "red", "soon": "yellow", "normal": "green"}.get(level, "green")
        alert_severity = "critical" if failure_prob >= 70 or rul <= 30 else ("warning" if failure_prob >= 40 or rul <= 60 else "normal")

        bat_st, bat_ic = _status(bat_score)
        th_st, th_ic = _status(thermal_score)
        mo_st, mo_ic = _status(motor_score)
        br_st, br_ic = _status(brake_score)
        us_st, us_ic = _status(usage_score)

        return {
            "kpis": {
                "failure_probability": failure_prob,
                "remaining_useful_life": rul,
                "anomaly_score": anomaly_score,
                "overall_health": overall_health,
            },
            "component_health": {
                "Battery System": {"score": bat_score, "status": bat_st, "icon": bat_ic},
                "Thermal System": {"score": thermal_score, "status": th_st, "icon": th_ic},
                "Motor System": {"score": motor_score, "status": mo_st, "icon": mo_ic},
                "Braking System": {"score": brake_score, "status": br_st, "icon": br_ic},
                "Usage Stress": {"score": usage_score, "status": us_st, "icon": us_ic},
            },
            "degradation_contributors": [
                {"feature": "State of Health", "value": round(soh, 3), "importance": round(1 - soh, 3)},
                {"feature": "Motor Temperature", "value": round(mot_temp, 3), "importance": round(min(1, mot_temp / 110), 3)},
                {"feature": "Brake Pad Wear", "value": round(brake_wear, 3), "importance": round(brake_wear, 3)},
            ],
            "maintenance_decision": {
                "level": level,
                "message": "🚨 Immediate Maintenance Required" if level == "immediate" else "⚠️ Schedule Maintenance Soon" if level == "soon" else "✅ Vehicle Operating Normally",
                "description": "Fallback analysis — OpenAI call failed. Using rule-based estimate.",
                "color": color,
            },
            "should_alert": failure_prob >= 70 or rul <= 30,
            "alert_severity": alert_severity,
        }
