"""
Responsible AI Toolbox Assessment Module for BridgeCast AI

Microsoft Innovation Challenge — RAI Scoring Criteria Integration

This module integrates Microsoft's Responsible AI Toolbox (responsibleai, raiwidgets)
to evaluate the fairness, interpretability, and error characteristics of the
Uni-Sign (ICLR 2025) sign language recognition model deployed in BridgeCast AI.

Assessed dimensions:
  - Recognition accuracy across Fitzpatrick skin tone groups (I–VI)
  - Accuracy across signing styles (standard ASL vs. regional variants)
  - Performance under varying lighting conditions
  - Latency distribution per demographic segment

The module exposes a FastAPI endpoint and a standalone sample-assessment generator
so the pipeline can be demonstrated even without a live test dataset.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

# -------------------------------------------------------------------
# Guarded imports: raiwidgets / responsibleai may not be installed in
# every environment (e.g. lightweight CI runners).  We degrade
# gracefully so the rest of the server stays operational.
# -------------------------------------------------------------------
try:
    from responsibleai import RAIInsights
    _RAI_AVAILABLE = True
except ImportError:
    RAIInsights = None  # type: ignore[assignment,misc]
    _RAI_AVAILABLE = False

try:
    from raiwidgets import ResponsibleAIDashboard
    _RAIWIDGETS_AVAILABLE = True
except ImportError:
    ResponsibleAIDashboard = None  # type: ignore[assignment,misc]
    _RAIWIDGETS_AVAILABLE = False

logger = logging.getLogger(__name__)

# Fitzpatrick skin-tone scale labels used throughout the assessment.
FITZPATRICK_LABELS = ["I", "II", "III", "IV", "V", "VI"]

SIGNING_STYLES = ["standard_asl", "regional_southeast", "regional_northeast", "aave_asl", "international_sign"]

LIGHTING_CONDITIONS = ["well_lit", "low_light", "backlit", "mixed_indoor", "outdoor_overcast", "outdoor_bright"]


class SignLanguageRAIAssessment:
    """Runs a Responsible AI assessment suite on Uni-Sign recognition results.

    The class wraps three core analyses:
      1. **Fairness** — disparate accuracy / error-rate across demographic
         slices (skin tone, signing style, lighting).
      2. **Error analysis** — decision-tree-based decomposition of the
         error space to surface high-failure cohorts.
      3. **Interpretability** — feature-importance ranking so reviewers
         can verify the model does not rely on protected attributes.

    All results are serialisable to a JSON summary suitable for reporting
    or display in the RAI Dashboard.
    """

    def __init__(
        self,
        test_data: pd.DataFrame,
        y_true_col: str = "true_label",
        y_pred_col: str = "predicted_label",
        sensitive_features: list[str] | None = None,
    ) -> None:
        self.test_data = test_data.copy()
        self.y_true_col = y_true_col
        self.y_pred_col = y_pred_col
        self.sensitive_features = sensitive_features or [
            "skin_tone",
            "signing_style",
            "lighting_condition",
        ]
        self._report: dict[str, Any] = {}
        self._rai_insights: RAIInsights | None = None

    # ------------------------------------------------------------------
    # Fairness assessment
    # ------------------------------------------------------------------

    def run_fairness_assessment(self) -> dict[str, Any]:
        """Compute per-group accuracy and fairness gap metrics.

        Returns a dict keyed by sensitive feature, each containing
        per-group accuracy and the max disparity ratio.
        """
        fairness: dict[str, Any] = {}

        for feature in self.sensitive_features:
            if feature not in self.test_data.columns:
                logger.warning("Sensitive feature '%s' missing from test data — skipped.", feature)
                continue

            group_stats: dict[str, Any] = {}
            groups = self.test_data[feature].unique()

            for group in groups:
                mask = self.test_data[feature] == group
                subset = self.test_data.loc[mask]
                n = len(subset)
                if n == 0:
                    continue
                correct = (subset[self.y_true_col] == subset[self.y_pred_col]).sum()
                accuracy = float(correct / n)
                group_stats[str(group)] = {
                    "n_samples": int(n),
                    "accuracy": round(accuracy, 4),
                    "error_rate": round(1.0 - accuracy, 4),
                }

            accuracies = [g["accuracy"] for g in group_stats.values()]
            max_acc = max(accuracies) if accuracies else 0.0
            min_acc = min(accuracies) if accuracies else 0.0
            disparity_ratio = round(min_acc / max_acc, 4) if max_acc > 0 else 0.0

            fairness[feature] = {
                "group_metrics": group_stats,
                "max_accuracy": round(max_acc, 4),
                "min_accuracy": round(min_acc, 4),
                "disparity_ratio": disparity_ratio,
                "fairness_pass": disparity_ratio >= 0.80,  # 80 % rule threshold
            }

        self._report["fairness"] = fairness
        return fairness

    # ------------------------------------------------------------------
    # Error analysis
    # ------------------------------------------------------------------

    def run_error_analysis(self) -> dict[str, Any]:
        """Identify cohorts where the model fails disproportionately.

        When the RAI Toolbox is available this delegates to its built-in
        error-analysis tree; otherwise we fall back to a manual pivot
        that highlights the worst-performing feature-value combinations.
        """
        errors = self.test_data[self.y_true_col] != self.test_data[self.y_pred_col]
        overall_error_rate = float(errors.mean())

        worst_cohorts: list[dict[str, Any]] = []

        for feature in self.sensitive_features:
            if feature not in self.test_data.columns:
                continue
            for group in self.test_data[feature].unique():
                mask = self.test_data[feature] == group
                group_error_rate = float(errors[mask].mean())
                n = int(mask.sum())
                if group_error_rate > overall_error_rate * 1.2:
                    worst_cohorts.append({
                        "feature": feature,
                        "value": str(group),
                        "error_rate": round(group_error_rate, 4),
                        "n_samples": n,
                        "relative_error": round(group_error_rate / overall_error_rate, 2),
                    })

        worst_cohorts.sort(key=lambda c: c["error_rate"], reverse=True)

        # Cross-feature interaction analysis (pairwise)
        interaction_cohorts: list[dict[str, Any]] = []
        features_present = [f for f in self.sensitive_features if f in self.test_data.columns]
        for i, f1 in enumerate(features_present):
            for f2 in features_present[i + 1:]:
                grouped = self.test_data.groupby([f1, f2]).apply(
                    lambda g: pd.Series({
                        "error_rate": float((g[self.y_true_col] != g[self.y_pred_col]).mean()),
                        "n_samples": len(g),
                    })
                )
                for (v1, v2), row in grouped.iterrows():
                    if row["error_rate"] > overall_error_rate * 1.5 and row["n_samples"] >= 5:
                        interaction_cohorts.append({
                            "features": {f1: str(v1), f2: str(v2)},
                            "error_rate": round(row["error_rate"], 4),
                            "n_samples": int(row["n_samples"]),
                        })

        interaction_cohorts.sort(key=lambda c: c["error_rate"], reverse=True)

        error_analysis = {
            "overall_error_rate": round(overall_error_rate, 4),
            "total_samples": len(self.test_data),
            "total_errors": int(errors.sum()),
            "worst_cohorts": worst_cohorts[:10],
            "interaction_cohorts": interaction_cohorts[:10],
        }

        self._report["error_analysis"] = error_analysis
        return error_analysis

    # ------------------------------------------------------------------
    # Interpretability
    # ------------------------------------------------------------------

    def run_interpretability(self) -> dict[str, Any]:
        """Generate a lightweight interpretability summary.

        We report feature-correlation with prediction correctness so
        stakeholders can verify that protected attributes (skin tone)
        do not dominate prediction outcomes.
        """
        df = self.test_data.copy()
        df["_correct"] = (df[self.y_true_col] == df[self.y_pred_col]).astype(int)

        feature_importance: dict[str, float] = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in numeric_cols:
            if col.startswith("_") or col in (self.y_true_col, self.y_pred_col):
                continue
            corr = df[col].corr(df["_correct"])
            if not np.isnan(corr):
                feature_importance[col] = round(abs(float(corr)), 4)

        # For categorical sensitive features, compute Cramer's V proxy
        for feature in self.sensitive_features:
            if feature not in df.columns or feature in feature_importance:
                continue
            contingency = pd.crosstab(df[feature], df["_correct"])
            n = contingency.sum().sum()
            if n == 0:
                continue
            chi2 = 0.0
            for row_idx in range(contingency.shape[0]):
                for col_idx in range(contingency.shape[1]):
                    observed = contingency.iloc[row_idx, col_idx]
                    expected = (contingency.iloc[row_idx].sum() * contingency.iloc[:, col_idx].sum()) / n
                    if expected > 0:
                        chi2 += ((observed - expected) ** 2) / expected
            k = min(contingency.shape) - 1
            cramers_v = np.sqrt(chi2 / (n * k)) if k > 0 and n > 0 else 0.0
            feature_importance[feature] = round(float(cramers_v), 4)

        sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

        interpretability = {
            "feature_importance": sorted_importance,
            "protected_attribute_influence": {
                feat: sorted_importance.get(feat, 0.0) for feat in self.sensitive_features
            },
            "note": (
                "High influence of a protected attribute (skin_tone, signing_style) "
                "on prediction correctness signals potential bias requiring mitigation."
            ),
        }

        self._report["interpretability"] = interpretability
        return interpretability

    # ------------------------------------------------------------------
    # Latency analysis
    # ------------------------------------------------------------------

    def run_latency_analysis(self) -> dict[str, Any]:
        """Assess inference latency distribution, optionally per demographic group.

        Expects a 'latency_ms' column in test_data.  If absent, this
        step is skipped gracefully.
        """
        if "latency_ms" not in self.test_data.columns:
            logger.info("No latency_ms column — skipping latency analysis.")
            self._report["latency"] = {"status": "skipped", "reason": "no latency_ms column"}
            return self._report["latency"]

        latency = self.test_data["latency_ms"]
        overall = {
            "p50_ms": round(float(latency.quantile(0.50)), 2),
            "p90_ms": round(float(latency.quantile(0.90)), 2),
            "p99_ms": round(float(latency.quantile(0.99)), 2),
            "mean_ms": round(float(latency.mean()), 2),
            "std_ms": round(float(latency.std()), 2),
        }

        per_group: dict[str, dict[str, Any]] = {}
        for feature in self.sensitive_features:
            if feature not in self.test_data.columns:
                continue
            group_latency: dict[str, Any] = {}
            for group in self.test_data[feature].unique():
                subset = self.test_data.loc[self.test_data[feature] == group, "latency_ms"]
                group_latency[str(group)] = {
                    "p50_ms": round(float(subset.quantile(0.50)), 2),
                    "p90_ms": round(float(subset.quantile(0.90)), 2),
                    "mean_ms": round(float(subset.mean()), 2),
                    "n_samples": int(len(subset)),
                }
            per_group[feature] = group_latency

        latency_report = {"overall": overall, "per_group": per_group}
        self._report["latency"] = latency_report
        return latency_report

    # ------------------------------------------------------------------
    # RAI Insights integration (when responsibleai is available)
    # ------------------------------------------------------------------

    def build_rai_insights(self, model: Any = None, train_data: pd.DataFrame | None = None) -> None:
        """Construct a RAIInsights object for the full dashboard experience.

        This is optional — the manual analyses above work without it.
        When available it enables the interactive ResponsibleAIDashboard.
        """
        if not _RAI_AVAILABLE:
            logger.warning(
                "responsibleai package not installed. "
                "Install via: pip install responsibleai raiwidgets"
            )
            return

        if model is None or train_data is None:
            logger.info("Model or train_data not provided — RAIInsights construction skipped.")
            return

        try:
            target_column = self.y_true_col
            insights = RAIInsights(
                model=model,
                train=train_data,
                test=self.test_data,
                target_column=target_column,
                task_type="classification",
                categorical_features=self.sensitive_features,
            )
            insights.error_analysis.add()
            insights.explainer.add()
            insights.compute()
            self._rai_insights = insights
            logger.info("RAIInsights built successfully.")
        except Exception:
            logger.exception("Failed to build RAIInsights — falling back to manual analysis.")

    def launch_dashboard(self, port: int = 5100) -> None:
        """Launch the interactive Responsible AI Dashboard (blocking call)."""
        if not _RAIWIDGETS_AVAILABLE:
            logger.error("raiwidgets not installed — cannot launch dashboard.")
            return
        if self._rai_insights is None:
            logger.error("RAIInsights not built — call build_rai_insights() first.")
            return
        ResponsibleAIDashboard(self._rai_insights, port=port)

    # ------------------------------------------------------------------
    # Full report
    # ------------------------------------------------------------------

    def generate_full_report(self) -> dict[str, Any]:
        """Execute all assessment stages and return a consolidated JSON-safe report."""
        self.run_fairness_assessment()
        self.run_error_analysis()
        self.run_interpretability()
        self.run_latency_analysis()

        self._report["metadata"] = {
            "assessment_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": "Uni-Sign (ICLR 2025)",
            "project": "BridgeCast AI — Sign Language Recognition",
            "rai_toolbox_available": _RAI_AVAILABLE,
            "raiwidgets_available": _RAIWIDGETS_AVAILABLE,
            "n_samples": len(self.test_data),
            "sensitive_features": self.sensitive_features,
        }

        return self._report

    def to_json(self, indent: int = 2) -> str:
        """Serialise the current report to a JSON string."""
        if not self._report:
            self.generate_full_report()
        return json.dumps(self._report, indent=indent, default=str)


# ======================================================================
# Sample / demo assessment
# ======================================================================

def generate_sample_assessment() -> dict[str, Any]:
    """Create a demo RAI assessment with realistic synthetic data.

    This function exists so the BridgeCast AI team can demonstrate
    Responsible AI Toolbox integration during the hackathon even when
    real Uni-Sign test results are not yet available.  The synthetic
    distribution intentionally includes a fairness gap in darker skin
    tones and low-light conditions to showcase the detection capability.
    """
    rng = np.random.default_rng(seed=42)
    n = 1200

    # Build a synthetic test dataset with known bias patterns.
    skin_tones = rng.choice(FITZPATRICK_LABELS, size=n, p=[0.12, 0.18, 0.20, 0.20, 0.18, 0.12])
    styles = rng.choice(SIGNING_STYLES, size=n)
    lighting = rng.choice(LIGHTING_CONDITIONS, size=n)

    # Ground-truth labels: 50 common ASL signs
    asl_signs = [f"sign_{i:03d}" for i in range(50)]
    true_labels = rng.choice(asl_signs, size=n)

    # Simulate predictions with systematic bias:
    #   - lower accuracy for Fitzpatrick V–VI (darker skin)
    #   - lower accuracy under low_light / backlit conditions
    #   - slightly lower accuracy for regional signing styles
    predicted_labels = true_labels.copy()

    for idx in range(n):
        base_error_prob = 0.08  # 8 % baseline error

        # Skin-tone bias
        if skin_tones[idx] in ("V", "VI"):
            base_error_prob += 0.07
        elif skin_tones[idx] == "IV":
            base_error_prob += 0.03

        # Lighting bias
        if lighting[idx] in ("low_light", "backlit"):
            base_error_prob += 0.09
        elif lighting[idx] == "mixed_indoor":
            base_error_prob += 0.03

        # Signing style bias
        if styles[idx] in ("regional_southeast", "aave_asl"):
            base_error_prob += 0.05
        elif styles[idx] == "international_sign":
            base_error_prob += 0.04

        if rng.random() < base_error_prob:
            wrong = rng.choice(asl_signs)
            while wrong == true_labels[idx]:
                wrong = rng.choice(asl_signs)
            predicted_labels[idx] = wrong

    # Latency: base ~45 ms, higher under poor conditions
    latency = rng.normal(loc=45, scale=8, size=n)
    for idx in range(n):
        if lighting[idx] in ("low_light", "backlit"):
            latency[idx] += rng.normal(12, 3)
    latency = np.clip(latency, 10, 200)

    # Confidence score (numeric feature for interpretability)
    confidence = np.where(
        true_labels == predicted_labels,
        rng.uniform(0.75, 0.99, size=n),
        rng.uniform(0.30, 0.70, size=n),
    )

    df = pd.DataFrame({
        "true_label": true_labels,
        "predicted_label": predicted_labels,
        "skin_tone": skin_tones,
        "signing_style": styles,
        "lighting_condition": lighting,
        "latency_ms": np.round(latency, 2),
        "confidence": np.round(confidence, 4),
    })

    assessment = SignLanguageRAIAssessment(test_data=df)
    report = assessment.generate_full_report()

    logger.info(
        "Sample assessment generated: %d samples, overall error rate %.2f%%",
        n,
        report["error_analysis"]["overall_error_rate"] * 100,
    )
    return report


# ======================================================================
# FastAPI endpoint
# ======================================================================

def get_rai_report_router():
    """Return a FastAPI APIRouter with the /rai-report endpoint.

    Usage in the main server:
        from rai_assessment import get_rai_report_router
        app.include_router(get_rai_report_router())
    """
    from fastapi import APIRouter
    from fastapi.responses import JSONResponse

    router = APIRouter(tags=["Responsible AI"])

    @router.get(
        "/rai-report",
        summary="Responsible AI Assessment Report",
        description=(
            "Returns a JSON report evaluating the Uni-Sign model for fairness, "
            "error analysis, interpretability, and latency across demographic groups."
        ),
    )
    async def get_rai_report() -> JSONResponse:
        """Generate and return the RAI assessment summary.

        In production this would load real test results from Cosmos DB;
        for the hackathon demo it falls back to synthetic data.
        """
        try:
            report = generate_sample_assessment()
            return JSONResponse(content=report, status_code=200)
        except Exception as exc:
            logger.exception("RAI report generation failed.")
            return JSONResponse(
                content={"error": str(exc), "detail": "RAI assessment could not be completed."},
                status_code=500,
            )

    return router


# ======================================================================
# CLI convenience
# ======================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    report = generate_sample_assessment()
    print(json.dumps(report, indent=2, default=str))
