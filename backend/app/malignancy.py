"""Cancerous vs non-cancerous estimate derived from the 4-class tumor probabilities.

The classifier only predicts tumor type. Malignancy is inferred from typical biology:
  glioma              -> malignant group (most gliomas are malignant; low-grade gliomas exist)
  meningioma          -> benign group (about 80% are WHO grade 1; atypical/anaplastic ones exist)
  pituitary adenoma   -> benign group (almost always non-cancerous)
  no tumor            -> no tumor
This is a type-based prior, not a histological grade.
"""

import math

GROUPS = {
    "malignant": ("glioma",),
    "benign": ("meningioma", "pituitary"),
    "no_tumor": ("no_tumor",),
}

ASSESSMENT = {
    "malignant": "likely_cancerous",
    "benign": "likely_non_cancerous",
    "no_tumor": "no_tumor_detected",
}


def confidence_level(p: float) -> str:
    if p >= 0.85:
        return "high"
    if p >= 0.6:
        return "moderate"
    return "low"


def assess(probabilities: dict[str, float]) -> dict:
    groups = {g: round(sum(probabilities.get(c, 0.0) for c in classes), 6) for g, classes in GROUPS.items()}
    top = max(groups, key=groups.get)
    probs = [p for p in probabilities.values() if p > 0]
    # Normalised entropy: 0 = certain, 1 = uniform over all classes.
    entropy = -sum(p * math.log(p) for p in probs) / math.log(len(probabilities)) if len(probabilities) > 1 else 0.0
    return {
        "malignant": groups["malignant"],
        "benign": groups["benign"],
        "no_tumor": groups["no_tumor"],
        "assessment": ASSESSMENT[top],
        "confidence": confidence_level(groups[top]),
        "uncertainty": round(entropy, 4),
        "basis": "tumor-type prior",
    }
