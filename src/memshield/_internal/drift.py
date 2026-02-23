"""Statistical baseline profiling and drift detection for memory access patterns."""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

from memshield._types import DriftAlert


@dataclass
class BaselineProfile:
    """Statistical profile of normal memory access patterns."""
    category_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    total_reads: int = 0
    content_length_sum: float = 0.0
    content_length_sq_sum: float = 0.0

    @property
    def content_length_mean(self) -> float:
        """Mean content length of accessed entries."""
        if self.total_reads == 0:
            return 0.0
        return self.content_length_sum / self.total_reads

    @property
    def content_length_std(self) -> float:
        """Standard deviation of content length."""
        if self.total_reads < 2:
            return 0.0
        variance = (
            self.content_length_sq_sum / self.total_reads
            - (self.content_length_sum / self.total_reads) ** 2
        )
        return math.sqrt(max(0.0, variance))

    def category_frequency(self, category: str) -> float:
        """Frequency of a category in the baseline."""
        if self.total_reads == 0:
            return 0.0
        return self.category_counts[category] / self.total_reads


class DriftDetector:
    """Detects deviation from baseline memory access patterns."""

    def __init__(self, *, deviation_threshold: float = 3.0, min_baseline_reads: int = 20) -> None:
        self._threshold = deviation_threshold
        self._min_reads = min_baseline_reads
        self._baseline = BaselineProfile()
        self._is_baselined = False

    @property
    def is_baselined(self) -> bool:
        """Whether enough data has been collected to detect drift."""
        return self._baseline.total_reads >= self._min_reads

    @property
    def baseline(self) -> BaselineProfile:
        """The current baseline profile."""
        return self._baseline

    def record_access(self, content: str, category: str = "default") -> None:
        """Record a memory access to build the baseline profile."""
        self._baseline.total_reads += 1
        self._baseline.category_counts[category] += 1
        length = float(len(content))
        self._baseline.content_length_sum += length
        self._baseline.content_length_sq_sum += length * length

    def check_drift(self, content: str, category: str = "default") -> list[DriftAlert]:
        """Check if a memory access deviates from the baseline.

        Returns a list of alerts. Empty list means no drift detected.
        """
        if not self.is_baselined:
            return []

        alerts: list[DriftAlert] = []

        # Check 1: novel category
        if category not in self._baseline.category_counts:
            alerts.append(DriftAlert(
                metric="novel_category",
                baseline_value=0.0,
                current_value=1.0,
                deviation=float("inf"),
                message=f"Access to previously unseen category: {category!r}",
            ))

        # Check 2: content length anomaly
        mean = self._baseline.content_length_mean
        std = self._baseline.content_length_std
        length = float(len(content))
        if std > 0:
            z_score = abs(length - mean) / std
            if z_score > self._threshold:
                alerts.append(DriftAlert(
                    metric="content_length",
                    baseline_value=mean,
                    current_value=length,
                    deviation=z_score,
                    message=(
                        f"Content length {len(content)} deviates from baseline "
                        f"(mean={mean:.0f}, std={std:.0f}, z={z_score:.1f})"
                    ),
                ))
        elif abs(length - mean) > 0:
            # Zero variance baseline — any deviation is anomalous
            alerts.append(DriftAlert(
                metric="content_length",
                baseline_value=mean,
                current_value=length,
                deviation=float("inf"),
                message=(
                    f"Content length {len(content)} deviates from baseline "
                    f"(all baseline entries had length {mean:.0f})"
                ),
            ))

        # Check 3: category frequency anomaly (rare category accessed)
        freq = self._baseline.category_frequency(category)
        if 0 < freq < 0.05:
            alerts.append(DriftAlert(
                metric="rare_category",
                baseline_value=freq,
                current_value=freq,
                deviation=1.0 / freq if freq > 0 else float("inf"),
                message=(
                    f"Access to rare category {category!r} "
                    f"(baseline frequency: {freq:.1%})"
                ),
            ))

        return alerts
