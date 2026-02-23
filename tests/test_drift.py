"""Tests for memshield._internal.drift."""
from __future__ import annotations

from memshield._internal.drift import BaselineProfile, DriftDetector


class TestBaselineProfile:
    """Tests for BaselineProfile."""

    def test_empty_profile(self) -> None:
        """Empty profile has zero values."""
        p = BaselineProfile()
        assert p.total_reads == 0
        assert p.content_length_mean == 0.0
        assert p.content_length_std == 0.0
        assert p.category_frequency("any") == 0.0

    def test_single_read_mean(self) -> None:
        """Mean with one read equals that read's length."""
        p = BaselineProfile()
        p.total_reads = 1
        p.content_length_sum = 100.0
        p.content_length_sq_sum = 10000.0
        assert p.content_length_mean == 100.0

    def test_single_read_std(self) -> None:
        """Std with fewer than 2 reads is 0."""
        p = BaselineProfile()
        p.total_reads = 1
        p.content_length_sum = 100.0
        p.content_length_sq_sum = 10000.0
        assert p.content_length_std == 0.0


class TestDriftDetector:
    """Tests for DriftDetector."""

    def test_not_baselined_initially(self) -> None:
        """Detector is not baselined with zero reads."""
        d = DriftDetector(min_baseline_reads=20)
        assert d.is_baselined is False

    def test_baselined_after_enough_reads(self) -> None:
        """Detector becomes baselined after min_baseline_reads."""
        d = DriftDetector(min_baseline_reads=5)
        for i in range(5):
            d.record_access(f"entry {i}", category="docs")
        assert d.is_baselined is True

    def test_no_drift_before_baseline(self) -> None:
        """No alerts generated before baseline is established."""
        d = DriftDetector(min_baseline_reads=100)
        d.record_access("some content", category="docs")
        alerts = d.check_drift("some content", category="docs")
        assert alerts == []

    def test_no_drift_for_normal_access(self) -> None:
        """Normal access within baseline produces no alerts."""
        d = DriftDetector(min_baseline_reads=5)
        for i in range(10):
            d.record_access("x" * 50, category="docs")
        alerts = d.check_drift("x" * 50, category="docs")
        assert alerts == []

    def test_novel_category_detected(self) -> None:
        """Access to a never-seen category triggers alert."""
        d = DriftDetector(min_baseline_reads=5)
        for i in range(10):
            d.record_access("content", category="docs")
        alerts = d.check_drift("content", category="secrets")
        novel = [a for a in alerts if a.metric == "novel_category"]
        assert len(novel) == 1
        assert "secrets" in novel[0].message

    def test_content_length_anomaly_detected(self) -> None:
        """Unusually long content triggers alert."""
        d = DriftDetector(min_baseline_reads=5, deviation_threshold=2.0)
        # Build baseline with short entries
        for i in range(20):
            d.record_access("x" * 50, category="docs")
        # Access with extremely long content
        alerts = d.check_drift("x" * 10000, category="docs")
        length_alerts = [a for a in alerts if a.metric == "content_length"]
        assert len(length_alerts) == 1

    def test_normal_length_no_alert(self) -> None:
        """Content within normal range produces no length alert."""
        d = DriftDetector(min_baseline_reads=5, deviation_threshold=3.0)
        # Build baseline with varied lengths so std > 0
        for i in range(20):
            d.record_access("x" * (45 + i % 10), category="docs")
        alerts = d.check_drift("x" * 50, category="docs")
        length_alerts = [a for a in alerts if a.metric == "content_length"]
        assert length_alerts == []

    def test_rare_category_detected(self) -> None:
        """Access to a very rare category triggers alert."""
        d = DriftDetector(min_baseline_reads=5)
        # Build baseline with mostly "docs", one "admin"
        for i in range(99):
            d.record_access("content", category="docs")
        d.record_access("content", category="admin")
        alerts = d.check_drift("content", category="admin")
        rare = [a for a in alerts if a.metric == "rare_category"]
        assert len(rare) == 1

    def test_record_access_updates_baseline(self) -> None:
        """Recording access increments counters."""
        d = DriftDetector(min_baseline_reads=5)
        d.record_access("hello", category="docs")
        assert d.baseline.total_reads == 1
        assert d.baseline.category_counts["docs"] == 1
