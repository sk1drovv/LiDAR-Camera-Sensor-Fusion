"""Unit tests for the evaluation framework (kod/analysis.py).

The tests cover the pure functions that decide whether a detection counts as
correct: IoU computation, distance binning and the aggregation of TP/FP/FN
into precision, recall and F1.
"""

import pytest

from analysis import FusionAnalyzer, _empty, _iou


# - IoU

@pytest.mark.parametrize("box_a, box_b, expected", [
    ([0, 0, 10, 10], [0, 0, 10, 10], 1.0),      # identical boxes
    ([0, 0, 10, 10], [20, 20, 30, 30], 0.0),    # no overlap at all
    ([0, 0, 10, 10], [10, 10, 20, 20], 0.0),    # touching at one corner only
    ([0, 0, 10, 10], [10, 0, 20, 10], 0.0),     # sharing an edge only
    ([0, 0, 10, 10], [5, 0, 15, 10], 1 / 3),    # half the width overlaps
    ([0, 0, 10, 10], [0, 0, 5, 10], 0.5),       # one box inside the other
])
def test_iou(box_a, box_b, expected):
    assert _iou(box_a, box_b) == pytest.approx(expected)


def test_iou_is_symmetric():
    box_a, box_b = [0, 0, 10, 10], [5, 5, 15, 15]
    assert _iou(box_a, box_b) == pytest.approx(_iou(box_b, box_a))


def test_iou_of_degenerate_box_is_zero():
    """A box with zero area must not cause a division by zero."""
    assert _iou([0, 0, 0, 0], [0, 0, 10, 10]) == 0.0


# - distance bins

@pytest.fixture
def analyzer():
    return FusionAnalyzer()


@pytest.mark.parametrize("distance, expected", [
    (0.0, "0-20m"),
    (10.0, "0-20m"),
    (20.0, "20-40m"),     # lower bound belongs to the upper bin
    (39.9, "20-40m"),
    (40.0, "40-80m"),
    (79.9, "40-80m"),
    (80.0, ">80m"),
    (250.0, ">80m"),
])
def test_dist_bin(analyzer, distance, expected):
    assert analyzer._dist_bin(distance) == expected


def test_bin_labels(analyzer):
    assert analyzer._bin_labels() == ["0-20m", "20-40m", "40-80m", ">80m"]


# - empty result

def test_empty_metrics_are_all_zero():
    result = _empty()
    for key in ("precision", "recall", "f1", "tp", "fp", "fn", "n_gt"):
        assert result[key] == 0


def test_unknown_method_returns_empty(analyzer):
    assert analyzer.evaluate_frame("no_such_method", [], []) == _empty()


# - frame evaluation

def _detection(bbox, confidence=0.9, cls="Car", distance=10.0):
    return {"bbox": bbox, "confidence": confidence,
            "class": cls, "distance_m": distance}


def _ground_truth(bbox, cls="Car", distance=10.0):
    return {"bbox": bbox, "class": cls, "distance_m": distance}


def test_perfect_match(analyzer):
    """One detection exactly on one annotation: precision and recall are 1."""
    result = analyzer.evaluate_frame(
        "camera",
        [_detection([0, 0, 10, 10])],
        [_ground_truth([0, 0, 10, 10])],
    )
    assert result["tp"] == 1
    assert result["fp"] == 0
    assert result["fn"] == 0
    assert result["precision"] == pytest.approx(1.0, abs=1e-6)
    assert result["recall"] == pytest.approx(1.0, abs=1e-6)
    assert result["f1"] == pytest.approx(1.0, abs=1e-6)


def test_detection_below_iou_threshold_is_a_false_positive(analyzer):
    """Overlap of 1/3 is under the 0.5 threshold, so nothing is matched."""
    result = analyzer.evaluate_frame(
        "camera",
        [_detection([5, 0, 15, 10])],
        [_ground_truth([0, 0, 10, 10])],
    )
    assert result["tp"] == 0
    assert result["fp"] == 1
    assert result["fn"] == 1


def test_missed_object_is_a_false_negative(analyzer):
    result = analyzer.evaluate_frame("camera", [], [_ground_truth([0, 0, 10, 10])])
    assert result["fn"] == 1
    assert result["recall"] == pytest.approx(0.0, abs=1e-6)


def test_spurious_detection_is_a_false_positive(analyzer):
    result = analyzer.evaluate_frame("camera", [_detection([0, 0, 10, 10])], [])
    assert result["fp"] == 1
    assert result["precision"] == pytest.approx(0.0, abs=1e-6)


def test_one_annotation_matches_at_most_one_detection(analyzer):
    """Two overlapping detections on one object: one TP, one FP."""
    result = analyzer.evaluate_frame(
        "camera",
        [_detection([0, 0, 10, 10], confidence=0.9),
         _detection([0, 0, 10, 10], confidence=0.6)],
        [_ground_truth([0, 0, 10, 10])],
    )
    assert result["tp"] == 1
    assert result["fp"] == 1


def test_empty_frame_produces_no_counts(analyzer):
    result = analyzer.evaluate_frame("camera", [], [])
    assert result["tp"] == result["fp"] == result["fn"] == 0
    assert result["n_gt"] == 0


# - accumulation

def test_metrics_accumulate_across_frames(analyzer):
    for _ in range(3):
        analyzer.evaluate_frame(
            "camera",
            [_detection([0, 0, 10, 10])],
            [_ground_truth([0, 0, 10, 10])],
        )
    totals = analyzer.get_metrics("camera")
    assert totals["tp"] == 3
    assert totals["n_gt"] == 3


def test_metrics_are_separated_by_class(analyzer):
    analyzer.evaluate_frame(
        "camera",
        [_detection([0, 0, 10, 10], cls="Car")],
        [_ground_truth([0, 0, 10, 10], cls="Car")],
    )
    assert analyzer.get_metrics("camera", cls="Car")["tp"] == 1
    assert analyzer.get_metrics("camera", cls="Pedestrian")["tp"] == 0


def test_metrics_are_separated_by_distance(analyzer):
    analyzer.evaluate_frame(
        "camera",
        [_detection([0, 0, 10, 10], distance=50.0)],
        [_ground_truth([0, 0, 10, 10], distance=50.0)],
    )
    assert analyzer.get_metrics("camera", dist_bin="40-80m")["tp"] == 1
    assert analyzer.get_metrics("camera", dist_bin="0-20m")["tp"] == 0


def test_methods_do_not_share_counters(analyzer):
    analyzer.evaluate_frame(
        "camera",
        [_detection([0, 0, 10, 10])],
        [_ground_truth([0, 0, 10, 10])],
    )
    assert analyzer.get_metrics("camera")["tp"] == 1
    assert analyzer.get_metrics("lidar")["tp"] == 0


def test_unknown_bin_returns_empty(analyzer):
    assert analyzer.get_metrics("camera", dist_bin="999-1000m") == _empty()
