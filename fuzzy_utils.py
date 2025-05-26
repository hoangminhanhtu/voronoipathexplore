"""Fuzzy logic helpers for battery-aware replanning."""

from typing import Tuple

__all__ = [
    "fuzzify_battery",
    "need_replan",
    "membership_battery",
    "membership_distance",
    "fuzzy_replan_decision",
    "fuzzy_neighbor_radius",
]


def fuzzify_battery(batt: float) -> Tuple[float, float, float]:
    """Return (low, medium, high) membership degrees for battery level."""
    low = max(min((50.0 - batt) / 50.0, 1.0), 0.0)
    high = max(min((batt - 50.0) / 50.0, 1.0), 0.0)
    med = max(1.0 - low - high, 0.0)
    return low, med, high


def need_replan(batt_deg: Tuple[float, float, float]) -> bool:
    """Return True if the fuzzy battery degrees indicate replanning."""
    low, med, high = batt_deg
    score = low * 1.0 + med * 0.5
    return score > 0.5


def membership_battery(batt: float) -> Tuple[float, float, float]:
    """Triangular memberships for battery percentage."""
    low = max(min((50.0 - batt) / 50.0, 1.0), 0.0)
    high = max(min((batt - 50.0) / 50.0, 1.0), 0.0)
    medium = max(min(batt / 50.0, (100.0 - batt) / 50.0), 0.0)
    return low, medium, high


def membership_distance(dist: float, max_dist: float) -> Tuple[float, float, float]:
    """Triangular memberships for remaining distance."""
    short = max(min((max_dist - dist) / max_dist, 1.0), 0.0)
    if dist <= max_dist / 2.0:
        med = dist / (max_dist / 2.0)
    else:
        med = max((max_dist - dist) / (max_dist / 2.0), 0.0)
    long_ = max(min(dist / max_dist, 1.0), 0.0)
    return short, med, long_


def fuzzy_replan_decision(batt: float, dist: float, max_dist: float) -> bool:
    """Return True if fuzzy logic advises replanning."""
    b_low, b_med, b_high = membership_battery(batt)
    d_short, d_med, d_long = membership_distance(dist, max_dist)
    r_high = min(b_low, d_long)
    r_med = max(min(b_med, d_long), min(b_low, d_med))
    r_low = min(b_high, d_short)
    total = r_low + r_med + r_high + 1e-6
    score = (r_low * 0.25 + r_med * 0.5 + r_high * 0.75) / total
    return score > 0.5


def fuzzy_neighbor_radius(batt: float) -> float:
    """Return suggested neighbor radius based on battery level."""
    low = max(min((30.0 - batt) / 30.0, 1.0), 0.0)
    med = max(min((batt - 20.0) / 30.0, (80.0 - batt) / 30.0), 0.0)
    high = max(min((batt - 50.0) / 50.0, 1.0), 0.0)
    weight_sum = low + med + high
    if weight_sum == 0:
        return 1.0
    return (low * 0.5 + med * 1.0 + high * 1.5) / weight_sum
