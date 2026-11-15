"""Checks the methods recover known answers from synthetic data.  Run:  python -m pytest -q"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis"))
from quality import capability, gage_rr_crossed, repeatability  # noqa: E402
from reliability import compare, fit, median_ranks, weibull_life  # noqa: E402


def _gage_data(sd_part, sd_oper, sd_rep, p=30, o=3, r=4, seed=0):
    rng = np.random.default_rng(seed)
    pe, oe = rng.normal(0, sd_part, p), rng.normal(0, sd_oper, o)
    part, oper, y = [], [], []
    for i in range(p):
        for j in range(o):
            for _ in range(r):
                part.append(i); oper.append(j); y.append(10 + pe[i] + oe[j] + rng.normal(0, sd_rep))
    return part, oper, y


def test_gage_rr_recovers_repeatability():
    sds = []
    for s in range(20):
        g = gage_rr_crossed(*_gage_data(1.0, 0.0, 0.1, seed=s), tol=6)
        sds.append(g["sd_repeatability"])
    assert abs(np.mean(sds) - 0.1) < 0.005


def test_gage_rr_detects_operator_bias():
    g = gage_rr_crossed(*_gage_data(1.0, 0.5, 0.05, o=6, seed=1), tol=6)
    assert g["sd_reproducibility"] > 0.2
    assert g["pct_tolerance_grr"] > 20


def test_repeatability_single_operator():
    part, _, y = _gage_data(1.0, 0.0, 0.2, o=1, r=5, seed=2)
    assert abs(repeatability(part, y, tol=6)["sd_repeatability"] - 0.2) < 0.02


def test_capability_formulas():
    x = np.random.default_rng(3).normal(10.0, 0.1, 20000)
    c = capability(x, 9.4, 10.6)
    assert abs(c["cp"] - 2.0) < 0.03 and abs(c["cpk"] - 2.0) < 0.03
    c = capability(x, 9.7, 10.6)      # off centre: Cpk set by the near (lower) limit
    assert abs(c["cpk"] - 1.0) < 0.02
    assert c["cp_ci"][0] < c["cp"] < c["cp_ci"][1]


def test_weibull_fit_with_censoring():
    rng = np.random.default_rng(4)
    t = 500 * rng.weibull(2.5, 4000)
    failed = t < 600                   # stop the test at 600 cycles
    t = np.minimum(t, 600)
    f = fit(t, failed)
    assert abs(f["beta"] - 2.5) < 0.12 and abs(f["eta"] - 500) < 10


def test_weibull_life_bounds_contain_estimate():
    rng = np.random.default_rng(5)
    t = 300 * rng.weibull(2.0, 12)
    w = weibull_life(t, np.ones(12, bool))
    assert w["b10"] < w["median"]                       # B10 can never exceed the median
    assert w["b10_ci"][0] < w["b10"] < w["b10_ci"][1]
    assert w["median_ci"][0] < w["median"] < w["median_ci"][1]


def test_compare_finds_real_difference():
    rng = np.random.default_rng(6)
    a, b = 100 * rng.weibull(2, 15), 500 * rng.weibull(2, 15)
    one = np.ones(15, bool)
    r = compare(a, one, b, one)
    assert r["p_same_distribution"] < 1e-3
    assert 3.5 < r["life_ratio"] < 7


def test_median_ranks_no_censoring_is_benard():
    t = np.arange(1, 11, dtype=float)
    pts = median_ranks(t, np.ones(10, bool))
    assert np.allclose(pts[:, 1], (np.arange(1, 11) - 0.3) / 10.4)
