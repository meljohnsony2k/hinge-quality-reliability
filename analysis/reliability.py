"""
Life data analysis with right censoring (units still working when the test stopped).

fit           Maximum likelihood fit of Weibull, lognormal or normal life distributions,
              with failures and censored units handled properly
compare       Likelihood ratio test: do two designs have the same life distribution?
weibull_life  Median, mean, B10 life with likelihood based confidence bounds
"""
import math

import numpy as np
from scipy import optimize, stats


def _nll_weibull(params, t, failed):
    beta, eta = np.exp(params)
    z = (t / eta) ** beta
    return -(np.sum(np.log(beta / eta) + (beta - 1) * np.log(t[failed] / eta) - z[failed]) - np.sum(z[~failed]))


def _nll_lognormal(params, t, failed):
    mu, sig = params[0], np.exp(params[1])
    lt = np.log(t)
    return -(np.sum(stats.norm.logpdf(lt[failed], mu, sig) - lt[failed]) + np.sum(stats.norm.logsf(lt[~failed], mu, sig)))


def _nll_normal(params, t, failed):
    mu, sig = params[0], np.exp(params[1])
    return -(np.sum(stats.norm.logpdf(t[failed], mu, sig)) + np.sum(stats.norm.logsf(t[~failed], mu, sig)))


NLL = {"weibull": _nll_weibull, "lognormal": _nll_lognormal, "normal": _nll_normal}


def fit(t, failed, dist="weibull"):
    t, failed = np.asarray(t, float), np.asarray(failed, bool)
    if dist == "weibull":
        x0 = [0.0, np.log(t.mean())]
    elif dist == "lognormal":
        x0 = [np.log(t).mean(), np.log(np.log(t).std() + 1e-3)]
    else:
        x0 = [t.mean(), np.log(t.std() + 1e-3)]
    res = optimize.minimize(NLL[dist], x0, args=(t, failed), method="Nelder-Mead",
                            options=dict(xatol=1e-10, fatol=1e-12, maxiter=20000))
    k = 2
    out = dict(dist=dist, nll=float(res.fun), aic=float(2 * k + 2 * res.fun), x=res.x)
    if dist == "weibull":
        out.update(beta=float(np.exp(res.x[0])), eta=float(np.exp(res.x[1])))
    else:
        out.update(mu=float(res.x[0]), sigma=float(np.exp(res.x[1])))
    return out


def weibull_quantile(beta, eta, p):
    return eta * (-math.log(1 - p)) ** (1 / beta)


def weibull_life(t, failed, level=0.95):
    """Point estimates plus profile likelihood confidence bounds for B10 and median."""
    t, failed = np.asarray(t, float), np.asarray(failed, bool)
    f = fit(t, failed, "weibull")
    beta, eta = f["beta"], f["eta"]
    out = dict(beta=beta, eta=eta, mean=eta * math.gamma(1 + 1 / beta),
               median=weibull_quantile(beta, eta, 0.5), b10=weibull_quantile(beta, eta, 0.10),
               n=int(len(t)), failures=int(failed.sum()))
    crit = stats.chi2.ppf(level, 1) / 2
    for name, p in (("b10", 0.10), ("median", 0.5)):
        def prof(q):        # best likelihood with quantile fixed at q: eta = q / (-ln(1-p))^(1/beta)
            g = lambda lb: _nll_weibull([lb, math.log(q) - math.log(-math.log(1 - p)) / math.exp(lb)], t, failed)
            r = optimize.minimize_scalar(g, bounds=(-3, 4), method="bounded")
            return r.fun - f["nll"] - crit
        q0 = out[name]
        lo = optimize.brentq(prof, q0 * 1e-3, q0 * 0.999999) if prof(q0 * 1e-3) > 0 else float("nan")
        hi = optimize.brentq(prof, q0 * 1.000001, q0 * 50) if prof(q0 * 50) > 0 else float("nan")
        out[f"{name}_ci"] = [lo, hi]
    return out


def compare(t1, f1, t2, f2):
    """Likelihood ratio tests between two designs (Weibull).
    H0a: same distribution (shape and scale).  H0b: same shape only (common beta)."""
    t1, f1, t2, f2 = (np.asarray(a) for a in (t1, f1, t2, f2))
    a, b = fit(t1, f1), fit(t2, f2)
    pooled = fit(np.r_[t1, t2], np.r_[f1, f2].astype(bool))
    lr_same = 2 * (pooled["nll"] - a["nll"] - b["nll"])

    def nll_common(p):
        lb, le1, le2 = p
        return _nll_weibull([lb, le1], t1.astype(float), f1.astype(bool)) + \
            _nll_weibull([lb, le2], t2.astype(float), f2.astype(bool))
    r = optimize.minimize(nll_common, [0.5, a["x"][1], b["x"][1]], method="Nelder-Mead",
                          options=dict(xatol=1e-10, fatol=1e-12, maxiter=20000))
    lr_shape = 2 * (r.fun - a["nll"] - b["nll"])
    return dict(lr_same_distribution=float(lr_same), p_same_distribution=float(stats.chi2.sf(lr_same, 2)),
                lr_same_shape=float(lr_shape), p_same_shape=float(stats.chi2.sf(lr_shape, 1)),
                common_beta=float(np.exp(r.x[0])), life_ratio=float(np.exp(r.x[2] - r.x[1])))


def median_ranks(t, failed):
    """Johnson's adjusted ranks with Benard's approximation, for probability plots with censoring."""
    order = np.argsort(t)
    t, failed = np.asarray(t, float)[order], np.asarray(failed, bool)[order]
    n = len(t)
    prev, pts = 0.0, []
    for i, (ti, fi) in enumerate(zip(t, failed)):
        if fi:
            inc = (n + 1 - prev) / (1 + (n - i))
            prev += inc
            pts.append((ti, (prev - 0.3) / (n + 0.4)))
    return np.array(pts)
