"""
Measurement system analysis and process capability.

gage_rr_crossed   Two way crossed ANOVA Gage R&R (AIAG method), parts x operators x repeats
repeatability     Single operator repeatability (parts x repeats)
capability        Cp / Cpk with 95% confidence intervals (Bissell's approximation for Cpk)
"""
import numpy as np
from scipy import stats

SPECS = {                      # nominal, LSL, USL (mm), from the hinge drawing
    "thickness_mm": (0.65, 0.50, 0.80),
    "length_mm": (15.0, 14.0, 16.0),
    "width_mm": (30.0, 29.5, 30.5),
}


def _tolerance(col):
    _, lsl, usl = SPECS[col]
    return usl - lsl


def gage_rr_crossed(part, oper, y, tol=None):
    """AIAG crossed Gage R&R. Returns variance components and the usual ratios."""
    part, oper, y = np.asarray(part), np.asarray(oper), np.asarray(y, float)
    P, O = np.unique(part), np.unique(oper)
    p, o = len(P), len(O)
    r = len(y) // (p * o)
    cell = np.array([[y[(part == a) & (oper == b)].mean() for b in O] for a in P])
    mp, mo, m = cell.mean(1), cell.mean(0), y.mean()
    ss_p = o * r * np.sum((mp - m) ** 2)
    ss_o = p * r * np.sum((mo - m) ** 2)
    ss_po = r * np.sum((cell - mp[:, None] - mo[None, :] + m) ** 2)
    ss_e = sum(np.sum((y[(part == a) & (oper == b)] - cell[i, j]) ** 2)
               for i, a in enumerate(P) for j, b in enumerate(O))
    df_p, df_o, df_po, df_e = p - 1, o - 1, (p - 1) * (o - 1), p * o * (r - 1)
    ms_p, ms_o, ms_po, ms_e = ss_p / df_p, ss_o / df_o, ss_po / df_po, ss_e / df_e
    p_inter = 1 - stats.f.cdf(ms_po / ms_e, df_po, df_e)
    if p_inter > 0.05:            # AIAG: pool the interaction when it isn't significant
        ms_e = (ss_po + ss_e) / (df_po + df_e)
        var_po = 0.0
    else:
        var_po = max((ms_po - ms_e) / r, 0.0)
    var_rep = ms_e
    var_o = max(((ss_o / df_o) - (ms_po if p_inter <= 0.05 else ms_e)) / (p * r), 0.0)
    var_p = max(((ss_p / df_p) - (ms_po if p_inter <= 0.05 else ms_e)) / (o * r), 0.0)
    return _summary(var_p, var_rep, var_o + var_po, tol, dict(p_interaction=float(p_inter)))


def repeatability(part, y, tol=None):
    """Single operator study: repeatability only (no reproducibility can be estimated)."""
    part, y = np.asarray(part), np.asarray(y, float)
    P = np.unique(part)
    r = len(y) // len(P)
    means = np.array([y[part == a].mean() for a in P])
    ms_e = sum(np.sum((y[part == a] - means[i]) ** 2) for i, a in enumerate(P)) / (len(P) * (r - 1))
    ms_p = r * np.var(means, ddof=1)
    var_p = max((ms_p - ms_e) / r, 0.0)
    return _summary(var_p, ms_e, 0.0, tol, dict(note="single operator: reproducibility not measured"))


def _summary(var_p, var_rep, var_repro, tol, extra):
    var_grr = var_rep + var_repro
    var_tot = var_grr + var_p
    sd = lambda v: float(np.sqrt(v))
    out = dict(sd_repeatability=sd(var_rep), sd_reproducibility=sd(var_repro), sd_grr=sd(var_grr),
               sd_part=sd(var_p), pct_contribution_grr=100 * var_grr / var_tot,
               pct_study_var_grr=100 * sd(var_grr) / sd(var_tot),
               ndc=int(np.floor(1.41 * sd(var_p) / sd(var_grr))) if var_grr > 0 else None, **extra)
    if tol:
        out["pct_tolerance_grr"] = 100 * 6 * sd(var_grr) / tol
    return out


def verdict(pct):
    """AIAG rule of thumb for %Study Var or %Tolerance."""
    return "acceptable" if pct < 10 else ("marginal" if pct <= 30 else "not acceptable")


def capability(x, lsl, usl, alpha=0.05):
    """Cp and Cpk (overall standard deviation) with confidence intervals."""
    x = np.asarray(x, float)
    n, mu, s = len(x), x.mean(), x.std(ddof=1)
    cp = (usl - lsl) / (6 * s)
    cpk = min(usl - mu, mu - lsl) / (3 * s)
    lo, hi = stats.chi2.ppf([alpha / 2, 1 - alpha / 2], n - 1)
    cp_ci = (cp * np.sqrt(lo / (n - 1)), cp * np.sqrt(hi / (n - 1)))
    z = stats.norm.ppf(1 - alpha / 2)
    se = np.sqrt(1 / (9 * n) + cpk ** 2 / (2 * (n - 1)))           # Bissell (1990)
    out_of_spec = stats.norm.cdf(lsl, mu, s) + stats.norm.sf(usl, mu, s)
    return dict(n=n, mean=float(mu), sd=float(s), cp=float(cp), cp_ci=[float(v) for v in cp_ci],
                cpk=float(cpk), cpk_ci=[float(cpk - z * se), float(cpk + z * se)],
                ppm_out_of_spec=float(1e6 * out_of_spec))
