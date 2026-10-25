"""Core functions of the analysis of prereg/preregistro.md (sections 5, 6 and 8).

Pure functions on arrays, with no file access. The pipeline that reads the runs and writes the
tables is analysis/analise_congelada.py.

Conventions
- m is the logit margin z_water - z_land of a scored pixel. Prediction: water if m > 0.
- y is the label, 1 (water) or 0 (land). Error: prediction different from the label.
- Confidence is |m|. MSP = sigmoid(|m|). The score of the prediction is s = 1 - MSP = sigmoid(-|m|).
- Histograms are ordered by increasing |m| (increasing confidence).
"""
import math
from fractions import Fraction

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit

ALPHA = Fraction(1, 10)
DELTA = 0.02
BOOT_B = 10_000
BOOT_SEED = 0
BIN_WIDTH = 0.005
BIN_MAX = 40.0
HIST_TOL = 1e-4
N_ECE_BINS = 15
MIN_PER_CLASS = 50


# ----------------------------------------------------------------------------------------------
# Conformal threshold and temperature (section 5)
# ----------------------------------------------------------------------------------------------

def true_class_score(m, y):
    """s = 1 - p(true class), with p(water) = sigmoid(m)."""
    m = np.asarray(m, dtype=np.float64)
    return np.where(np.asarray(y) == 1, expit(-m), expit(m))


def conformal_threshold(m_val, y_val, alpha=ALPHA):
    """q_hat: the ceil((n + 1)(1 - alpha))/n quantile (numpy method 'higher') of s on the validation pixels.

    Returns (q_hat, tau) with tau = 1 - q_hat.
    """
    s = true_class_score(m_val, y_val)
    n = s.size
    k = math.ceil((n + 1) * (1 - Fraction(alpha)))
    nivel = min(1.0, k / n)
    q_hat = float(np.quantile(s, nivel, method="higher"))
    return q_hat, 1.0 - q_hat


def _nll_signed(z, temperature, bloco=4_000_000):
    total = 0.0
    for i in range(0, z.size, bloco):
        total += float(np.logaddexp(0.0, -z[i:i + bloco].astype(np.float64) / temperature).sum())
    return total / z.size


def fit_temperature(m_val, y_val):
    """One temperature per run, fitted on the validation pixels by negative log-likelihood."""
    z = np.where(np.asarray(y_val) == 1, m_val, -np.asarray(m_val, dtype=np.float32)).astype(np.float32)
    res = minimize_scalar(lambda lt: _nll_signed(z, math.exp(lt)), bounds=(-3.0, 3.0), method="bounded",
                          options={"xatol": 1e-6})
    return float(math.exp(res.x))


# ----------------------------------------------------------------------------------------------
# Histograms of |m|
# ----------------------------------------------------------------------------------------------

class Bins:
    """Bins of |m| of width `width` from 0 to `vmax`, plus one bin above vmax."""

    def __init__(self, width=BIN_WIDTH, vmax=BIN_MAX):
        self.width = float(width)
        self.vmax = float(vmax)
        self.n = int(round(self.vmax / self.width)) + 1

    def index(self, a):
        return np.minimum((np.asarray(a, dtype=np.float64) / self.width).astype(np.int64), self.n - 1)

    def __repr__(self):
        return f"Bins(width={self.width}, vmax={self.vmax}, n={self.n})"


def refine_bins(bins, max_abs_m):
    """Rule of section 6 when the histogram AUROC_e differs from the exact value by more than 1e-4:
    first raise the upper limit to the maximum |m| over all runs, rounded up to a multiple of 5;
    if that is not enough, halve the width. This function returns the next step."""
    vmax_novo = 5.0 * math.ceil(max_abs_m / 5.0)
    if vmax_novo > bins.vmax:
        return Bins(bins.width, vmax_novo)
    return Bins(bins.width / 2.0, bins.vmax)


def unit_histograms(unit_id, a, err, n_units, bins, mascara=None, bloco=20_000_000):
    """Per-unit histograms of |m| for errors and for correct pixels. Shape (n_units, bins.n) each.

    mascara: optional boolean array; only the pixels where it is True are counted."""
    he = np.zeros(n_units * bins.n, dtype=np.int64)
    hc = np.zeros(n_units * bins.n, dtype=np.int64)
    for i in range(0, a.size, bloco):
        sl = slice(i, i + bloco)
        chave = unit_id[sl].astype(np.int64) * bins.n + bins.index(a[sl])
        e = err[sl]
        if mascara is not None:
            m = mascara[sl]
            ce, cc = chave[e & m], chave[~e & m]
        else:
            ce, cc = chave[e], chave[~e]
        he += np.bincount(ce, minlength=n_units * bins.n)
        hc += np.bincount(cc, minlength=n_units * bins.n)
    return he.reshape(n_units, bins.n).astype(np.int32), hc.reshape(n_units, bins.n).astype(np.int32)


# ----------------------------------------------------------------------------------------------
# AUROC_e, AUGRC, E-AURC
# ----------------------------------------------------------------------------------------------

def auroc_from_hist(he, hc):
    """AUROC_e from histograms ordered by increasing confidence. Errors are the positive class and
    the score is 1 - MSP, so an error ranks above a correct pixel when its |m| is smaller. Ties count 1/2.
    Accepts 1-D or 2-D (rows are independent histograms). Returns NaN when a class is empty."""
    he = np.asarray(he, dtype=np.float64)
    hc = np.asarray(hc, dtype=np.float64)
    abaixo = np.cumsum(he, axis=-1) - he
    num = (hc * (abaixo + 0.5 * he)).sum(-1)
    den = hc.sum(-1) * he.sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def auroc_exact(a_err, a_cor, bloco=2_000_000):
    """Exact AUROC_e from the |m| of the errors and of the correct pixels (ties count 1/2).

    The two arrays are compared in a common floating type, so float32 margins stay float32."""
    tipo = np.result_type(np.asarray(a_err).dtype, np.asarray(a_cor).dtype)
    if tipo.kind != "f":
        tipo = np.dtype(np.float64)
    a_err = np.sort(np.asarray(a_err, dtype=tipo))
    a_cor = np.asarray(a_cor, dtype=tipo)
    ne, nc = a_err.size, a_cor.size
    if ne == 0 or nc == 0:
        return float("nan")
    total = 0.0
    for i in range(0, nc, bloco):
        c = a_cor[i:i + bloco]
        menor = np.searchsorted(a_err, c, side="left")    # errors with |m| strictly below
        ate = np.searchsorted(a_err, c, side="right")      # errors with |m| below or equal
        total += float(menor.sum()) + 0.5 * float((ate - menor).sum())
    return total / (float(ne) * float(nc))


def augrc_from_hist(he, hc):
    """Area under the generalized risk-coverage curve (Traub et al., 2024).

    Pixels are accepted in decreasing |m|. The generalized risk at a coverage is the number of accepted
    errors divided by the number of all scored pixels. Inside a bin the order is random, so the expected
    number of accepted errors after k of the n_b pixels of a bin is E_prev + k e_b / n_b. The area is the
    sum over the pixels of the generalized risk after accepting that pixel, divided by N."""
    he = np.asarray(he, dtype=np.float64)[..., ::-1]
    hc = np.asarray(hc, dtype=np.float64)[..., ::-1]
    nb = he + hc
    e_prev = np.cumsum(he, axis=-1) - he
    soma = (nb * e_prev + he * (nb + 1.0) / 2.0).sum(-1)
    n = nb.sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, soma / (n * n), np.nan)


def augrc_exact(a, err):
    """AUGRC from pixels, with random order inside groups of equal |m| (same expectation as the histogram)."""
    a = np.asarray(a, dtype=np.float32)
    err = np.asarray(err, dtype=bool)
    n = a.size
    if n == 0:
        return float("nan")
    ordem = np.argsort(-a, kind="stable")
    a = a[ordem]
    e = err[ordem].astype(np.float64)
    fim = np.flatnonzero(np.diff(a) != 0) + 1
    inicio = np.concatenate([[0], fim])
    limite = np.concatenate([fim, [n]])
    cum_e = np.concatenate([[0.0], np.cumsum(e)])
    soma = 0.0
    for i0, i1 in zip(inicio, limite):
        nb = i1 - i0
        eb = cum_e[i1] - cum_e[i0]
        soma += nb * cum_e[i0] + eb * (nb + 1.0) / 2.0
    return soma / (float(n) * float(n))


def _aurc_de_grupos(nb, eb, n_total, e_total):
    """Excess AURC from groups of tied pixels listed in decreasing confidence.

    nb: pixels per group, eb: errors per group. Inside a group the order is random, so after k of its
    n pixels the expected number of errors is E_prev + k e/n. The selective risk after accepting the
    j-th pixel overall is (expected errors)/j, and AURC is its mean over the N pixels. The sums over k
    have closed forms with the digamma function. The optimal AURC puts all errors last.
    nb and eb may be 1-D; empty groups are allowed."""
    from scipy.special import digamma
    nb = np.asarray(nb, dtype=np.float64)
    eb = np.asarray(eb, dtype=np.float64)
    c_prev = np.cumsum(nb) - nb
    e_prev = np.cumsum(eb) - eb
    ativo = nb > 0
    nb, eb, c_prev, e_prev = nb[ativo], eb[ativo], c_prev[ativo], e_prev[ativo]
    soma_inv = digamma(c_prev + nb + 1.0) - digamma(c_prev + 1.0)       # sum_{k=1..n} 1/(C+k)
    # sum_k (E + k e/n)/(C+k) = E S + (e/n) (n - C S)
    soma = e_prev * soma_inv + (eb / nb) * (nb - c_prev * soma_inv)
    aurc = float(soma.sum()) / n_total
    n_ok = n_total - e_total
    h_dif = digamma(n_total + 1.0) - digamma(n_ok + 1.0)                # H_N - H_{N-E}
    otimo = (e_total - n_ok * h_dif) / n_total
    return aurc - otimo


def eaurc_from_hist(he, hc):
    """E-AURC from histograms ordered by increasing confidence (1-D), bins treated as tied groups."""
    he = np.asarray(he, dtype=np.float64)[::-1]
    hc = np.asarray(hc, dtype=np.float64)[::-1]
    n = float(he.sum() + hc.sum())
    e = float(he.sum())
    if n == 0 or e == 0:
        return float("nan")
    return _aurc_de_grupos(he + hc, he, n, e)


def eaurc_exact(a, err):
    """E-AURC from pixels, with groups of equal |m| treated as tied."""
    a = np.asarray(a)
    err = np.asarray(err, dtype=bool)
    n = a.size
    if n == 0 or not err.any():
        return float("nan")
    ordem = np.argsort(-a, kind="stable")
    a = a[ordem]
    e = err[ordem].astype(np.float64)
    fim = np.flatnonzero(a[1:] != a[:-1]) + 1
    inicio = np.concatenate([[0], fim])
    limite = np.concatenate([fim, [n]])
    cum = np.concatenate([[0.0], np.cumsum(e)])
    return _aurc_de_grupos((limite - inicio).astype(np.float64), cum[limite] - cum[inicio], float(n), float(e.sum()))


# ----------------------------------------------------------------------------------------------
# Counts per unit and metrics derived from them
# ----------------------------------------------------------------------------------------------

CONTAGENS = ("n", "erro", "erro_nao_sinalizado", "sinalizado", "coberto", "tp", "fp", "fn", "tn")


def unit_counts(unit_id, m, y, tau, n_units):
    """Counts per unit, columns in the order of CONTAGENS.

    A pixel is flagged when MSP < tau (empty conformal set). For tau > 0.5 the true class is in the set only
    for a correct prediction with MSP >= tau, so covered = correct and not flagged.
    Water is the positive class of tp, fp, fn, tn."""
    unit_id = np.asarray(unit_id, dtype=np.int64)
    m = np.asarray(m)
    y = np.asarray(y).astype(bool)
    a = np.abs(m).astype(np.float64)
    msp = expit(a)
    pred = m > 0
    erro = pred != y
    sinal = msp < tau

    def cont(mask):
        return np.bincount(unit_id[mask], minlength=n_units).astype(np.float64)

    out = np.zeros((n_units, len(CONTAGENS)), dtype=np.float64)
    out[:, 0] = np.bincount(unit_id, minlength=n_units)
    out[:, 1] = cont(erro)
    out[:, 2] = cont(erro & ~sinal)
    out[:, 3] = cont(sinal)
    out[:, 4] = cont(~erro & ~sinal)
    out[:, 5] = cont(pred & y)
    out[:, 6] = cont(pred & ~y)
    out[:, 7] = cont(~pred & y)
    out[:, 8] = cont(~pred & ~y)
    return out


def count_metrics(c):
    """Metrics from summed counts. c[..., k] in the order of CONTAGENS. Returns a dict of arrays."""
    c = np.asarray(c, dtype=np.float64)
    n, err, unf, sin, cob, tp, fp, fn, tn = (c[..., i] for i in range(len(CONTAGENS)))
    with np.errstate(invalid="ignore", divide="ignore"):
        iou_agua = tp / (tp + fp + fn)
        iou_terra = tn / (tn + fp + fn)
        return {
            "taxa_erro": err / n,
            "miou": (iou_agua + iou_terra) / 2.0,
            "cobertura": cob / n,
            "conjunto_vazio": sin / n,
            "falha_silenciosa": unf / n,
            "erro_nao_sinalizado": unf / err,
        }


# ----------------------------------------------------------------------------------------------
# Calibration (section 5)
# ----------------------------------------------------------------------------------------------

def calibration_sorted(m_s, y_s, temperature=1.0, n_bins=N_ECE_BINS, bloco=4_000_000):
    """Signed ECE with equal-mass bins, NLL and Brier for the water probability sigma(m / T).

    m_s, y_s: margins and labels already in increasing order of |m| (stable, so ties keep the pixel order).
    The n_bins groups are made on the rank, with the remainder spread over the first bins as
    numpy.array_split does. Signed ECE = sum_b (n_b / N) (mean confidence_b - accuracy_b); positive means
    overconfident. Brier is (p_water - y)^2 averaged over pixels (binary form)."""
    n = m_s.size
    y_s = np.asarray(y_s).astype(bool)
    cortes = np.cumsum([len(x) for x in np.array_split(np.arange(n), n_bins)])
    soma_conf = np.zeros(n_bins)
    soma_acc = np.zeros(n_bins)
    nll = 0.0
    brier = 0.0
    T = float(temperature)
    for i in range(0, n, bloco):
        mi = m_s[i:i + bloco].astype(np.float64)
        yi = y_s[i:i + bloco]
        conf = expit(np.abs(mi) / T)
        certo = ((mi > 0) == yi).astype(np.float64)
        b = np.searchsorted(cortes, np.arange(i, i + mi.size), side="right")
        soma_conf += np.bincount(b, weights=conf, minlength=n_bins)
        soma_acc += np.bincount(b, weights=certo, minlength=n_bins)
        z = np.where(yi, mi, -mi) / T
        nll += float(np.logaddexp(0.0, -z).sum())
        brier += float(((expit(mi / T) - yi) ** 2).sum())
    return {"ece_com_sinal": float(((soma_conf - soma_acc) / n).sum()), "nll": nll / n, "brier": brier / n}


def calibration(m, y, temperature=1.0, n_bins=N_ECE_BINS):
    """calibration_sorted on pixels in any order (sorted here by |m|, stable)."""
    m = np.asarray(m)
    ordem = np.argsort(np.abs(m), kind="stable")
    return calibration_sorted(m[ordem], np.asarray(y)[ordem], temperature, n_bins)


# ----------------------------------------------------------------------------------------------
# Effective margin of an ensemble (mean probability)
# ----------------------------------------------------------------------------------------------

def mean_probability_margin(m_stack):
    """Signed logit of the mean water probability of K margins, m_stack of shape (K, N).

    logit(mean p) = log(mean p) - log(mean (1 - p)), each term through logsumexp of log-sigmoids,
    so it stays finite and accurate for large |m|. Returns float64."""
    from scipy.special import log_expit, logsumexp
    m_stack = np.asarray(m_stack, dtype=np.float64)
    k = m_stack.shape[0]
    lp = logsumexp(log_expit(m_stack), axis=0) - math.log(k)
    lq = logsumexp(log_expit(-m_stack), axis=0) - math.log(k)
    return lp - lq
