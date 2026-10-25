"""Resampling machinery of the analysis of prereg/preregistro.md (section 6).

A replicate is described by two things: how many times each unit (chip, window or block) enters it,
and which of the seeds of each regime enter it. The draws come from numpy.random.default_rng(0). The seed draws of R1 and R3 come first, then the unit
draws in the order of the strata, then the seed draw of R2, so nothing in the primary comparison depends on
R2 and the draws do not depend on the names of the codes.
"""
import math

import numpy as np

import nucleo as N


def make_rng(seed=N.BOOT_SEED):
    return np.random.default_rng(seed)


def seed_draw(rng, B, n_seeds):
    """Indices of the seeds that enter each replicate for one regime. Shape (B, n_seeds).

    Each regime has its own draw, independent of the others."""
    return rng.integers(0, n_seeds, size=(B, n_seeds))


def stratified_counts(rng, strata, n_units, B):
    """Multiplicity of each unit in B replicates when units are resampled with replacement within strata.

    strata: list of integer arrays with the unit indices of each stratum, in a fixed order.
    Returns an int16 array (B, n_units). Strata are not resampled."""
    cnt = np.zeros((B, n_units), dtype=np.int32)
    for idx in strata:
        idx = np.asarray(idx, dtype=np.int64)
        k = idx.size
        if k == 0:
            continue
        sorteio = idx[rng.integers(0, k, size=(B, k))]
        base = (np.arange(B, dtype=np.int64) * n_units)[:, None]
        cnt += np.bincount((sorteio + base).ravel(), minlength=B * n_units).reshape(B, n_units).astype(np.int32)
    return cnt.astype(np.int16)


def two_stage_counts(rng, strata, n_units, B):
    """Two-stage resampling: the strata (events) are drawn with replacement, then, for each time a stratum
    is drawn, its units are drawn with replacement within it. Returns an int16 array (B, n_units)."""
    cnt = np.zeros((B, n_units), dtype=np.int32)
    n_strata = len(strata)
    for b in range(B):
        sorteados = rng.integers(0, n_strata, size=n_strata)
        for e in sorteados:
            idx = np.asarray(strata[e], dtype=np.int64)
            if idx.size == 0:
                continue
            escolhidos = idx[rng.integers(0, idx.size, size=idx.size)]
            cnt[b] += np.bincount(escolhidos, minlength=n_units).astype(np.int32)
    return cnt.astype(np.int16)


def hist_metrics(cnt, he, hc, chunk=200):
    """AUROC_e and AUGRC of the pooled histogram of every replicate.

    cnt: (B, U) multiplicities; he, hc: (U, bins) histograms of errors and correct pixels in increasing
    confidence. Columns of bins that are empty in every unit are dropped (they change nothing).
    Returns two arrays of length B; NaN where a replicate has no error or no correct pixel."""
    B = cnt.shape[0]
    usados = (he.sum(0) + hc.sum(0)) > 0
    he_c = np.ascontiguousarray(he[:, usados], dtype=np.float64)
    hc_c = np.ascontiguousarray(hc[:, usados], dtype=np.float64)
    auroc = np.empty(B)
    augrc = np.empty(B)
    for i in range(0, B, chunk):
        c = cnt[i:i + chunk].astype(np.float64)
        e = c @ he_c
        k = c @ hc_c
        auroc[i:i + chunk] = N.auroc_from_hist(e, k)
        augrc[i:i + chunk] = N.augrc_from_hist(e, k)
    return auroc, augrc


def count_metrics_replicates(cnt, counts):
    """Counts-based metrics of every replicate. counts: (U, len(CONTAGENS)). Returns a dict of (B,) arrays."""
    soma = cnt.astype(np.float64) @ counts
    return N.count_metrics(soma)


def unit_auroc(he, hc):
    """AUROC_e of every unit from its histograms (NaN where a unit has no error or no correct pixel)."""
    return N.auroc_from_hist(he, hc)


def seed_mean(valores, sorteio):
    """Mean over the resampled seeds. valores: (K, B) one row per seed of one regime; sorteio: (B, K)
    indices drawn for this regime. Returns (B,)."""
    v = np.asarray(valores).T
    return np.take_along_axis(v, sorteio, axis=1).mean(axis=1)


def quantile_interval(x, nivel):
    """Percentile interval of the replicates at the given central level (0.90 or 0.95), numpy linear rule,
    ignoring NaN replicates."""
    x = np.asarray(x)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    lo = (1.0 - nivel) / 2.0
    return float(np.quantile(x, lo)), float(np.quantile(x, 1.0 - lo))


def p_value(x):
    """p = 2 min(share <= 0, share >= 0), shares as (count + 1)/(B + 1), capped at 1."""
    x = np.asarray(x)
    x = x[~np.isnan(x)]
    B = x.size
    if B == 0:
        return float("nan")
    menor = (np.sum(x <= 0) + 1.0) / (B + 1.0)
    maior = (np.sum(x >= 0) + 1.0) / (B + 1.0)
    return float(min(1.0, 2.0 * min(menor, maior)))


def holm(pvalores):
    """Holm-adjusted p-values (family-wise), in the order given."""
    p = np.asarray(pvalores, dtype=np.float64)
    if np.isnan(p).any():
        raise ValueError("Holm correction needs defined p-values")
    m = p.size
    ordem = np.argsort(p, kind="stable")
    ajust = np.empty(m)
    maximo = 0.0
    for posicao, i in enumerate(ordem):
        maximo = max(maximo, (m - posicao) * p[i])
        ajust[i] = min(1.0, maximo)
    return ajust


def mde(x, z_alpha=1.6448536269514722, z_power=0.8416212335729143):
    """Minimum detectable effect for 80% power with the 90% interval of the decision rule:
    (z_{0.95} + z_{0.80}) times the standard deviation of the replicates."""
    x = np.asarray(x)
    x = x[~np.isnan(x)]
    if x.size < 2:
        return float("nan")
    return float((z_alpha + z_power) * np.std(x, ddof=1))


def ceil_two_thirds(n):
    """ceil(2 n / 3) with integer arithmetic."""
    return -((-2 * int(n)) // 3)
