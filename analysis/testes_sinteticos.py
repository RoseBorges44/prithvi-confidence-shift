#!/usr/bin/env python3
"""Tests of the analysis on synthetic margins (prereg/preregistro.md, section 10).

Run: python analysis/testes_sinteticos.py [work directory]
Each test prints OK or raises. The end-to-end tests write synthetic data in the work directory
(a temporary directory if none is given) and run the frozen pipeline on it.
"""
import json
import math
import os
import sys
import tempfile
import time

import numpy as np
from scipy.special import expit
from scipy.stats import rankdata

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import analise_congelada as A  # noqa: E402
import categorias as C  # noqa: E402
import gerador_sintetico as G  # noqa: E402
import nucleo as N  # noqa: E402
import reamostragem as R  # noqa: E402


# ----------------------------------------------------------------------------------------------
# Independent reference implementations (deliberately simple)
# ----------------------------------------------------------------------------------------------

def ref_auroc(a, err):
    """AUROC_e by average ranks: P(|m| of an error < |m| of a correct pixel), ties one half."""
    a = np.asarray(a, dtype=np.float64)
    err = np.asarray(err, dtype=bool)
    ne, nc = err.sum(), (~err).sum()
    if ne == 0 or nc == 0:
        return float("nan")
    r = rankdata(a)                       # ascending, ties averaged
    u_err_maior = r[err].sum() - ne * (ne + 1) / 2.0
    return 1.0 - u_err_maior / (ne * nc)


def ref_augrc(a, err):
    """AUGRC with random order inside ties, by its expectation over tie groups."""
    a = np.asarray(a, dtype=np.float64)
    err = np.asarray(err, dtype=bool)
    n = a.size
    valores, inv = np.unique(-a, return_inverse=True)       # groups in decreasing |m|
    tam = np.bincount(inv)
    erros = np.bincount(inv, weights=err.astype(float))
    total, antes = 0.0, 0.0
    for t, e in zip(tam, erros):
        for k in range(1, int(t) + 1):
            total += antes + k * e / t
        antes += e
    return total / (n * n)


def ref_conformal(m, y, alpha=0.1):
    """tau = 1 - q_hat, with q_hat = numpy quantile at level k/n, k = ceil((n + 1)(1 - alpha)), method 'higher'.
    The virtual index of that level is k (n - 1)/n = k - k/n, so 'higher' takes the element of index k
    (0-based) of the sorted scores when k < n, and the largest score otherwise."""
    from fractions import Fraction
    s = np.where(y == 1, expit(-m.astype(np.float64)), expit(m.astype(np.float64)))
    s = np.sort(s)
    n = s.size
    k = math.ceil((n + 1) * (1 - Fraction(alpha).limit_denominator(1000)))
    return 1.0 - s[min(k, n - 1)]


def ref_miou(pred, y):
    ious = []
    for c in (0, 1):
        inter = np.sum((pred == c) & (y == c))
        uniao = np.sum((pred == c) | (y == c))
        ious.append(inter / uniao)
    return float(np.mean(ious))


# ----------------------------------------------------------------------------------------------
# Unit tests of the core
# ----------------------------------------------------------------------------------------------

def test_auroc_histograma_e_exato():
    rng = np.random.default_rng(1)
    for n, mu in ((2000, 3.0), (300_000, 4.0)):
        y = rng.random(n) < 0.3
        d = rng.normal(mu, 3.0, n)
        m = np.where(y, d, -d).astype(np.float32)
        a, err = np.abs(m), (m > 0) != y
        ex = N.auroc_exact(a[err], a[~err])
        assert abs(ex - ref_auroc(a, err)) < 1e-12
        b = N.Bins()
        he, hc = N.unit_histograms(np.zeros(n, dtype=np.int32), a, err, 1, b)
        assert abs(N.auroc_from_hist(he[0], hc[0]) - ex) < N.HIST_TOL
    # ties: identical values must count one half
    a = np.array([1.0, 1.0, 2.0, 2.0])
    err = np.array([True, False, True, False])
    assert abs(N.auroc_exact(a[err], a[~err]) - ref_auroc(a, err)) < 1e-12


def test_auroc_analitico():
    rng = np.random.default_rng(2)
    mu, sigma = 4.0, 3.0
    n = 4_000_000
    d = rng.normal(mu, sigma, n)
    a, err = np.abs(d).astype(np.float32), d < 0
    est = N.auroc_exact(a[err], a[~err])
    alvo = G.auroc_e_analitico(mu, sigma)
    assert abs(est - alvo) < 0.002, (est, alvo)


def test_augrc_e_eaurc():
    rng = np.random.default_rng(3)
    a = np.round(rng.random(600) * 10) / 10       # many ties
    err = rng.random(600) < 0.4 * (1 - a)
    assert abs(N.augrc_exact(a, err) - ref_augrc(a, err)) < 1e-12
    b = N.Bins(width=0.1, vmax=1.0)
    idx = np.round(a * 10).astype(int)
    he = np.bincount(idx[err], minlength=11)
    hc = np.bincount(idx[~err], minlength=11)
    assert abs(N.augrc_from_hist(he, hc) - ref_augrc(a, err)) < 1e-12
    # E-AURC without ties against the definition
    a = rng.random(500)
    err = rng.random(500) < 0.5 - 0.4 * a
    o = np.argsort(-a)
    cum = np.cumsum(err[o])
    k = np.arange(1, 501)
    aurc = np.mean(cum / k)
    e = err.sum()
    otimo = np.mean(np.clip(k - (500 - e), 0, None) / k)
    assert abs(N.eaurc_exact(a, err) - (aurc - otimo)) < 1e-12
    # a random score gives AUGRC close to half the error rate
    a = rng.random(1_000_000)
    err = rng.random(1_000_000) < 0.1
    assert abs(N.augrc_exact(a, err) - err.mean() / 2) < 5e-4


def test_limiar_conformal_e_cobertura():
    rng = np.random.default_rng(4)
    m = rng.normal(0, 4, 200_001).astype(np.float32)
    y = (m + rng.normal(0, 2, m.size) > 0).astype(np.uint8)
    q, tau = N.conformal_threshold(m, y)
    assert abs(tau - ref_conformal(m, y)) < 1e-15
    s = N.true_class_score(m, y)
    assert np.mean(s <= q) >= 0.9


def test_temperatura_recupera_valor_conhecido():
    rng = np.random.default_rng(5)
    T0 = 1.7
    z = rng.normal(0, 4, 2_000_000)
    y = (rng.random(z.size) < expit(z / T0)).astype(np.uint8)
    T = N.fit_temperature(z.astype(np.float32), y)
    assert abs(T - T0) < 0.02, T


def test_contagens_e_metricas():
    rng = np.random.default_rng(6)
    n = 50_000
    m = rng.normal(0, 4, n).astype(np.float32)
    y = (m + rng.normal(0, 3, n) > 0).astype(np.uint8)
    u = rng.integers(0, 7, n)
    tau = 0.9
    c = N.unit_counts(u, m, y, tau, 7).sum(0)
    msp = expit(np.abs(m).astype(np.float64))
    pred = (m > 0).astype(np.uint8)
    err = pred != y
    cm = N.count_metrics(c)
    assert abs(cm["taxa_erro"] - err.mean()) < 1e-15
    assert abs(cm["erro_nao_sinalizado"] - np.mean(msp[err] >= tau)) < 1e-15
    assert abs(cm["falha_silenciosa"] - np.mean(err & (msp >= tau))) < 1e-15
    assert abs(cm["conjunto_vazio"] - np.mean(msp < tau)) < 1e-15
    assert abs(cm["cobertura"] - np.mean(~err & (msp >= tau))) < 1e-15
    assert abs(cm["miou"] - ref_miou(pred, y)) < 1e-12


def test_calibracao():
    rng = np.random.default_rng(7)
    m = rng.normal(0, 4, 30_001).astype(np.float32)
    y = (m + rng.normal(0, 3, m.size) > 0).astype(np.uint8)
    for T in (1.0, 1.6):
        r = N.calibration(m, y, T)
        a = np.abs(m).astype(np.float64)
        o = np.argsort(a, kind="stable")
        ece = 0.0
        for p in np.array_split(o, 15):
            ece += p.size / m.size * (expit(a[p] / T).mean() - ((m[p] > 0) == (y[p] == 1)).mean())
        z = np.where(y == 1, m, -m).astype(np.float64) / T
        assert abs(r["ece_com_sinal"] - ece) < 1e-12
        assert abs(r["nll"] - np.mean(np.log1p(np.exp(-z)))) < 1e-12
        assert abs(r["brier"] - np.mean((expit(m.astype(np.float64) / T) - y) ** 2)) < 1e-12


def test_margem_do_conjunto():
    rng = np.random.default_rng(8)
    ms = rng.normal(0, 6, (5, 2000))
    p = expit(ms).mean(0)
    assert np.abs(N.mean_probability_margin(ms) - np.log(p / (1 - p))).max() < 1e-9
    assert np.isfinite(N.mean_probability_margin(np.array([[60.0, -70.0], [55.0, -80.0]]))).all()


def test_refino_das_faixas():
    b = N.Bins()
    b1 = N.refine_bins(b, 52.3)
    assert b1.vmax == 55.0 and b1.width == b.width
    b2 = N.refine_bins(b1, 52.3)
    assert b2.vmax == 55.0 and b2.width == b.width / 2
    b3 = N.refine_bins(b, 21.0)
    assert b3.vmax == 40.0 and b3.width == b.width / 2


# ----------------------------------------------------------------------------------------------
# Categories, Holm, p-values, draws
# ----------------------------------------------------------------------------------------------

def test_categorias_h1_h2():
    d = 0.02
    casos = [((0.0, -0.01, 0.015), "H2, preservation"),
             ((-0.03, -0.05, -0.01), "H1, relevant degradation"),
             ((-0.02, -0.04, -0.001), "H1, relevant degradation"),
             ((-0.01, -0.03, -0.001), "Degradation smaller than the margin"),
             ((-0.005, -0.015, -0.001), "H2, preservation"),
             ((0.03, 0.01, 0.05), "Improvement"),
             ((-0.01, -0.04, 0.02), "Inconclusive")]
    for (p, lo, hi), esperado in casos:
        assert C.category_h1_h2(p, lo, hi, d) == esperado, (p, lo, hi)


def test_categorias_h3():
    casos = [((0.0, -0.015, 0.015, -0.01, 3, 2, 5), "Null"),
             ((-0.03, -0.05, -0.01, -0.02, 4, 1, 5), "Distortion, strong"),
             ((-0.03, -0.05, -0.01, -0.02, 3, 2, 5), "Distortion, weak"),
             ((-0.03, -0.05, -0.01, 0.01, 5, 0, 5), "Distortion, weak"),
             ((-0.015, -0.03, -0.001, -0.02, 5, 0, 5), "Distortion, weak"),
             ((0.03, 0.01, 0.05, 0.02, 0, 4, 5), "Rigidity, strong"),
             ((0.03, 0.01, 0.05, 0.02, 0, 3, 5), "Rigidity, weak"),
             ((0.02, 0.001, 0.04, 0.0, 0, 5, 5), "Rigidity, weak"),
             ((0.0, -0.03, 0.03, 0.0, 2, 2, 5), "Inconclusive"),
             ((-0.03, -0.05, -0.01, -0.02, 4, 0, 6), "Distortion, strong"),
             ((-0.03, -0.05, -0.01, -0.02, 2, 0, 4), "Distortion, weak"),
             ((-0.03, -0.05, -0.01, -0.02, 3, 0, 4), "Distortion, strong")]
    for args, esperado in casos:
        assert C.category_h3(*args) == esperado, args
    assert [R.ceil_two_thirds(n) for n in (3, 4, 5, 6, 9)] == [2, 3, 4, 4, 6]


def test_holm_e_p():
    p = [0.01, 0.04, 0.03]
    assert np.allclose(R.holm(p), [0.03, 0.06, 0.06])
    assert np.allclose(R.holm([0.5, 0.01, 0.2]), [0.5, 0.03, 0.4])
    x = np.array([1.0] * 99 + [-1.0])
    assert abs(R.p_value(x) - 2 * 2 / 101) < 1e-15
    assert R.p_value(np.ones(10)) == 2 * 1 / 11
    assert R.p_value(np.array([0.0, 1.0, -1.0])) == 1.0


def test_sorteios():
    rng = R.make_rng()
    estratos = [np.array([0, 1, 2]), np.array([3]), np.array([4, 5, 6, 7, 8])]
    cnt = R.stratified_counts(rng, estratos, 9, 5000)
    for e in estratos:
        assert (cnt[:, e].sum(1) == e.size).all()
    assert (cnt[:, 3] == 1).all()
    assert abs(cnt.mean() - 1.0) < 0.02
    # same seed, same draws
    a = R.stratified_counts(R.make_rng(), estratos, 9, 50)
    b = R.stratified_counts(R.make_rng(), estratos, 9, 50)
    assert (a == b).all()
    cnt2 = R.two_stage_counts(R.make_rng(), estratos, 9, 3000)
    assert abs(cnt2.sum(1).mean() - 9.0) < 0.2
    s = R.seed_draws(R.make_rng(), 1000, 3, 5)
    assert s.shape == (1000, 3, 5) and s.min() == 0 and s.max() == 4
    v = np.arange(5 * 4, dtype=float).reshape(5, 4)          # 5 seeds, 4 replicates
    sort = np.array([[0, 0, 0, 0, 0], [4, 4, 4, 4, 4], [0, 1, 2, 3, 4], [1, 1, 2, 2, 3]])
    esperado = [v[0, 0], v[4, 1], v[:, 2].mean(), np.mean([v[1, 3], v[1, 3], v[2, 3], v[2, 3], v[3, 3]])]
    assert np.allclose(R.seed_mean(v, sort), esperado)


# ----------------------------------------------------------------------------------------------
# End-to-end tests on synthetic data
# ----------------------------------------------------------------------------------------------

def rodar_pipeline(dados, saida, B):
    return A.main(["--repo", dados["repo"], "--margens", dados["margens"], "--chave", dados["chave"],
                   "--saida", saida, "--B", str(B)])


def pixels(dados, cod, sem, conjuntos, campo="margem", unidades=None):
    """Margins and labels of some sets of one run, optionally restricted to some units, read from disk."""
    ms, ys, us = [], [], []
    for conj in conjuntos:
        z = np.load(os.path.join(dados["margens"], f"run_{cod}_s{sem}_{conj}.npz"))
        rot = np.load(os.path.join(dados["margens"], f"labels_{conj}.npz"))
        ini = np.concatenate([[0], np.cumsum(z["tamanhos"])])
        for i, chave in enumerate(z["chaves"]):
            if z["tamanhos"][i] == 0 or (unidades is not None and str(chave) not in unidades):
                continue
            ms.append(z[campo][ini[i]:ini[i + 1]])
            ys.append(rot["rotulo"][ini[i]:ini[i + 1]])
            us.append(np.full(z["tamanhos"][i], str(chave), dtype=object))
    return np.concatenate(ms), np.concatenate(ys), np.concatenate(us)


def unidades_primarias(dados):
    import csv
    ev = {r["map_id"]: r for r in csv.DictReader(open(os.path.join(dados["repo"], "splits", "t2_eventos.csv")))}
    jan = list(csv.DictReader(open(os.path.join(dados["repo"], "splits", "t2_janelas.csv"))))
    prim = {r["window_id"] for r in jan if ev[r["map_id"]]["core"] == "True" and r["kept_primary"] == "True"}
    bol = {c for c, n in dados["unidades"]["bolivia"] if n > 0}
    return prim | bol, jan, ev


def test_ponta_a_ponta_analitico(base):
    """Homogeneous synthetic data: pooled AUROC_e of every regime close to the closed form, and the planted
    effects give the expected categories."""
    raiz = os.path.join(base, "analitico")
    dados = G.gerar(raiz, semente=101, escala=1.5, heterogeneidade=0.0, desloc_semente=0.0)
    res = rodar_pipeline(dados, os.path.join(raiz, "saida"), 400)
    pool = res["cenarios"]["primaria"]["agregadores"]["pool"]
    for reg in ("R1", "R2", "R3"):
        for dom, chave in (("alvo", "alvo"), ("fonte", "fonte")):
            mu, sigma = G.PARAMETROS[reg][dom]
            alvo = G.auroc_e_analitico(mu, sigma)
            est = pool["auroc"][reg][chave]["ponto"]
            assert abs(est - alvo) < 0.006, (reg, dom, est, alvo)
    esperado_shift = (G.auroc_e_analitico(*G.PARAMETROS["R3"]["alvo"])
                      - G.auroc_e_analitico(*G.PARAMETROS["R3"]["fonte"]))
    assert abs(pool["delta_shift"]["R3"]["ponto"] - esperado_shift) < 0.008
    assert pool["delta_shift"]["R3"]["categoria"] == "H1, relevant degradation"
    assert pool["delta_reg"]["R3-R1"]["categoria"] == "Distortion, strong"
    lo, hi = pool["delta_reg"]["R3-R1"]["ic90"]
    assert lo < pool["delta_reg"]["R3-R1"]["ponto"] < hi


def test_ponta_a_ponta_reconstrucao(base):
    """Heterogeneous synthetic data: every primary point estimate recomputed from the files with independent
    code; the bootstrap replicates rebuilt pixel by pixel; the blocks; determinism of the output."""
    raiz = os.path.join(base, "reconstrucao")
    dados = G.gerar(raiz, semente=202, escala=0.5)
    saida = os.path.join(raiz, "saida")
    res = rodar_pipeline(dados, saida, 300)
    prim, jan, ev = unidades_primarias(dados)
    chave = json.load(open(dados["chave"]))
    reg2cod = {v: k for k, v in chave.items()}
    pool = res["cenarios"]["primaria"]["agregadores"]["pool"]
    nao_sin = {}
    for reg, cod in reg2cod.items():
        auc_t, auc_s, aug_t, ns_t, err_t, miou_t = [], [], [], [], [], []
        for sem in G.SEMENTES:
            mv, yv, _ = pixels(dados, cod, sem, ["val"])
            tau = ref_conformal(mv, yv)
            m, y, _ = pixels(dados, cod, sem, ["bolivia", "t2"], unidades=prim)
            a, err = np.abs(m), (m > 0) != (y == 1)
            auc_t.append(ref_auroc(a, err))
            aug_t.append(ref_augrc(a, err) if a.size < 400_000 else N.augrc_exact(a, err))
            msp = expit(a.astype(np.float64))
            ns_t.append(np.mean(msp[err] >= tau))
            err_t.append(err.mean())
            miou_t.append(ref_miou((m > 0).astype(int), y.astype(int)))
            ms, ys, _ = pixels(dados, cod, sem, ["teste"])
            auc_s.append(ref_auroc(np.abs(ms), (ms > 0) != (ys == 1)))
        assert abs(pool["auroc"][reg]["alvo"]["ponto"] - np.mean(auc_t)) < 1e-10
        assert abs(pool["auroc"][reg]["fonte"]["ponto"] - np.mean(auc_s)) < 1e-10
        r = res["regimes"][reg]["alvo"]
        assert abs(r["augrc"]["media"] - np.mean(aug_t)) < 1e-10
        assert abs(r["erro_nao_sinalizado"]["media"] - np.mean(ns_t)) < 1e-12
        assert abs(r["taxa_erro"]["media"] - np.mean(err_t)) < 1e-12
        assert abs(r["miou"]["media"] - np.mean(miou_t)) < 1e-12
        nao_sin[reg] = np.mean(ns_t)
    s2 = res["cenarios"]["primaria"]["s2"]["ponto"]
    assert abs(s2 - 100 * (nao_sin["R3"] - nao_sin["R1"])) < 1e-9

    # bootstrap replicates rebuilt from the pixels
    ctx = A.montar_contexto(dados["repo"], dados["margens"])
    corridas = A.listar_corridas(dados["margens"])
    codigos = A.verificar_corridas(corridas)
    est, ens, dif, _ = A.processar_tudo(ctx, corridas, N.Bins(), True)
    sel = A.selecoes_alvo(ctx["alvo"])
    esp = A.especificacoes(ctx)["primaria"]
    el_s, el_t = A.elegiveis(est, "fonte"), A.elegiveis(est, "alvo")
    B = 12
    out = A.rodar_cenario(ctx, est, ens, codigos, esp, sel, B, el_s, el_t)
    rng = R.make_rng()
    S = R.seed_draws(rng, B, len(codigos), A.N_SEMENTES)
    cnt_s = R.stratified_counts(rng, out["P"]["est_s"], out["P"]["n_s"], B)
    cnt_t = R.stratified_counts(rng, out["P"]["est_t"], out["P"]["n_t"], B)
    assert (S == out["S"]).all()
    ids_t = [ctx["alvo"].ids[i] for i in out["P"]["idx"]]
    ids_s = ctx["fonte"].ids
    cod = reg2cod["R3"]
    for k, sem in enumerate(G.SEMENTES[:2]):
        m, y, u = pixels(dados, cod, sem, ["bolivia", "t2"], unidades=set(ids_t))
        ms, ys, us = pixels(dados, cod, sem, ["teste"])
        pos_t = {c: i for i, c in enumerate(ids_t)}
        pos_s = {c: i for i, c in enumerate(ids_s)}
        ut = np.array([pos_t[c] for c in u])
        us_ = np.array([pos_s[c] for c in us])
        for b in range(3):
            w = cnt_t[b][ut]
            rep_m, rep_y = np.repeat(m, w), np.repeat(y, w)
            ref = ref_auroc(np.abs(rep_m), (rep_m > 0) != (rep_y == 1))
            assert abs(out["Rb"][cod]["tgt_auc"][k][b] - ref) < N.HIST_TOL, (b, ref)
            w = cnt_s[b][us_]
            rep_m, rep_y = np.repeat(ms, w), np.repeat(ys, w)
            ref = ref_auroc(np.abs(rep_m), (rep_m > 0) != (rep_y == 1))
            assert abs(out["Rb"][cod]["src_auc"][k][b] - ref) < N.HIST_TOL
    # seed averaging of the regime value
    j = codigos.index(cod)
    manual = np.array([out["Rb"][cod]["tgt_auc"][S[b, j], b].mean() for b in range(B)])
    assert np.allclose(A.media_replicas(out["Rb"][cod], "tgt_auc", S[:, j, :]), manual)

    # blocks: 2 x 2 by grid rank inside each scene
    esp_b = A.especificacoes(ctx)["blocos_espaciais"]
    P = A.preparar_cenario(ctx, esp_b, sel)
    linhas = {}
    colunas = {}
    for r in jan:
        linhas.setdefault(r["map_id"], set()).add(int(r["row_off"]))
        colunas.setdefault(r["map_id"], set()).add(int(r["col_off"]))
    grupo = {}
    for i in P["idx"]:
        w = ctx["alvo"].ids[i]
        r = next(x for x in jan if x["window_id"] == w) if ctx["alvo"].attr["conjunto"][i] == "t2" else None
        if r is None:
            grupo[w] = w
        else:
            rl = sorted(linhas[r["map_id"]]).index(int(r["row_off"])) // 2
            rc = sorted(colunas[r["map_id"]]).index(int(r["col_off"])) // 2
            grupo[w] = (r["map_id"], rl, rc)
    assert P["Ag"].shape[0] == len(set(grupo.values()))
    assert (P["Ag"].sum(0) == 1).all()

    # determinism: a second run gives the same output
    res2 = rodar_pipeline(dados, os.path.join(raiz, "saida2"), 300)
    j1 = json.load(open(os.path.join(saida, "resultados.json")))
    j2 = json.load(open(os.path.join(raiz, "saida2", "resultados.json")))
    assert j1 == j2


def test_controles_de_entrada(base):
    """Wrong inputs stop the analysis."""
    raiz = os.path.join(base, "controles")
    dados = G.gerar(raiz, semente=303, escala=0.2)
    chave = json.load(open(dados["chave"]))
    codigos = sorted(chave)
    ruim = os.path.join(raiz, "chave_ruim.json")
    json.dump({k: "R1" for k in chave}, open(ruim, "w"))
    try:
        A.ler_chave(ruim, codigos)
        raise AssertionError("duplicated regime accepted")
    except ValueError:
        pass
    json.dump({codigos[0]: "R1", codigos[1]: "R3"}, open(ruim, "w"))
    try:
        A.ler_chave(ruim, codigos)
        raise AssertionError("key file with a missing code accepted")
    except ValueError:
        pass
    assert A.ler_chave(dados["chave"], codigos) == {"R1": "k7", "R2": "q2", "R3": "x9"}
    # a run file with a missing seed
    os.remove(os.path.join(dados["margens"], "run_k7_s5_t2.npz"))
    try:
        A.verificar_corridas(A.listar_corridas(dados["margens"]))
        raise AssertionError("missing set accepted")
    except ValueError:
        pass
    # windows of the cloud 0.50 maps absent
    raiz2 = os.path.join(base, "controles2")
    dados2 = G.gerar(raiz2, semente=304, escala=0.2, nuvem050=False)
    try:
        A.montar_contexto(dados2["repo"], dados2["margens"])
        raise AssertionError("missing cloud 0.50 windows accepted")
    except ValueError:
        pass
    ctx = A.montar_contexto(dados2["repo"], dados2["margens"], usar_nuvem050=False)
    assert "nuvem_050" not in A.especificacoes(ctx)


TESTES_UNIDADE = [test_auroc_histograma_e_exato, test_auroc_analitico, test_augrc_e_eaurc,
                  test_limiar_conformal_e_cobertura, test_temperatura_recupera_valor_conhecido,
                  test_contagens_e_metricas, test_calibracao, test_margem_do_conjunto, test_refino_das_faixas,
                  test_categorias_h1_h2, test_categorias_h3, test_holm_e_p, test_sorteios]
TESTES_PONTA = [test_controles_de_entrada, test_ponta_a_ponta_analitico, test_ponta_a_ponta_reconstrucao]


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="testes_analise_")
    os.makedirs(base, exist_ok=True)
    falhas = 0
    for t in TESTES_UNIDADE + TESTES_PONTA:
        t0 = time.time()
        try:
            t(base) if t in TESTES_PONTA else t()
            print(f"OK    {t.__name__} ({time.time() - t0:.0f}s)", flush=True)
        except Exception as e:  # noqa: BLE001
            falhas += 1
            print(f"FALHA {t.__name__}: {type(e).__name__}: {e}", flush=True)
            import traceback
            traceback.print_exc()
    print("todos os testes passaram" if falhas == 0 else f"{falhas} teste(s) falharam")
    return falhas


if __name__ == "__main__":
    sys.exit(main())
