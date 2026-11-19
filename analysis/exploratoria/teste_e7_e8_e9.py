#!/usr/bin/env python3
"""Tests of the functions of E7, E8 and E9 on synthetic data. No real data is read.

    python analysis/exploratoria/teste_e7_e8_e9.py
"""
import os
import sys

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
sys.path.insert(0, os.path.dirname(AQUI))
import e7_cka as E7  # noqa: E402
import e8_misto as E8  # noqa: E402
import e9_fragmentacao as E9  # noqa: E402

falhas = []


def checar(cond, o_que):
    print(("ok    " if cond else "FALHA ") + o_que)
    if not cond:
        falhas.append(o_que)


def teste_cka():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3000, 40))
    y = x @ rng.normal(size=(40, 30)) + rng.normal(scale=2.0, size=(3000, 30))
    direto = E7.cka_linear(x, y)
    s = E7.somas_vazias(40, 30)
    for i in range(0, 3000, 700):
        E7.acumular(s, x[i:i + 700], y[i:i + 700])
    checar(abs(E7.cka_de_somas(**s) - direto) < 1e-10, f"CKA from sums equals the direct value ({direto:.4f})")
    partes = [E7.somas_vazias(40, 30) for _ in range(3)]
    for k, (a, b) in enumerate(((0, 1000), (1000, 2200), (2200, 3000))):
        E7.acumular(partes[k], x[a:b], y[a:b])
    checar(abs(E7.cka_de_somas(**E7.juntar(partes)) - direto) < 1e-10, "CKA of joined groups equals the pooled value")
    q, _ = np.linalg.qr(rng.normal(size=(40, 40)))
    checar(abs(E7.cka_linear(x, 3.0 * x @ q + 5.0) - 1.0) < 1e-10, "CKA is 1 under rotation, scaling and shift")
    z = rng.normal(size=(3000, 30))
    checar(E7.cka_linear(x, z) < 0.05, "CKA of independent features is near 0")


def teste_manchas():
    g = np.zeros((6, 6), dtype=bool)
    g[1, 1] = g[2, 2] = True          # touch only at a corner
    g[4, 4:6] = True
    checar(E9.contar_manchas(g, 8) == 2 and E9.contar_manchas(g, 4) == 3, "patches with 8 and 4 neighbours")
    masc = np.zeros((E9.TAM, E9.TAM), dtype=bool)
    masc[:40, :40] = True
    m = np.full(1600, -3.0, dtype=np.float32)
    m[:5] = 2.0                         # five water pixels in one row: one patch
    y = np.zeros(1600, dtype=np.uint8)
    r = E9.unidade(masc, m, y)
    checar(r["manchas8"] == 1 and r["agua"] == 5 and abs(r["erro"] - 5 / 1600) < 1e-12, "patches and error of a unit")
    fi = E9.indice([3, 3], [999, 1000])
    checar(np.isnan(fi[0]) and abs(fi[1] - 30.0) < 1e-9, "index is NaN below 1,000 water pixels and 3/0.1 km2 at 1,000")
    ruim = E9.unidades_ruins(np.arange(8, dtype=float))
    checar(ruim.sum() == 2 and ruim[-2:].all(), "bad units are the top quarter of the error rate")


def auroc_bruto(s, p, w):
    num = den = 0.0
    for i in np.flatnonzero(p):
        for j in np.flatnonzero(~p):
            den += w[i] * w[j]
            num += w[i] * w[j] * (1.0 if s[i] > s[j] else 0.5 if s[i] == s[j] else 0.0)
    return num / den


def teste_auroc():
    rng = np.random.default_rng(1)
    s = rng.integers(0, 6, size=40).astype(float)
    p = rng.random(40) < 0.3
    w1 = np.ones(40)
    checar(abs(E9.auroc_ponderado(s, p, w1) - auroc_bruto(s, p, w1)) < 1e-12, "AUROC with ties equals brute force")
    W = rng.integers(0, 3, size=(5, 40)).astype(float)
    v = E9.auroc_ponderado(s, p, W)
    checar(all(abs(v[b] - auroc_bruto(s, p, W[b])) < 1e-12 for b in range(5)), "weighted AUROC per replicate")


def teste_amostra():
    rng = np.random.default_rng(2)
    n = 1200
    inicio = np.array([0, 500, 1200])
    unit_id = np.repeat([0, 1], [500, 700])
    y = rng.random(n) < 0.4
    m = np.where(rng.random(n) < 0.7, np.where(y, 2.0, -2.0), np.where(y, -1.0, 1.0)).astype(np.float32)
    faixa = E8.faixa3(rng.integers(-1, 30, size=n).astype(np.int16))
    a1 = E8.amostrar(np.random.default_rng(5), m, y, 0.8, unit_id, inicio, faixa, max_por_unidade=100)
    a2 = E8.amostrar(np.random.default_rng(5), m, y, 0.8, unit_id, inicio, faixa, max_por_unidade=100)
    erros = np.bincount(unit_id[(m > 0) != y], minlength=2)
    checar(np.array_equal(a1["unidade"], a2["unidade"]) and np.array_equal(a1["nao_sinalizado"], a2["nao_sinalizado"]),
           "the draw is fixed by the seed")
    checar(np.array_equal(np.bincount(a1["unidade"]), np.minimum(erros, 100)), "at most 100 errors per unit")
    checar(np.allclose(np.bincount(a1["unidade"], weights=a1["peso"]), erros), "weights add up to the errors of the unit")
    d = np.array([-1, 1, 2, 3, 10, 11], dtype=np.int16)
    checar(E8.faixa3(d).tolist() == [2, 0, 0, 1, 1, 2], "distance bands")


def teste_modelo():
    rng = np.random.default_rng(3)
    n_u, por_u = 60, 60
    unidade = np.repeat(np.arange(n_u), por_u)
    n = unidade.size
    regime = rng.choice(["R1", "R2", "R3"], size=n)
    dominio = np.where(unidade < 30, "teste", "alvo")
    classe = rng.choice(["falsa_agua", "falsa_terra"], size=n)
    faixa = rng.choice(["ate2", "3a10", "mais10"], size=n)
    agua_z = rng.normal(size=n_u)[unidade]
    efeito_u = rng.normal(scale=0.3, size=n_u)[unidade]
    eta = -1.0 + 1.0 * (dominio == "alvo") + 0.0 * (regime == "R3") + efeito_u
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-eta))).astype(np.int8)
    dados = {"y": y, "regime": regime, "dominio": dominio, "classe": classe, "faixa": faixa, "agua_z": agua_z,
             "unidade_global": unidade.astype(str), "evento": np.where(unidade < 30, "e1", "e2"),
             "execucao": regime, "peso": np.ones(n),
             "niveis": {"regime": ["R1", "R2", "R3"], "dominio": ["teste", "alvo"],
                        "classe": ["falsa_agua", "falsa_terra"], "faixa": ["ate2", "3a10", "mais10"]}}
    termos = ["regime", "dominio", "regime:dominio", "classe", "faixa", "agua_z"]
    exog, nomes = E8.matriz_fixa(dados, termos)
    checar(exog.shape[1] == 10 and nomes[0] == "intercepto", f"fixed design has 10 columns ({exog.shape[1]})")
    r = E8.ajustar_misto(dados, termos, ["unidade_global", "evento", "execucao"])
    dom = next(e for e in r["fixos"] if e["termo"] == "dominio[alvo]")
    checar(0.5 < dom["estimativa"] < 1.6, f"mixed model recovers the domain effect ({dom['estimativa']:.2f}, true 1.0)")
    g = E8.ajustar_gee(dados, termos)
    dom = next(e for e in g["fixos"] if e["termo"] == "dominio[alvo]")
    checar(0.5 < dom["estimativa"] < 1.6, f"GEE recovers the domain effect ({dom['estimativa']:.2f})")
    w = E8.ajustar_gee(dados, termos, pesos=True)
    dom = next(e for e in w["fixos"] if e["termo"] == "dominio[alvo]")
    checar(0.5 < dom["estimativa"] < 1.6, f"weighted GEE recovers the domain effect ({dom['estimativa']:.2f})")


if __name__ == "__main__":
    teste_cka()
    teste_manchas()
    teste_auroc()
    teste_amostra()
    teste_modelo()
    print(f"\n{len(falhas)} falha(s)")
    sys.exit(1 if falhas else 0)
