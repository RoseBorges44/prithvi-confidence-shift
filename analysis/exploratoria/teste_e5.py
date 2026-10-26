#!/usr/bin/env python3
"""Tests of e5_r0.py on synthetic data.

1. normalizar_chaves accepts checkpoint keys with or without the prefix 'model.', reports unused keys and stops
   when a model tensor is missing.
2. The margins of one synthetic run, written as r0_<set>.npz, give the same AUROC_e, error rate, unflagged error
   rate and tau as an independent computation, and a file with other units is refused.

Usage: python analysis/exploratoria/teste_e5.py
"""
import os
import shutil
import sys
import tempfile

import numpy as np
from scipy.special import expit
from scipy.stats import rankdata

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(AQUI))
sys.path.insert(0, AQUI)
import analise_congelada as AC  # noqa: E402
import e5_r0 as E  # noqa: E402
import gerador_sintetico as G  # noqa: E402


def auroc_e_direto(m, y):
    """P(confidence of an error < confidence of a correct pixel), ties 1/2, by ranks."""
    a = np.abs(m.astype(np.float64))
    err = (m > 0) != y.astype(bool)
    r = rankdata(-a)                         # the least confident pixel gets the largest rank
    n1, n0 = err.sum(), (~err).sum()
    u = r[err].sum() - n1 * (n1 + 1) / 2     # Mann-Whitney U of the errors
    return float(u / (n1 * n0))


def teste_chaves():
    modelo = ["encoder.a", "decoder.b", "head.c"]
    mapa, sobras = E.normalizar_chaves(["model.encoder.a", "model.decoder.b", "model.head.c", "criterion.w"], modelo)
    assert mapa == {k: "model." + k for k in modelo} and sobras == ["criterion.w"], (mapa, sobras)
    mapa, sobras = E.normalizar_chaves(modelo, modelo)
    assert mapa == {k: k for k in modelo} and sobras == []
    try:
        E.normalizar_chaves(["model.encoder.a"], modelo)
    except ValueError as e:
        assert "have no tensor" in str(e)
    else:
        raise AssertionError("a missing tensor was not detected")
    print("keys: ok")


def teste_metricas(raiz):
    dados = G.gerar(os.path.join(raiz, "dados"), semente=23, escala=0.3)
    conf = os.path.join(raiz, "conf")
    AC.main(["--repo", dados["repo"], "--margens", dados["margens"], "--chave", dados["chave"], "--saida", conf,
             "--B", "50"])
    pasta = os.path.join(raiz, "r0")
    os.makedirs(pasta)
    for conj in E.CONJUNTOS:
        z = np.load(os.path.join(dados["margens"], f"run_x9_s1_{conj}.npz"))
        np.savez(os.path.join(pasta, f"r0_{conj}.npz"), chaves=z["chaves"], tamanhos=z["tamanhos"], margem=z["margem"])
    out = E.main(["--margens-r0", pasta, "--margens", dados["margens"], "--confirmatoria", conf,
                  "--saida", os.path.join(raiz, "e5")])
    rot = {c: AC.ler_rotulos(dados["margens"], c) for c in E.CONJUNTOS}
    m = {c: np.load(os.path.join(pasta, f"r0_{c}.npz"))["margem"] for c in E.CONJUNTOS}
    yv = rot["val"]["rotulo"].astype(bool)
    s = np.where(yv, expit(-m["val"].astype(np.float64)), expit(m["val"].astype(np.float64)))
    n = s.size
    k = ((n + 1) * 9 + 9) // 10                       # ceil((n + 1)(1 - alpha)), alpha = 1/10, in integers
    indice = ((n - 1) * k + n - 1) // n               # numpy quantile at level k/n, method 'higher' (nucleo.py)
    tau = 1.0 - np.sort(s)[indice]
    assert abs(out["tau"]["semana6"] - tau) < 1e-12, (out["tau"]["semana6"], tau)
    for conj in ("teste", "bolivia"):
        y = rot[conj]["rotulo"].astype(bool)
        err = (m[conj] > 0) != y
        nao_sin = expit(np.abs(m[conj].astype(np.float64))) >= tau
        assert abs(out[conj]["auroc_e"]["semana6"] - auroc_e_direto(m[conj], y)) < 1e-9, conj
        assert abs(out[conj]["taxa_erro"]["semana6"] - err.mean()) < 1e-12, conj
        assert abs(out[conj]["erro_nao_sinalizado"]["semana6"] - (err & nao_sin).sum() / err.sum()) < 1e-12, conj
    assert "dentro_semana6" in out["bolivia"]["auroc_e"]
    # a file with other units is refused
    z = dict(np.load(os.path.join(pasta, "r0_teste.npz")))
    z["chaves"] = z["chaves"][::-1]
    np.savez(os.path.join(pasta, "r0_teste.npz"), **z)
    try:
        E.main(["--margens-r0", pasta, "--margens", dados["margens"], "--confirmatoria", conf,
                "--saida", os.path.join(raiz, "e5b")])
    except ValueError as e:
        assert "units or sizes differ" in str(e), e
    else:
        raise AssertionError("units in another order were not detected")
    print("metrics: ok")


def main():
    teste_chaves()
    raiz = tempfile.mkdtemp(prefix="e5_")
    try:
        teste_metricas(raiz)
        print("all tests passed")
    finally:
        shutil.rmtree(raiz, ignore_errors=True)


if __name__ == "__main__":
    main()
