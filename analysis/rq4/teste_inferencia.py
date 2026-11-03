#!/usr/bin/env python3
"""Tests of inferencia.py. Usage: python analysis/rq4/teste_inferencia.py"""
import os
import sys

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import inferencia as I  # noqa: E402
import mascaras as M  # noqa: E402


def falha(f, *a):
    try:
        f(*a)
    except (ValueError, KeyError):
        return True
    return False


def grade_exemplo(h=1100, w=1030, seed=0):
    rng = np.random.default_rng(seed)
    cods = {}
    for k, d in enumerate(M.DISTANCIAS_M):
        c = np.full((h, w), M.FORA, np.uint8)
        c[:, :300 + 100 * k] = M.TERRA
        c[:, 100:200] = M.AGUA
        c[rng.random((h, w)) < 0.05] = M.BORDA
        c[0:6, 0:512] = M.SEM_DADO
        cods[d] = c
    nrefs = {d: (cods[d] <= M.AGUA).astype(np.uint8) for d in M.DISTANCIAS_M}
    estr = rng.integers(1, 4, (h, w)).astype(np.uint8)
    return cods, nrefs, estr


def tabela(cods):
    linhas = []
    for d in M.DISTANCIAS_M:
        for j in M.janelas(cods[d], "T", min_pontuados=1):
            linhas.append({**{k: str(v) for k, v in j.items()}, "distancia_m": str(d)})
    return linhas


def teste_janelas_e_ordem():
    cods, nrefs, estr = grade_exemplo()
    linhas = tabela(cods)
    jan = I.janelas_da_inferencia(linhas, "T")
    assert [(l, c) for _, l, c in jan] == sorted((l, c) for _, l, c in jan)
    assert I.janelas_da_inferencia(linhas, "outro") == []
    novos = I.atribuidos(1100, 1030, jan)
    rot = I.rotulos_tile(cods, nrefs, estr, jan, novos)
    assert I.conferir_com_tabela(rot, linhas, "T") == len(jan) * 3
    assert rot["codigo"].shape == (3, int(rot["tamanhos"].sum())) and rot["estrato"].shape == (rot["codigo"].shape[1],)
    assert int(rot["tamanhos_pontuados"].sum()) == int(rot["pontuado"].sum())
    # pixel order: window by window, assigned pixels in row-major order
    i = 1
    _, l, c = jan[i]
    ini = int(rot["tamanhos"][:i].sum())
    esperado = cods[1000][l:l + 512, c:c + 512][novos[i]]
    assert np.array_equal(rot["codigo"][0, ini:ini + rot["tamanhos"][i]], esperado)
    # a changed table is caught
    ruim = [dict(r) for r in linhas]
    k = next(j for j, r in enumerate(ruim) if r["kept"] == "True" and int(r["scored_pixels"]) > 0)
    ruim[k]["scored_pixels"] = str(int(ruim[k]["scored_pixels"]) + 1)
    assert falha(I.conferir_com_tabela, rot, ruim, "T")


def teste_atribuidos_cobre_uma_vez():
    h, w = 1100, 1030
    todas = [(f"w{l}_{c}", l, c) for l in M.inicios(h) for c in M.inicios(w)]
    novos = I.atribuidos(h, w, todas)
    cont = np.zeros((h, w), np.int32)
    for (_, l, c), n in zip(todas, novos):
        cont[l:l + 512, c:c + 512] += n
    assert (cont == 1).all()
    assert falha(I.atribuidos, h, w, [("x", 5, 0)])
    assert falha(I.atribuidos, h, w, [todas[0], todas[0]])


def teste_recortar_3d():
    g = np.arange(2 * 600 * 600).reshape(2, 600, 600)
    novo = np.zeros((512, 512), bool)
    novo[0, :3] = True
    assert I.recortar(g, 10, 20, novo).tolist() == [[6020, 6021, 6022], [366020, 366021, 366022]]


def teste_quantizar():
    q = I.quantizar(np.array([0.1, -0.13, 2.74, 40.0, -40.0], np.float32))
    assert q.dtype == np.int8 and q.tolist() == [0, -1, 11, 127, -127]


if __name__ == "__main__":
    for nome, f in sorted(globals().items()):
        if nome.startswith("teste_") and callable(f):
            f()
            print("ok", nome)
