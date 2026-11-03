#!/usr/bin/env python3
"""Tests of mascaras.py. Usage: python analysis/rq4/teste_mascaras.py"""
import os
import sys

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import mascaras as M  # noqa: E402


def falha(f, *a):
    try:
        f(*a)
    except (ValueError, AssertionError):
        return True
    return False


def teste_borda_igual_semana6():
    y = np.full((12, 12), -1, np.int16)
    y[2:10, 2:6] = 1
    y[2:10, 6:10] = 0
    b = M.borda_de(y)
    assert b[5, 4] and b[5, 5] and b[5, 6] and b[5, 7]          # within 2 pixels of the other class
    assert not b[5, 3] and not b[5, 8] and not b[5, 2]
    assert not b[0, 0] and not b[1, 4]                           # -1 never forms or receives a boundary
    y2 = y.copy()
    y2[:, 6:10] = -1
    assert not M.borda_de(y2).any()                               # water next to -1 only: no boundary


def teste_distancia_e_area():
    m = 3
    inund = np.zeros((20 + 2 * m, 20 + 2 * m), bool)
    inund[m + 10, m + 10] = True
    d = M.distancia_px(inund)
    a = M.area_mapeada(d, 20.0, margem=m)                         # 2 pixels
    assert a.shape == (20, 20)
    assert a[10, 10] and a[10, 12] and a[12, 10] and not a[10, 13] and not a[12, 12]   # sqrt(8) > 2
    assert M.area_mapeada(d, 30.0, margem=m)[12, 12]                # sqrt(8) <= 3
    # flood in the margin, outside the tile, still makes the pixels near it mapped
    inund2 = np.zeros_like(inund)
    inund2[1, m + 5] = True
    a2 = M.area_mapeada(M.distancia_px(inund2), 30.0, margem=m)
    assert a2[0, 5] and a2[1, 5] and not a2[2, 5]
    assert not M.area_mapeada(M.distancia_px(np.zeros_like(inund)), 1e9, margem=m).any()


def teste_transform_com_margem():
    t = (10.0, 0.0, 399960.0, 0.0, -10.0, 6800020.0)
    assert M.transform_com_margem(t, 200) == (10.0, 0.0, 397960.0, 0.0, -10.0, 6802020.0)


def teste_rasterizar_centro():
    from shapely.geometry import box
    t = (10.0, 0.0, 0.0, 0.0, -10.0, 100.0)
    # covers x 0 to 14 and y 86 to 100: centre of pixel (0, 0) at (5, 95) inside, of (0, 1) at (15, 95) outside
    r = M.rasterizar([box(0, 86, 14, 100)], t, (10, 10))
    assert r[0, 0] and not r[0, 1] and not r[1, 0] and r.sum() == 1
    assert not M.rasterizar([], t, (10, 10)).any()


def camadas_exemplo():
    f = (6, 8)
    z = np.zeros(f, bool)
    cob_u = z.copy(); cob_u[:, :6] = True
    inund = z.copy(); inund[:, :2] = True
    perm = z.copy(); perm[0, 4] = True
    u = M.camada_ufrgs(inund, cob_u, perm)
    aoi = z.copy(); aoi[:, 1:8] = True
    peg = z.copy(); peg[:5, :] = True
    na = z.copy(); na[0, 7] = True
    agua_e = z.copy(); agua_e[:, 1] = True
    agua_e[1, 3] = True                                            # water where UFRGS says land: disagreement
    np_e = z.copy(); np_e[2, 1] = True; np_e[3, 6] = True          # trace on UFRGS water; trace outside UFRGS
    e = M.camada_emsr720(aoi, peg, na, agua_e, np_e, perm)
    return f, u, e


def teste_consenso():
    f, u, e = camadas_exemplo()
    rot, mot, n = M.consenso([u, e], f)
    assert n[0, 0] == 1 and n[0, 1] == 2 and n[5, 1] == 1 and n[0, 7] == 0 and n[0, 6] == 1 and n[5, 7] == 0
    assert rot[0, 0] == 1 and rot[0, 1] == 1                       # water, one and two references
    assert rot[1, 3] == -1 and mot[1, 3] == M.DISCORDAM
    assert rot[2, 1] == -1 and mot[2, 1] == M.CLASSE_NAO_PONTUADA   # trace wins over the other reference's water
    assert mot[3, 6] == M.CLASSE_NAO_PONTUADA
    assert rot[0, 4] == 1 and n[0, 4] == 2                          # permanent water in both
    assert rot[2, 3] == 0                                           # land in both
    assert mot[0, 7] == M.FORA and mot[5, 7] == M.FORA              # not analysed; outside the footprint
    assert rot[5, 6] == -1 and mot[5, 6] == M.FORA                  # outside UFRGS, outside the footprint
    # two layers of the same reference count once
    e2 = dict(e)
    rot2, mot2, n2 = M.consenso([u, e, e2], f)
    assert (n2 == n).all() and (rot2 == rot).all()


def teste_codigos_ordem():
    f = (4, 4)
    rot = np.zeros(f, np.int8); rot[:, 2:] = 1
    mot = np.zeros(f, np.uint8)
    sem = np.zeros(f, bool); sem[0, 0] = True
    borda = np.zeros(f, bool); borda[:, 1] = True; borda[0, 0] = True
    prob = np.zeros(f, np.uint8); prob[1, 1] = 40; prob[2, 2] = 39; prob[3, 3] = 255
    scl = np.full(f, 4, np.uint8); scl[1, 1] = 9; scl[2, 2] = 3; scl[3, 2] = 0
    cod, cond = M.codigos(sem, rot, mot, borda, prob, scl)
    assert cod[0, 0] == M.SEM_DADO and cod[1, 1] == M.BORDA and cod[2, 2] == M.SCL
    assert cod[3, 3] == M.SEM_MASCARA and cod[3, 2] == M.SEM_MASCARA
    assert cod[0, 2] == M.AGUA and cod[0, 3] == M.AGUA and cod[1, 0] == M.TERRA
    assert cond[M.NUVEM][1, 1] and cond[M.SCL][1, 1] and not cond[M.NUVEM][3, 3]
    prob[0, 1] = 101
    assert falha(M.codigos, sem, rot, mot, borda, prob, scl)


def teste_estrato_permanente():
    wc = np.array([[50, 80, 10], [40, 90, 95]], np.uint8)
    e = M.estrato(wc)
    assert e.tolist() == [[1, 2, 3], [3, 3, 3]]
    assert falha(M.estrato, np.array([[0, 10]], np.uint8))
    assert M.permanente(np.array([89, 90, 100], np.uint8)).tolist() == [False, True, True]
    assert falha(M.permanente, np.array([101], np.uint8))


def teste_janelas():
    assert M.inicios(1100) == [0, 512, 588] and M.inicios(1024) == [0, 512] and M.inicios(10980)[-1] == 10468
    assert len(M.inicios(10980)) == 22
    h, w = 1100, 1030
    cod = np.full((h, w), M.FORA, np.uint8)
    cod[:, :300] = M.AGUA
    cod[:, 300:600] = M.TERRA
    cod[0:6, 0:512] = M.SEM_DADO                                    # 3072 of 262144 > 1%
    linhas = M.janelas(cod, "T")
    assert len(linhas) == 3 * 3
    total = sum(l["scored_pixels"] for l in linhas)
    assert total == int((cod <= 1).sum())                           # every scored pixel in exactly one window
    l0 = linhas[0]
    assert l0["row_off"] == 0 and l0["col_off"] == 0 and not l0["kept"] and l0["frac_missing_input"] > 0.01
    l1 = [l for l in linhas if (l["row_off"], l["col_off"]) == (0, 512)][0]
    assert l1["water_pixels"] == 0 and l1["land_pixels"] == 512 * 88 and l1["kept"]
    l2 = [l for l in linhas if (l["row_off"], l["col_off"]) == (0, 518)][0]   # last column, against the edge
    assert l2["scored_pixels"] == 0                                 # its pixels belong to the earlier windows
    l3 = [l for l in linhas if (l["row_off"], l["col_off"]) == (588, 0)][0]   # last row: only rows 1024 to 1099 new
    assert l3["scored_pixels"] == 76 * 512 and l3["kept"]


def teste_contagens():
    cod = np.array([[0, 1, 1], [3, 6, 1]], np.uint8)
    nrefs = np.array([[1, 2, 1], [0, 1, 2]], np.uint8)
    estr = np.array([[1, 2, 3], [3, 3, 1]], np.uint8)
    cond = {k: np.zeros(cod.shape, bool) for k in M.ORDEM}
    c = M.contagens(cod, cond, nrefs, estr)
    assert c["por_codigo"]["1"] == 3 and c["por_codigo"]["0"] == 1 and c["por_codigo"]["3"] == 1
    assert c["pontuados_por_nrefs"] == {"1": {"agua": 1, "terra": 1}, "2": {"agua": 2, "terra": 0}}
    assert c["pontuados_por_estrato"]["construído"] == {"agua": 1, "terra": 1}
    assert c["pontuados_por_estrato"]["água permanente"] == {"agua": 1, "terra": 0}


def teste_sha_conteudo():
    a = np.arange(6, dtype=np.uint8).reshape(2, 3)
    assert M.sha_conteudo(a) == M.sha_conteudo(a.copy()) != M.sha_conteudo(a.reshape(3, 2))
    assert M.sha_conteudo(a) != M.sha_conteudo(a.astype(np.int8))


def teste_emsr720():
    from shapely.geometry import box
    g = box(0, 0, 1, 1)
    ev = [({"notation": "Flooded area", "event_type": "5-Flood", "dmg_src_id": "4"}, g),
          ({"notation": "Flood trace", "event_type": "5-Flood", "dmg_src_id": 4}, g),
          ({"notation": "Not Applicable", "event_type": "6-Mass Movement", "dmg_src_id": "4.0"}, g)]
    pg = [({"obj_type": "Image Footprint", "or_src_id": "1"}, box(9, 9, 10, 10)),
          ({"obj_type": "Image Footprint", "or_src_id": "4"}, g),
          ({"obj_type": "Not Analysed - No data", "or_src_id": "997"}, g)]
    s = M.separar_emsr720(ev, pg)
    assert s["fonte_dano"] == "4" and len(s["agua"]) == 1 and len(s["nao_pontuada"]) == 2
    assert len(s["pegada"]) == 1 and s["pegada"][0].bounds == (0, 0, 1, 1) and len(s["nao_analisada"]) == 1
    assert falha(M.separar_emsr720, ev + [({"notation": "Flooded area", "event_type": "5", "dmg_src_id": 5}, g)], pg)
    assert falha(M.separar_emsr720, ev, pg[:1])
    assert falha(M.separar_emsr720, [({"notation": "Dike breach", "event_type": "5-Flood", "dmg_src_id": 4}, g)], pg)
    assert falha(M.separar_emsr720, [({"notation": "Not Applicable", "event_type": "5-Flood", "dmg_src_id": 4}, g)], pg)
    assert falha(M.separar_emsr720, ev, pg + [({"obj_type": "Other", "or_src_id": "4"}, g)])
    linhas = [{"src_id": "4", "src_date": "2024/05/06", "cat": "Post-event"},
              {"src_id": "1", "src_date": "2023/02/01", "cat": "Pre-event"}]
    assert M.fonte_do_dia(linhas, "4") and not M.fonte_do_dia(linhas, "1") and not M.fonte_do_dia(linhas, "7")


def teste_alinhamento():
    rng = np.random.default_rng(0)
    h, w = 120, 120
    img = np.zeros((h, w), bool)
    for _ in range(25):
        r, c = rng.integers(5, h - 25, 2)
        img[r:r + rng.integers(5, 20), c:c + rng.integers(5, 20)] = True
    camada = np.roll(img, (1, -2), axis=(0, 1))                     # layer shifted 1 down and 2 left
    res, n = M.alinhamento(img, camada, np.ones((h, w), bool))
    melhor = max(res, key=res.get)
    assert n > 0 and melhor == (1, -2) and res[melhor] == 1.0
    res0, _ = M.alinhamento(img, img, np.ones((h, w), bool))
    assert max(res0, key=res0.get) == (0, 0)


if __name__ == "__main__":
    for nome, f in sorted(globals().items()):
        if nome.startswith("teste_") and callable(f):
            f()
            print("ok", nome)
