#!/usr/bin/env python3
"""Tests of e3_borda.py.

A. Step 1 on small GeoTIFFs written here (Sen1Floods11-like chips, one T2 scene cut into four windows): the
   distances must equal a direct computation, 1 <= d <= 2 must equal the boundary flag of the inference
   notebook, and a changed boundary flag must stop the step.
B. Step 2 on the synthetic margins of analysis/gerador_sintetico.py, after the frozen analysis: every check
   against the confirmatory run must pass and the tables must be consistent.

Usage: python analysis/exploratoria/teste_e3.py
"""
import csv
import os
import shutil
import sys
import tempfile

import warnings

import numpy as np
from scipy.ndimage import maximum_filter

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(AQUI))
sys.path.insert(0, AQUI)
import analise_congelada as AC  # noqa: E402
import e3_borda as E  # noqa: E402
import gerador_sintetico as G  # noqa: E402


def borda_de(y):
    """Copy of the boundary flag of notebooks/semana6_inferencia_alvos.ipynb."""
    agua = (y == 1).astype(np.uint8)
    terra = (y == 0).astype(np.uint8)
    perto_agua = maximum_filter(agua, size=5, mode="constant", cval=0).astype(bool)
    perto_terra = maximum_filter(terra, size=5, mode="constant", cval=0).astype(bool)
    return ((agua == 1) & perto_terra) | ((terra == 1) & perto_agua)


def gravar_tif(caminho, dados, nodata=None):
    import rasterio
    dados = np.asarray(dados)
    if dados.ndim == 2:
        dados = dados[None]
    with rasterio.open(caminho, "w", driver="GTiff", height=dados.shape[1], width=dados.shape[2],
                       count=dados.shape[0], dtype=dados.dtype, nodata=nodata) as s:
        s.write(dados)


def manchas(rng, forma, p_agua=0.3, bloco=16, ruido=0.01):
    """Blobby water/land labels, so that boundaries look like real ones."""
    base = rng.random((forma[0] // bloco + 1, forma[1] // bloco + 1)) < p_agua
    y = np.kron(base, np.ones((bloco, bloco), dtype=bool))[:forma[0], :forma[1]].astype(np.int16)
    troca = rng.random(forma) < ruido
    return np.where(troca, 1 - y, y).astype(np.int16)


def teste_passo1(raiz):
    rng = np.random.default_rng(5)
    repo, marg, lb = (os.path.join(raiz, p) for p in ("repo", "margens", "s1f11"))
    jan, wf2 = os.path.join(raiz, "janelas"), os.path.join(raiz, "wf2")
    for p in (os.path.join(repo, "splits"), marg, lb, os.path.join(jan, "label"), os.path.join(wf2, "test", "gt")):
        os.makedirs(p, exist_ok=True)
    esperado = {}
    for conj, chaves in (("val", ["Ghana_1", "India_2"]), ("teste", ["USA_3", "Spain_4"]), ("bolivia", ["Bolivia_5"])):
        tam, rots, dists, bordas = [], [], [], []
        for c in chaves:
            y = manchas(rng, (512, 512))
            y[rng.random((512, 512)) < 0.05] = -1
            if c == "India_2":
                y[y == 1] = 0                      # a chip without water: every distance is -1
            gravar_tif(os.path.join(lb, f"{c}_LabelHand.tif"), y)
            p = y >= 0
            tam.append(int(p.sum())); rots.append(y[p].astype(np.uint8)); bordas.append(borda_de(y)[p])
            dists.append(E.distancia_outra_classe(y)[p])
        dados = {"chaves": np.array(chaves), "tamanhos": np.array(tam, np.int64), "rotulo": np.concatenate(rots)}
        if conj != "val":
            dados["borda"] = np.concatenate(bordas)
        np.savez(os.path.join(marg, f"labels_{conj}.npz"), **dados)
        esperado[conj] = np.concatenate(dists)
    # one T2 scene of 1024 x 1024 cut into four windows; scene channel 0: 1 valid, channel 1: 1 land, 2 water
    cena = manchas(rng, (1024, 1024), bloco=96, ruido=0.0005)
    valido = rng.random((1024, 1024)) > 0.03
    gt = np.stack([valido.astype(np.uint8), (cena + 1).astype(np.uint8)])
    gravar_tif(os.path.join(wf2, "test", "gt", "EMSRX_MAPA.tif"), gt)
    y_cena = np.where(valido, cena, -1).astype(np.int16)
    b_cena, d_cena = borda_de(y_cena), E.distancia_outra_classe(y_cena)
    linhas, tam, rots, dists, bordas, chaves = [], [], [], [], [], []
    for lin, col in ((0, 0), (0, 512), (512, 0), (512, 512)):
        jid = f"EMSRX_MAPA_r{lin:05d}_c{col:05d}"
        y = y_cena[lin:lin + 512, col:col + 512].copy()
        y[rng.random((512, 512)) < 0.02] = -1     # pixels without input data, unscored in the window only
        gravar_tif(os.path.join(jan, "label", f"{jid}.tif"), y)
        p = y >= 0
        linhas.append({"window_id": jid, "map_id": "EMSRX_MAPA", "row_off": str(lin), "col_off": str(col)})
        chaves.append(jid); tam.append(int(p.sum())); rots.append(y[p].astype(np.uint8))
        bordas.append(b_cena[lin:lin + 512, col:col + 512][p]); dists.append(d_cena[lin:lin + 512, col:col + 512][p])
    np.savez(os.path.join(marg, "labels_t2.npz"), chaves=np.array(chaves), tamanhos=np.array(tam, np.int64),
             rotulo=np.concatenate(rots), borda=np.concatenate(bordas))
    esperado["t2"] = np.concatenate(dists)
    with open(os.path.join(repo, "splits", "t2_janelas.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0]))
        w.writeheader(); w.writerows(linhas)

    saida = os.path.join(raiz, "dist")
    args = ["distancias", "--repo", repo, "--margens", marg, "--s1f11-rotulos", lb, "--janelas", jan, "--wf2", wf2]
    E.main(args + ["--saida", saida])
    for conj, d in esperado.items():
        z = np.load(os.path.join(saida, f"dist_{conj}.npz"))
        assert np.array_equal(z["distancia"], d.astype(np.int16)), conj
    assert (esperado["val"] == -1).any(), "no chip without the other class"
    assert (esperado["t2"] > 20).any(), "no T2 pixel farther than 20 from the other class"

    # a changed boundary flag stops the step
    z = dict(np.load(os.path.join(marg, "labels_teste.npz")))
    z["borda"] = z["borda"].copy(); z["borda"][0] = ~z["borda"][0]
    np.savez(os.path.join(marg, "labels_teste.npz"), **z)
    try:
        E.main(args + ["--saida", os.path.join(raiz, "dist2")])
    except ValueError as e:
        assert "boundary flag" in str(e), e
    else:
        raise AssertionError("a changed boundary flag was not detected")
    print("step 1: ok")


def teste_passo2(raiz):
    dados = G.gerar(os.path.join(raiz, "dados"), semente=17, escala=0.8)
    conf = os.path.join(raiz, "conf")
    AC.main(["--repo", dados["repo"], "--margens", dados["margens"], "--chave", dados["chave"], "--saida", conf,
             "--B", "100"])
    # distances consistent with the synthetic boundary flags: 1-2 on the flag, otherwise 3-40 or -1
    rng = np.random.default_rng(3)
    pasta = os.path.join(raiz, "dist_sintetico")
    os.makedirs(pasta)
    for conj in E.CONJ_DIST:
        rot = AC.ler_rotulos(dados["margens"], conj)
        n = rot["rotulo"].size
        d = rng.integers(3, 41, n).astype(np.int16)
        d[rng.random(n) < 0.05] = -1
        if rot["borda"] is not None:
            d[rot["borda"]] = rng.integers(1, 3, int(rot["borda"].sum()))
        np.savez(os.path.join(pasta, f"dist_{conj}.npz"), chaves=np.array(rot["chaves"]), tamanhos=rot["tamanhos"],
                 distancia=d)
    saida = os.path.join(raiz, "e3")
    out = E.main(["analise", "--repo", dados["repo"], "--margens", dados["margens"], "--chave", dados["chave"],
                  "--confirmatoria", conf, "--distancias", pasta, "--saida", saida])
    for (reg, conj), faixas in out["faixas"].items():
        s = sum(f["parcela_pixels"] for f in faixas)
        assert abs(s - 1.0) < 1e-12, (reg, conj, s)
    for chave, v in out["correlacoes"].items():
        assert -1 <= v["parcial_dado_borda"] <= 1, chave
    for nome in ("e3_faixas.csv", "e3_unidades.csv", "e3_correlacoes.csv"):
        assert os.path.exists(os.path.join(saida, "tabelas", nome)), nome
    assert os.path.exists(os.path.join(saida, "figuras", "e3_faixas.png"))
    print("step 2: ok")
    return saida


def main():
    warnings.filterwarnings("ignore", message="Dataset has no geotransform")
    raiz = tempfile.mkdtemp(prefix="e3_")
    try:
        teste_passo1(raiz)
        teste_passo2(raiz)
        print("all tests passed")
    finally:
        if "--manter" not in sys.argv:
            shutil.rmtree(raiz, ignore_errors=True)
        else:
            print("kept in", raiz)


if __name__ == "__main__":
    main()
