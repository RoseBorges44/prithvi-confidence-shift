"""Synthetic repository and margins for testing the analysis (prereg/preregistro.md, section 10).

The margin of a pixel is m = (2 y - 1) d with the signed evidence d ~ Normal(mu_unit, sigma). A pixel is an
error when d < 0, and its confidence is |d|, so the AUROC_e of a homogeneous population has a closed form
(`auroc_e_analitico`). Every regime has its own (mu, sigma) on the source and on the target, so the effects
that the analysis must find are planted and known.
"""
import csv
import json
import os
import shutil

import numpy as np
from scipy import integrate, stats

AQUI = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(AQUI)

# regime -> domain -> (mu, sigma); 'val' is drawn like the source
PARAMETROS = {
    "R1": {"fonte": (6.0, 3.0), "alvo": (5.2, 3.0)},
    "R2": {"fonte": (6.2, 3.0), "alvo": (5.0, 3.0)},
    "R3": {"fonte": (6.4, 3.0), "alvo": (4.2, 3.2)},
}
CODIGOS = (("k7", "R1"), ("q2", "R2"), ("x9", "R3"))
SEMENTES = (1, 2, 3, 4, 5)
NUVEM_MAPAS = ("EMSR286_08ITUANGONORTH_DEL_MONIT02_v1", "EMSR286_09ITUANGOSOUTH_DEL_MONIT02_v1",
               "EMSR347_07ZOMBA_DEL_MONIT01_v1")


def auroc_e_analitico(mu, sigma):
    """AUROC_e when the signed evidence is Normal(mu, sigma): P(|d| of an error < |d| of a correct pixel)."""
    p_err = stats.norm.cdf(-mu / sigma)
    p_cor = stats.norm.cdf(mu / sigma)

    def f_err(t):
        return stats.norm.pdf(-t, mu, sigma) / p_err

    def s_cor(t):
        return stats.norm.sf(t, mu, sigma) / p_cor

    val, _ = integrate.quad(lambda t: f_err(t) * s_cor(t), 0, np.inf, limit=200)
    return val


def taxa_erro_analitica(mu, sigma):
    return stats.norm.cdf(-mu / sigma)


def _ler_csv(caminho):
    with open(caminho, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _escrever_csv(caminho, linhas, campos):
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(linhas)


def _chave(nome):
    import re
    return re.search(r"([A-Za-z-]+_\d+)", nome).group(1)


def gerar(raiz, semente=12345, escala=1.0, heterogeneidade=0.4, parametros=None, nuvem050=True, tta=True,
          janelas_por_mapa_nuvem=6, desloc_semente=0.05):
    """Write `raiz/repo/splits/...` and `raiz/margens/...`. Returns a dict with the paths and the layout.

    escala multiplies the number of pixels of every unit (1.0 gives about 0.5 million scored pixels)."""
    parametros = parametros or PARAMETROS
    rng = np.random.default_rng(semente)
    repo = os.path.join(raiz, "repo")
    marg = os.path.join(raiz, "margens")
    os.makedirs(os.path.join(repo, "splits"), exist_ok=True)
    os.makedirs(marg, exist_ok=True)
    for nome in ("sen1floods11_val.txt", "sen1floods11_test.txt", "sen1floods11_bolivia.txt", "t2_eventos.csv"):
        shutil.copy(os.path.join(REPO, "splits", nome), os.path.join(repo, "splits", nome))

    eventos = {r["map_id"]: r for r in _ler_csv(os.path.join(REPO, "splits", "t2_eventos.csv"))}
    linhas = _ler_csv(os.path.join(REPO, "splits", "t2_janelas.csv"))
    campos = list(linhas[0].keys())
    nuvem_ok = {m for m, r in eventos.items() if r["c1_country"] == "True" and r["c2_aligned"] == "True"
                and float(r["cloud_frac"]) <= 0.5}
    usados = {m for m, r in eventos.items() if r["core"] == "True" or r["full_set"] == "True"
              or (nuvem050 and m in nuvem_ok)}
    if nuvem050:
        for m in NUVEM_MAPAS:
            for i in range(janelas_por_mapa_nuvem):
                linhas.append({"window_id": f"{m}_r{(i // 3) * 512:05d}_c{(i % 3) * 512:05d}", "map_id": m,
                               "event": eventos[m]["event"], "core": "False", "row_off": str((i // 3) * 512),
                               "col_off": str((i % 3) * 512), "frac_missing_input": "0.0", "scored_pixels": "0",
                               "water_pixels": "0", "land_pixels": "0", "kept_primary": "True",
                               "kept_sensitivity": "True"})

    # layout of the units
    def tamanho():
        return max(50, int(rng.integers(900, 3000) * escala))

    unidades = {"val": [], "teste": [], "bolivia": [], "t2": []}
    for conj, arq in (("val", "sen1floods11_val.txt"), ("teste", "sen1floods11_test.txt"),
                      ("bolivia", "sen1floods11_bolivia.txt")):
        chaves = [_chave(l.strip()) for l in open(os.path.join(REPO, "splits", arq)) if l.strip()]
        vazio = set(rng.choice(len(chaves), size={"val": 3, "teste": 1, "bolivia": 0}[conj], replace=False)) \
            if conj != "bolivia" else set()
        unidades[conj] = [(c, 0 if i in vazio else tamanho()) for i, c in enumerate(chaves)]
    for r in linhas:
        if r["map_id"] in usados and r["kept_sensitivity"] == "True":
            unidades["t2"].append((r["window_id"], tamanho()))

    # labels, boundary flags and unit means (shared by all runs)
    rotulos, bordas, mu_unid = {}, {}, {}
    for conj, lista in unidades.items():
        ys, bs, zs = [], [], []
        for chave, n in lista:
            frac = rng.uniform(0.08, 0.5)
            ys.append((rng.random(n) < frac).astype(np.uint8))
            bs.append(rng.random(n) < 0.1)
            zs.append(rng.normal(0, 1))
        rotulos[conj] = ys
        bordas[conj] = bs
        mu_unid[conj] = zs
    # fix the window table with the generated counts
    contagem = {}
    for (chave, n), y in zip(unidades["t2"], rotulos["t2"]):
        contagem[chave] = (n, int(y.sum()), int(n - y.sum()))
    for r in linhas:
        if r["window_id"] in contagem:
            n, a, t = contagem[r["window_id"]]
            r["scored_pixels"], r["water_pixels"], r["land_pixels"] = str(n), str(a), str(t)
    _escrever_csv(os.path.join(repo, "splits", "t2_janelas.csv"), linhas, campos)

    # label files
    for conj, lista in unidades.items():
        dados = {"chaves": np.array([c for c, _ in lista]), "tamanhos": np.array([n for _, n in lista], dtype=np.int64),
                 "rotulo": np.concatenate(rotulos[conj]).astype(np.uint8)}
        if conj != "val":
            dados["borda"] = np.concatenate(bordas[conj])
        np.savez_compressed(os.path.join(marg, f"labels_{conj}.npz"), **dados)

    # runs
    dominio = {"val": "fonte", "teste": "fonte", "bolivia": "alvo", "t2": "alvo"}
    for cod, reg in CODIGOS:
        for sem in SEMENTES:
            deslocamento = rng.normal(0, desloc_semente) if desloc_semente > 0 else 0.0
            for conj, lista in unidades.items():
                mu, sigma = parametros[reg][dominio[conj]]
                ms, mt = [], []
                for (chave, n), y, z in zip(lista, rotulos[conj], mu_unid[conj]):
                    d = rng.normal(mu + deslocamento + heterogeneidade * z, sigma, n)
                    m = ((2.0 * y - 1.0) * d).astype(np.float32)
                    ms.append(m)
                    mt.append((0.9 * m + rng.normal(0, 0.3, n)).astype(np.float32))
                dados = {"chaves": np.array([c for c, _ in lista]),
                         "tamanhos": np.array([n for _, n in lista], dtype=np.int64), "margem": np.concatenate(ms)}
                if tta and conj != "val":
                    dados["margem_tta"] = np.concatenate(mt)
                np.savez_compressed(os.path.join(marg, f"run_{cod}_s{sem}_{conj}.npz"), **dados)
    chave = {cod: reg for cod, reg in CODIGOS}
    with open(os.path.join(marg, "chave_codigos.json"), "w") as f:
        json.dump(chave, f)
    return {"repo": repo, "margens": marg, "chave": os.path.join(marg, "chave_codigos.json"), "unidades": unidades,
            "rotulos": rotulos, "bordas": bordas, "linhas": linhas}
