#!/usr/bin/env python3
"""Exploratory analysis E7 of prereg/plano_exploratorio_2.md: CKA between the pretrained and the fine-tuned encoder.

The token outputs are collected on GPU by notebooks/semana11_e7_cka.ipynb, which accumulates, per group of
inputs and per encoder block, the sums needed by linear CKA (`somas_vazias`, `cka_de_somas`) and writes one JSON
per run. This script joins those JSON files with the AUROC_e of each run on the pooled target, recomputed from
the margins, and writes the tables of E7.

    python analysis/exploratoria/e7_cka.py --repo REPO --margens DIR --chave KEY --confirmatoria DIR \
        --cka DIR_JSON --saida OUT

Nothing here is confirmatory. The frozen scripts are imported read-only and their SHA-256 is checked first.
"""
import argparse
import json
import os
import platform
import sys

import numpy as np
import scipy
from scipy.stats import spearmanr

AQUI = os.path.dirname(os.path.abspath(__file__))
ANALISE = os.path.dirname(AQUI)
sys.path.insert(0, ANALISE)
sys.path.insert(0, AQUI)
import analise_congelada as AC  # noqa: E402
import exploratorias as X  # noqa: E402

CAMADAS = (5, 11, 17, 23)
GRUPOS = ("teste",) + AC.EVENTOS_PRIMARIOS
TOL_AUROC = 5e-5             # plan: the regime mean must match regimes.csv to the 4th decimal
MIN_CKA_R1 = 0.9999          # plan: check run of R1
log = AC.log


# ----------------------------------------------------------------------------------------------
# Linear CKA
# ----------------------------------------------------------------------------------------------

def cka_linear(x, y):
    """Linear CKA of two representations of the same rows, x (n, p) and y (n, q), centered over the rows."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean(0)
    y = y - y.mean(0)
    xy = np.linalg.norm(x.T @ y) ** 2
    return float(xy / (np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y)))


def somas_vazias(p, q):
    """Accumulator of the sums of linear CKA for features of sizes p and q."""
    return {"n": 0, "sx": np.zeros(p), "sy": np.zeros(q), "sxx": np.zeros((p, p)), "syy": np.zeros((q, q)),
            "sxy": np.zeros((p, q))}


def acumular(s, x, y):
    """Adds the rows x (n, p) and y (n, q) to the sums, in float64."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    s["n"] += x.shape[0]
    s["sx"] += x.sum(0)
    s["sy"] += y.sum(0)
    s["sxx"] += x.T @ x
    s["syy"] += y.T @ y
    s["sxy"] += x.T @ y


def juntar(lista):
    """Sum of several accumulators (e.g. the pooled target from its events)."""
    out = {k: (0 if k == "n" else np.zeros_like(v)) for k, v in lista[0].items()}
    for s in lista:
        for k in out:
            out[k] = out[k] + s[k]
    return out


def cka_de_somas(n, sx, sy, sxx, syy, sxy):
    """Linear CKA from the accumulated sums; the centering over the n rows is applied here."""
    sx, sy = np.asarray(sx, np.float64), np.asarray(sy, np.float64)
    cxy = np.asarray(sxy, np.float64) - np.outer(sx, sy) / n
    cxx = np.asarray(sxx, np.float64) - np.outer(sx, sx) / n
    cyy = np.asarray(syy, np.float64) - np.outer(sy, sy) / n
    return float(np.sum(cxy * cxy) / (np.linalg.norm(cxx) * np.linalg.norm(cyy)))


# ----------------------------------------------------------------------------------------------
# Joining the CKA files with AUROC_e
# ----------------------------------------------------------------------------------------------

def ler_cka(pasta, reg2cod):
    """{(regime, seed): record} from cka_<regime>_s<seed>.json; checks layers, groups and the R1 run."""
    out = {}
    for nome in sorted(os.listdir(pasta)):
        if not (nome.startswith("cka_") and nome.endswith(".json")):
            continue
        with open(os.path.join(pasta, nome), encoding="utf-8") as f:
            r = json.load(f)
        reg, sem = r["regime"], int(r["semente"])
        if r["codigo"] != reg2cod[reg]:
            raise ValueError(f"{nome}: code {r['codigo']} differs from the key")
        if tuple(r["camadas"]) != CAMADAS:
            raise ValueError(f"{nome}: layers {r['camadas']}")
        faltam = [g for g in GRUPOS + ("alvo",) if g not in r["cka"]]
        if faltam:
            raise ValueError(f"{nome}: groups missing: {faltam}")
        out[(reg, sem)] = r
    r1 = [k for k in out if k[0] == "R1"]
    if len(r1) != 1:
        raise ValueError(f"Expected one check run of R1, found {len(r1)}")
    menor = min(v for g in out[r1[0]]["cka"].values() for v in g.values())
    if menor < MIN_CKA_R1:
        raise ValueError(f"R1 check failed: CKA {menor:.6f} below {MIN_CKA_R1}")
    for reg in ("R2", "R3"):
        sementes = sorted(s for g, s in out if g == reg)
        if sementes != [1, 2, 3, 4, 5]:
            raise ValueError(f"{reg}: CKA files for seeds {sementes}, expected 1 to 5")
    return out


def auroc_por_corrida(args, res_c, reg2cod):
    ctx = AC.montar_contexto(args.repo, args.margens, usar_nuvem050=not res_c["meta"]["sem_nuvem050"])
    corridas = AC.listar_corridas(args.margens)
    codigos = AC.verificar_corridas(corridas)
    if AC.ler_chave(args.chave, codigos) != reg2cod:
        raise ValueError("The key differs from the one recorded by the confirmatory run")
    uni, px = X.universo_primario(ctx["alvo"])
    out = {}
    for reg in AC.REGIMES:
        cod = reg2cod[reg]
        for sem in sorted(corridas[cod]):
            m = AC.margens_universo(corridas[cod][sem], ctx, "alvo")[px]
            err = (m > 0) != uni.rotulo.astype(bool)
            ordem = AC.OrdemGlobal(np.abs(m), err, uni.unit_id, None)
            out[(reg, sem)] = ordem.metricas(np.ones(uni.n, dtype=bool))["auroc"]
            del m, ordem
        media = float(np.mean([out[(reg, s)] for s in sorted(corridas[cod])]))
        conf = res_c["regimes"][reg]["alvo"]["auroc"]["media"]
        if abs(media - conf) > TOL_AUROC:
            raise ValueError(f"AUROC_e of {reg} on the target: {media:.6f} here, {conf:.6f} in the confirmatory run")
    log("AUROC_e per run on the pooled target: regime means match the confirmatory run")
    return out


def figura(cka, auc, caminho):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    cores = {"R1": "#4a6fa5", "R2": "#8c8c8c", "R3": "#c0504d"}
    fig, eixo = plt.subplots(figsize=(4.8, 3.6))
    for reg in AC.REGIMES:
        pts = [(cka[k], auc[k]) for k in sorted(auc) if k[0] == reg and k in cka]
        if reg == "R1":
            pts = [(1.0, auc[k]) for k in sorted(auc) if k[0] == "R1"]
        if pts:
            xs, ys = zip(*pts)
            eixo.scatter(xs, ys, s=22, color=cores[reg], label=reg, lw=0)
    eixo.set_xlabel("CKA no alvo, bloco 23 (pré-treinado contra ajustado)", fontsize=8)
    eixo.set_ylabel("AUROC$_e$ no alvo", fontsize=8)
    eixo.xaxis.set_major_formatter(FuncFormatter(X._virgula))
    eixo.yaxis.set_major_formatter(FuncFormatter(X._virgula))
    eixo.tick_params(labelsize=8)
    eixo.grid(color="#eeeeee", lw=0.6)
    eixo.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(caminho, dpi=200)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--margens", required=True)
    ap.add_argument("--chave", required=True)
    ap.add_argument("--confirmatoria", required=True)
    ap.add_argument("--cka", required=True, help="folder with cka_<regime>_s<seed>.json")
    ap.add_argument("--saida", required=True)
    args = ap.parse_args(argv)

    log("E7 start")
    shas = X.conferir_congelados()
    conf, saida = os.path.abspath(args.confirmatoria), os.path.abspath(args.saida)
    if saida == conf or saida.startswith(conf + os.sep):
        raise ValueError("--saida must be outside the folder of the confirmatory run")
    if os.path.isdir(saida) and os.listdir(saida):
        raise ValueError(f"{saida} is not empty; choose a new folder")
    with open(os.path.join(conf, "resultados.json"), encoding="utf-8") as f:
        res_c = json.load(f)
    reg2cod = res_c["meta"]["chave"]
    registros = ler_cka(args.cka, reg2cod)
    log(f"{len(registros)} CKA files; the R1 check run has CKA of at least {MIN_CKA_R1} in every layer and group")
    auc = auroc_por_corrida(args, res_c, reg2cod)

    linhas, resumo = [], []
    for (reg, sem), r in sorted(registros.items()):
        for g in GRUPOS + ("alvo",):
            for c in CAMADAS:
                linhas.append([reg, sem, g, c, AC.fmt(r["cka"][g][str(c)]), r["tokens"][g]])
    for reg in ("R2", "R3"):
        for g in GRUPOS + ("alvo",):
            for c in CAMADAS:
                v = [registros[(reg, s)]["cka"][g][str(c)] for s in range(1, 6)]
                e = AC.estatistica_sementes(v)
                resumo.append([reg, g, c, AC.fmt(e["media"]), AC.fmt(e["dp"]), AC.fmt(e["minimo"]), AC.fmt(e["maximo"])])
    cka23 = {k: r["cka"]["alvo"]["23"] for k, r in registros.items()}
    chaves = [(reg, s) for reg in ("R2", "R3") for s in range(1, 6)]
    rho = float(spearmanr([cka23[k] for k in chaves], [auc[k] for k in chaves])[0])

    os.makedirs(os.path.join(saida, "tabelas"), exist_ok=True)
    os.makedirs(os.path.join(saida, "figuras"), exist_ok=True)
    tab = os.path.join(saida, "tabelas")
    AC.escrever_csv(os.path.join(tab, "e7_cka.csv"), ["regime", "semente", "grupo", "bloco", "cka", "tokens"], linhas)
    AC.escrever_csv(os.path.join(tab, "e7_cka_resumo.csv"),
                    ["regime", "grupo", "bloco", "media", "dp", "minimo", "maximo"], resumo)
    AC.escrever_csv(os.path.join(tab, "e7_auroc.csv"), ["regime", "semente", "auroc_e_alvo", "cka_alvo_bloco23"],
                    [[reg, s, AC.fmt(auc[(reg, s)]), AC.fmt(cka23.get((reg, s), 1.0 if reg == "R1" else None))]
                     for reg, s in sorted(auc)])
    figura(cka23, auc, os.path.join(saida, "figuras", "e7_cka_auroc.png"))
    meta = {"plano": "prereg/plano_exploratorio_2.md, E7", "camadas": CAMADAS, "congelados_sha256": shas,
            "e7_sha256": AC.sha256_arquivo(os.path.abspath(__file__)), "chave": reg2cod,
            "versoes": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
            "conferencias": "R1 com CKA >= 0.9999; media do AUROC_e por regime igual a da rodada confirmatoria "
                            f"(tolerancia {TOL_AUROC})"}
    with open(os.path.join(saida, "e7.json"), "w", encoding="utf-8") as f:
        json.dump(AC.para_json({"meta": meta, "spearman_cka23_auroc_r2_r3": rho, "corridas": len(chaves),
                                "auroc": {f"{r}|{s}": v for (r, s), v in auc.items()},
                                "cka": {f"{r}|{s}": v["cka"] for (r, s), v in registros.items()}}),
                  f, indent=1, ensure_ascii=False)
    log(f"Spearman between CKA (target, block 23) and AUROC_e over the 10 runs of R2 and R3: {rho:.3f}")
    log("done")
    return {"spearman": rho}


if __name__ == "__main__":
    main()
