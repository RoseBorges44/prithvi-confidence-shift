#!/usr/bin/env python3
"""Exploratory analysis E9 of prereg/plano_exploratorio_2.md: fragmentation index per unit.

Two steps.

1. mascaras: re-reads the labels of the source test, Bolivia and the T2 windows with the code of the inference
   notebook and writes, per unit, the mask of scored pixels on the 512 x 512 grid. It stops unless the units, the
   sizes and the labels equal labels_<set>.npz.

    python analysis/exploratoria/e9_fragmentacao.py mascaras --repo REPO --margens DIR --s1f11-rotulos DIR \
        --janelas DIR --saida DIR_MASK

2. analise: per run and unit, the predicted water mask (margin > 0) on the grid, its connected patches, the
   fragmentation index (patches per km2 of predicted water, Kolluru et al. 2026), the error rate and the mean MSP;
   per regime and set, the AUROC of each score for detecting bad units (error rate in the top quarter), their
   difference with a 90% interval, and the index per event.

    python analysis/exploratoria/e9_fragmentacao.py analise --repo REPO --margens DIR --chave KEY \
        --confirmatoria DIR --mascaras DIR_MASK --saida OUT

Nothing here is confirmatory. The frozen scripts are imported read-only and their SHA-256 is checked first.
"""
import argparse
import json
import os
import platform
import sys

import numpy as np
import scipy
from scipy import ndimage
from scipy.special import expit
from scipy.stats import rankdata, spearmanr

AQUI = os.path.dirname(os.path.abspath(__file__))
ANALISE = os.path.dirname(AQUI)
sys.path.insert(0, ANALISE)
sys.path.insert(0, AQUI)
import analise_congelada as AC  # noqa: E402
import e3_borda as E3  # noqa: E402
import exploratorias as X  # noqa: E402
import nucleo as N  # noqa: E402
import reamostragem as R  # noqa: E402

TAM = 512
KM2_POR_PIXEL = 1e-4                    # 10 m pixels
MIN_PIXELS_AGUA = 1000                  # plan: 0.1 km2 of predicted water
QUANTIL_RUIM = 0.75                     # plan: error rate in the top quarter of its set
SEMENTE_BOOT = 3                        # plan
B_PADRAO = 10_000
ESTRUTURA = {8: np.ones((3, 3), dtype=bool), 4: ndimage.generate_binary_structure(2, 1)}
CONJ_MASK = ("teste", "bolivia", "t2")
log = AC.log


# ----------------------------------------------------------------------------------------------
# Step 1: masks of scored pixels
# ----------------------------------------------------------------------------------------------

def ler_rotulo_janela(caminho):
    import rasterio
    with rasterio.open(caminho) as s:
        if s.count != 1 or (s.height, s.width) != (TAM, TAM):
            raise ValueError(f"{caminho}: window label {s.count} x {s.height} x {s.width}")
        return E3.preencher(s.read(1, masked=True), -1.0).astype(np.int16)


def passo_mascaras(args):
    X.conferir_congelados()
    os.makedirs(args.saida, exist_ok=True)
    for conj in CONJ_MASK:
        rot = AC.ler_rotulos(args.margens, conj)
        mascaras, tamanhos, rotulos = [], [], []
        for c in rot["chaves"]:
            if conj == "t2":
                y = ler_rotulo_janela(os.path.join(args.janelas, "label", f"{c}.tif"))
            else:
                y = E3.ler_rotulo_s1f11(os.path.join(args.s1f11_rotulos, f"{c}_LabelHand.tif"))
            pont = y >= 0
            mascaras.append(np.packbits(pont.ravel()))
            tamanhos.append(int(pont.sum()))
            rotulos.append(y[pont].astype(np.uint8))
        if not np.array_equal(np.array(tamanhos, dtype=np.int64), rot["tamanhos"]):
            raise ValueError(f"{conj}: scored pixels per unit differ from labels_{conj}.npz")
        if not np.array_equal(np.concatenate(rotulos), rot["rotulo"]):
            raise ValueError(f"{conj}: labels differ from labels_{conj}.npz")
        np.savez_compressed(os.path.join(args.saida, f"mascara_{conj}.npz"), chaves=np.array(rot["chaves"]),
                            tamanhos=np.array(tamanhos, dtype=np.int64), mascara=np.stack(mascaras))
        log(f"{conj}: {len(rot['chaves'])} units; scored pixels and labels equal to labels_{conj}.npz")
    log("masks done")


def ler_mascaras(pasta, conj, rot):
    z = np.load(os.path.join(pasta, f"mascara_{conj}.npz"), allow_pickle=False)
    if [str(c) for c in z["chaves"]] != rot["chaves"] or not np.array_equal(z["tamanhos"], rot["tamanhos"]):
        raise ValueError(f"mascara_{conj}.npz: units or sizes differ from labels_{conj}.npz")
    return {c: np.unpackbits(z["mascara"][i])[:TAM * TAM].astype(bool).reshape(TAM, TAM)
            for i, c in enumerate(rot["chaves"])}


# ----------------------------------------------------------------------------------------------
# Patches and scores
# ----------------------------------------------------------------------------------------------

def contar_manchas(agua, vizinhanca):
    """Number of connected patches of a boolean grid."""
    return int(ndimage.label(agua, structure=ESTRUTURA[vizinhanca])[1])


def unidade(mascara, m, y):
    """Patches (8 and 4 neighbours), predicted water pixels, error rate and mean MSP of one unit.

    Unscored pixels are not water and do not join patches."""
    if int(mascara.sum()) != m.size:
        raise ValueError("Mask and margins differ in size")
    pred = m > 0
    grade = np.zeros(mascara.shape, dtype=bool)
    grade[mascara] = pred
    return {"manchas8": contar_manchas(grade, 8), "manchas4": contar_manchas(grade, 4),
            "agua": int(pred.sum()), "erro": float(np.mean(pred != y.astype(bool))),
            "msp": float(np.mean(expit(np.abs(m).astype(np.float64))))}


def indice(manchas, agua):
    """Patches per km2 of predicted water; NaN below the minimum area."""
    manchas, agua = np.asarray(manchas, np.float64), np.asarray(agua, np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(agua >= MIN_PIXELS_AGUA, manchas / (agua * KM2_POR_PIXEL), np.nan)


def unidades_ruins(erro):
    """Units whose error rate is in the top quarter of their set (at or above the 75th percentile)."""
    erro = np.asarray(erro, dtype=np.float64)
    return erro >= np.quantile(erro, QUANTIL_RUIM)


def auroc_ponderado(escore, positivo, pesos):
    """AUROC of `escore` (higher means positive) with unit multiplicities; ties count 1/2.

    escore, positivo: (n,); pesos: (n,) or (B, n). Returns a float or an array (B,)."""
    escore = np.asarray(escore, dtype=np.float64)
    positivo = np.asarray(positivo, dtype=bool)
    w = np.atleast_2d(np.asarray(pesos, dtype=np.float64))
    ordem = np.argsort(escore, kind="stable")
    s = escore[ordem]
    p = positivo[ordem]
    w = w[:, ordem]
    wp, wn = w * p, w * ~p
    inicio = np.concatenate([[0], np.flatnonzero(s[1:] != s[:-1]) + 1])
    gp = np.add.reduceat(wp, inicio, axis=1)
    gn = np.add.reduceat(wn, inicio, axis=1)
    abaixo = np.cumsum(gn, axis=1) - gn
    num = np.sum(gp * (abaixo + 0.5 * gn), axis=1)
    den = gp.sum(1) * gn.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    return float(out[0]) if np.ndim(pesos) == 1 else out


# ----------------------------------------------------------------------------------------------
# Step 2: analysis
# ----------------------------------------------------------------------------------------------

def processar_corrida(arq, ctx, mascaras, uni_alvo, px):
    out = {}
    for nome, u_obj, m in (("teste", ctx["fonte"], AC.margens_universo(arq, ctx, "fonte")),
                           ("alvo", uni_alvo, AC.margens_universo(arq, ctx, "alvo")[px])):
        y = u_obj.rotulo
        cols = {k: np.zeros(u_obj.n) for k in ("manchas8", "manchas4", "agua", "erro", "msp")}
        for u in range(u_obj.n):
            a, b = u_obj.inicio[u], u_obj.inicio[u + 1]
            r = unidade(mascaras[u_obj.ids[u]], m[a:b], y[a:b])
            for k in cols:
                cols[k][u] = r[k]
        out[nome] = cols
    return out


def resumo_regime(est, sementes, u_obj, nome, cnt):
    """AUROCs of the index and of 1 - mean MSP, their difference with a 90% interval, and the Spearman."""
    res = {}
    for viz in (8, 4):
        auc_fi, auc_msp, rho, n_idx, reps = [], [], [], [], []
        for s in sementes:
            c = est[s][nome]
            fi = indice(c[f"manchas{viz}"], c["agua"])
            ok = np.isfinite(fi)
            ruim = unidades_ruins(c["erro"])
            esc_msp = 1.0 - c["msp"]
            auc_fi.append(auroc_ponderado(fi[ok], ruim[ok], np.ones(ok.sum())))
            auc_msp.append(auroc_ponderado(esc_msp[ok], ruim[ok], np.ones(ok.sum())))
            rho.append(float(spearmanr(fi[ok], esc_msp[ok])[0]))
            n_idx.append(int(ok.sum()))
            w = cnt[:, ok]
            reps.append(auroc_ponderado(fi[ok], ruim[ok], w) - auroc_ponderado(esc_msp[ok], ruim[ok], w))
        dif = np.mean(reps, axis=0)
        res[viz] = {"auroc_indice": float(np.mean(auc_fi)), "auroc_msp": float(np.mean(auc_msp)),
                    "diferenca": float(np.mean(auc_fi) - np.mean(auc_msp)),
                    "ic90": [float(v) for v in R.quantile_interval(dif, 0.90)],
                    "replicas_nan": int(np.isnan(dif).sum()), "spearman": float(np.mean(rho)),
                    "unidades_com_indice": n_idx, "unidades": int(u_obj.n)}
    return res


def passo_analise(args):
    log("E9 analysis start")
    shas = X.conferir_congelados()
    conf, saida = os.path.abspath(args.confirmatoria), os.path.abspath(args.saida)
    if saida == conf or saida.startswith(conf + os.sep):
        raise ValueError("--saida must be outside the folder of the confirmatory run")
    if os.path.isdir(saida) and os.listdir(saida):
        raise ValueError(f"{saida} is not empty; choose a new folder")
    with open(os.path.join(conf, "resultados.json"), encoding="utf-8") as f:
        res_c = json.load(f)
    ctx = AC.montar_contexto(args.repo, args.margens, usar_nuvem050=not res_c["meta"]["sem_nuvem050"])
    corridas = AC.listar_corridas(args.margens)
    codigos = AC.verificar_corridas(corridas)
    reg2cod = AC.ler_chave(args.chave, codigos)
    if reg2cod != res_c["meta"]["chave"]:
        raise ValueError("The key differs from the one recorded by the confirmatory run")
    rot = ctx["rot"]
    mascaras = {}
    for conj in CONJ_MASK:
        mascaras.update(ler_mascaras(args.mascaras, conj, rot[conj]))
    log("masks match the label files")
    uni, px = X.universo_primario(ctx["alvo"])
    fonte = ctx["fonte"]

    taus = {(e["codigo"], int(e["semente"])): e["tau"] for e in res_c["execucoes"]}
    est = {reg: {} for reg in AC.REGIMES}
    for reg in AC.REGIMES:
        cod = reg2cod[reg]
        for sem in sorted(corridas[cod]):
            arq = corridas[cod][sem]
            m_v = AC.ler_margens(arq["val"], rot["val"])
            _, tau = N.conformal_threshold(m_v, rot["val"]["rotulo"])
            del m_v
            X.igual(tau, taus[(cod, sem)], f"tau of {reg} s{sem}")
            est[reg][sem] = processar_corrida(arq, ctx, mascaras, uni, px)
            log(f"  run {reg} s{sem} done")

    rng = np.random.default_rng(SEMENTE_BOOT)
    estratos = {"teste": [np.flatnonzero(fonte.evento == e) for e in sorted(set(fonte.evento))],
                "alvo": [np.flatnonzero(uni.evento == e) for e in X.EVENTOS]}
    resumo, linhas, linhas_u, linhas_ev = {}, [], [], []
    for reg in AC.REGIMES:
        sementes = sorted(est[reg])
        for nome, u_obj in (("teste", fonte), ("alvo", uni)):
            cnt = R.stratified_counts(rng, estratos[nome], u_obj.n, args.B)
            r = resumo_regime(est[reg], sementes, u_obj, nome, cnt)
            resumo[(reg, nome)] = r
            for viz, d in r.items():
                linhas.append([reg, nome, viz, AC.fmt(d["auroc_indice"]), AC.fmt(d["auroc_msp"]), AC.fmt(d["diferenca"]),
                               AC.fmt(d["ic90"][0]), AC.fmt(d["ic90"][1]), d["replicas_nan"], AC.fmt(d["spearman"], 3),
                               min(d["unidades_com_indice"]), max(d["unidades_com_indice"]), d["unidades"]])
            media = {k: np.mean([est[reg][s][nome][k] for s in sementes], axis=0)
                     for k in ("manchas8", "manchas4", "agua", "erro", "msp")}
            for u in range(u_obj.n):
                linhas_u.append([reg, nome, u_obj.ids[u], u_obj.evento[u], int(u_obj.tamanhos[u])]
                                + [AC.fmt(float(media[k][u])) for k in ("manchas8", "manchas4", "agua", "erro", "msp")])
            for e in sorted(set(u_obj.evento)):
                sel = u_obj.evento == e
                v = []
                for s in sementes:
                    c = est[reg][s][nome]
                    agua = c["agua"][sel].sum()
                    v.append((c["manchas8"][sel].sum() / (agua * KM2_POR_PIXEL) if agua > 0 else np.nan,
                              c["manchas4"][sel].sum() / (agua * KM2_POR_PIXEL) if agua > 0 else np.nan,
                              float(np.sum(c["erro"][sel] * u_obj.tamanhos[sel]) / u_obj.tamanhos[sel].sum())))
                v = np.array(v)
                linhas_ev.append([reg, nome, e, int(sel.sum())] + [AC.fmt(float(x)) for x in v.mean(0)])

    os.makedirs(os.path.join(saida, "tabelas"), exist_ok=True)
    tab = os.path.join(saida, "tabelas")
    AC.escrever_csv(os.path.join(tab, "e9_auroc.csv"),
                    ["regime", "conjunto", "vizinhanca", "auroc_indice", "auroc_1_menos_msp_media", "diferenca",
                     "ic90_inf", "ic90_sup", "replicas_nan", "spearman_indice_msp", "unidades_com_indice_min",
                     "unidades_com_indice_max", "unidades"], linhas)
    AC.escrever_csv(os.path.join(tab, "e9_unidades.csv"),
                    ["regime", "conjunto", "unidade", "evento", "pixels", "manchas8", "manchas4", "pixels_agua_prevista",
                     "taxa_erro", "msp_media"], linhas_u)
    AC.escrever_csv(os.path.join(tab, "e9_eventos.csv"),
                    ["regime", "conjunto", "evento", "unidades", "indice8", "indice4", "taxa_erro"], linhas_ev)
    meta = {"plano": "prereg/plano_exploratorio_2.md, E9", "min_pixels_agua": MIN_PIXELS_AGUA,
            "quantil_unidade_ruim": QUANTIL_RUIM, "B": args.B, "semente_bootstrap": SEMENTE_BOOT,
            "congelados_sha256": shas, "e9_sha256": AC.sha256_arquivo(os.path.abspath(__file__)), "chave": reg2cod,
            "versoes": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
            "conferencias": "tau igual ao da rodada confirmatoria; mascaras conferidas contra labels_<conjunto>.npz"}
    with open(os.path.join(saida, "e9.json"), "w", encoding="utf-8") as f:
        json.dump(AC.para_json({"meta": meta, "resumo": {f"{r}|{c}": {f"vizinhanca{viz}": d for viz, d in v.items()}
                                                              for (r, c), v in resumo.items()}}),
                  f, indent=1, ensure_ascii=False)
    log("done")
    return resumo


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="passo", required=True)
    a = sub.add_parser("mascaras", help="step 1")
    a.add_argument("--repo", required=True)
    a.add_argument("--margens", required=True)
    a.add_argument("--s1f11-rotulos", required=True)
    a.add_argument("--janelas", required=True, help="folder of the T2 windows (with label/)")
    a.add_argument("--saida", required=True)
    b = sub.add_parser("analise", help="step 2")
    b.add_argument("--repo", required=True)
    b.add_argument("--margens", required=True)
    b.add_argument("--chave", required=True)
    b.add_argument("--confirmatoria", required=True)
    b.add_argument("--mascaras", required=True)
    b.add_argument("--saida", required=True)
    b.add_argument("--B", type=int, default=B_PADRAO)
    args = ap.parse_args(argv)
    return passo_mascaras(args) if args.passo == "mascaras" else passo_analise(args)


if __name__ == "__main__":
    main()
