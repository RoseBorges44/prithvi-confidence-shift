#!/usr/bin/env python3
"""Exploratory analysis E3 of prereg/plano_exploratorio.md: distance to the label boundary.

Two steps.

1. distancias: for every scored pixel, the Chebyshev distance d to the nearest pixel of the other class in the
   label (pixels labeled -1 are neither class; -1 in the output when the chip or scene has no pixel of the other
   class). Sen1Floods11 chips are read one by one, so chip edges are not boundaries. For the T2 windows, d is
   computed on the whole scene label and then cut, as the boundary flag of the inference notebook. The labels are
   read with the same code as notebooks/semana6_inferencia_alvos.ipynb, and the step stops unless the units, the
   sizes and the labels equal labels_<set>.npz and the pixels with 1 <= d <= 2 are exactly the `borda` pixels.

    python analysis/exploratoria/e3_borda.py distancias --repo REPO --margens DIR --s1f11-rotulos DIR \
        --janelas DIR --wf2 DIR --saida DIR_DIST

2. analise: error rate, empty-set rate, unflagged error rate and AUROC_e per distance band, per regime and set;
   per unit, the Spearman correlation between the empty-set rate and the water fraction, the partial correlation
   given the boundary share b (share of pixels with d <= 2), and the correlation with the empty-set rate of the
   pixels with d > 10. Before writing anything it checks tau of every run and the plain Spearman values against
   the confirmatory run.

    python analysis/exploratoria/e3_borda.py analise --repo REPO --margens DIR --chave KEY --confirmatoria DIR \
        --distancias DIR_DIST --saida OUT

Nothing here is confirmatory. The frozen scripts are imported read-only and their SHA-256 is checked first.
"""
import argparse
import csv
import json
import os
import platform
import sys

import numpy as np
import scipy
from scipy.special import expit
from scipy.stats import rankdata, spearmanr

AQUI = os.path.dirname(os.path.abspath(__file__))
ANALISE = os.path.dirname(AQUI)
sys.path.insert(0, ANALISE)
sys.path.insert(0, AQUI)
import analise_congelada as AC  # noqa: E402
import exploratorias as X  # noqa: E402
import nucleo as N  # noqa: E402

TAM = 512
FAIXAS = ("1-2", "3-5", "6-10", "11-20", ">20")
LIMITES = (2, 5, 10, 20)                 # upper limit of each band but the last
INTERIOR = 10                            # plan: pixels with d > 10
MIN_INTERIOR = 1000                      # plan: units with at least 1,000 such pixels
SEM_OUTRA = -1
CONJ_DIST = ("val", "teste", "bolivia", "t2")
log = AC.log


# ----------------------------------------------------------------------------------------------
# Step 1: distances
# ----------------------------------------------------------------------------------------------

def distancia_outra_classe(y):
    """Chebyshev distance from each pixel to the nearest pixel of the other class; -1 where there is none.
    Pixels labeled -1 are neither class and get -1."""
    from scipy.ndimage import distance_transform_cdt
    agua, terra = y == 1, y == 0
    d = np.full(y.shape, SEM_OUTRA, dtype=np.int32)
    if terra.any():
        d[agua] = distance_transform_cdt(~terra, metric="chessboard")[agua]
    if agua.any():
        d[terra] = distance_transform_cdt(~agua, metric="chessboard")[terra]
    return d


def para_int16(d):
    return np.minimum(d, np.iinfo(np.int16).max).astype(np.int16)


def preencher(x, valor):
    """As in the inference notebook: masked values become `valor`."""
    x = np.ma.filled(x.astype(np.float32), np.nan)
    return np.where(np.isnan(x), np.float32(valor), x).astype(np.float32)


def ler_rotulo_s1f11(caminho):
    import rasterio
    with rasterio.open(caminho) as s:
        if s.count != 1 or (s.height, s.width) != (TAM, TAM):
            raise ValueError(f"{caminho}: label {s.count} x {s.height} x {s.width}")
        y = preencher(s.read(1, masked=True), -1.0).astype(np.int16)
    if not set(np.unique(y).tolist()) <= {-1, 0, 1}:
        raise ValueError(f"{caminho}: label values {np.unique(y)}")
    return y


def distancias_s1f11(chaves, pasta_lb):
    """Per chip, in the order of `chaves`: scored labels and distances (row-major order of the scored pixels)."""
    tamanhos, rotulos, dists = [], [], []
    for c in chaves:
        y = ler_rotulo_s1f11(os.path.join(pasta_lb, f"{c}_LabelHand.tif"))
        pont = y >= 0
        tamanhos.append(int(pont.sum()))
        rotulos.append(y[pont].astype(np.uint8))
        dists.append(para_int16(distancia_outra_classe(y)[pont]))
    return np.array(tamanhos, dtype=np.int64), np.concatenate(rotulos), np.concatenate(dists)


def distancias_t2(chaves, linhas, pasta_janelas, pasta_wf2):
    """Per window, in the order of `chaves`: scored labels of the window and distances computed on the scene."""
    import rasterio
    por_id = {r["window_id"]: r for r in linhas}
    faltam = [c for c in chaves if c not in por_id]
    if faltam:
        raise ValueError(f"Windows missing from the window tables: {faltam[:3]}")
    saida = {}
    for mapa in sorted({por_id[c]["map_id"] for c in chaves}):
        with rasterio.open(os.path.join(pasta_wf2, "test", "gt", f"{mapa}.tif")) as g:
            gt = g.read()
        cena = np.full(gt.shape[1:], -1, np.int16)
        cena[gt[1] == 1] = 0
        cena[gt[1] == 2] = 1
        cena[gt[0] != 1] = -1
        del gt
        d_cena = distancia_outra_classe(cena)
        for c in (c for c in chaves if por_id[c]["map_id"] == mapa):
            r = por_id[c]
            lin, col = int(r["row_off"]), int(r["col_off"])
            with rasterio.open(os.path.join(pasta_janelas, "label", f"{c}.tif")) as s:
                if s.count != 1 or (s.height, s.width) != (TAM, TAM):
                    raise ValueError(f"{c}: window label {s.count} x {s.height} x {s.width}")
                y = preencher(s.read(1, masked=True), -1.0).astype(np.int16)
            pont = y >= 0
            recorte = cena[lin:lin + TAM, col:col + TAM]
            if recorte.shape != (TAM, TAM) or not np.array_equal(y[pont], recorte[pont]):
                raise ValueError(f"{c}: window label differs from the scene crop")
            saida[c] = (int(pont.sum()), y[pont].astype(np.uint8), para_int16(d_cena[lin:lin + TAM, col:col + TAM][pont]))
        del cena, d_cena
        log(f"  {mapa}: done")
    tamanhos = np.array([saida[c][0] for c in chaves], dtype=np.int64)
    vazio8, vazio16 = np.zeros(0, np.uint8), np.zeros(0, np.int16)
    rotulo = np.concatenate([saida[c][1] for c in chaves] or [vazio8])
    dist = np.concatenate([saida[c][2] for c in chaves] or [vazio16])
    return tamanhos, rotulo, dist


def conferir_distancias(conj, rot, tamanhos, rotulo, dist):
    """Units, sizes and labels must equal labels_<set>.npz, and 1 <= d <= 2 must be exactly the borda flag."""
    if not np.array_equal(tamanhos, rot["tamanhos"]):
        raise ValueError(f"{conj}: scored pixels per unit differ from labels_{conj}.npz")
    if not np.array_equal(rotulo, rot["rotulo"]):
        raise ValueError(f"{conj}: labels differ from labels_{conj}.npz")
    if dist.size != rotulo.size or (dist == 0).any():
        raise ValueError(f"{conj}: distances have the wrong size or a zero")
    if rot["borda"] is not None and not np.array_equal((dist >= 1) & (dist <= 2), rot["borda"]):
        raise ValueError(f"{conj}: pixels with 1 <= d <= 2 differ from the boundary flag of labels_{conj}.npz")


def passo_distancias(args):
    X.conferir_congelados()
    os.makedirs(args.saida, exist_ok=True)
    janelas = AC.ler_csv(os.path.join(args.repo, "splits", "t2_janelas.csv"))
    caminho_nuvem = os.path.join(args.repo, "splits", "t2_janelas_nuvem050.csv")
    if os.path.exists(caminho_nuvem):
        janelas += AC.ler_csv(caminho_nuvem)
    resumo = {}
    for conj in CONJ_DIST:
        rot = AC.ler_rotulos(args.margens, conj)
        if conj == "t2":
            tam, y, d = distancias_t2(rot["chaves"], janelas, args.janelas, args.wf2)
        else:
            tam, y, d = distancias_s1f11(rot["chaves"], args.s1f11_rotulos)
        conferir_distancias(conj, rot, tam, y, d)
        np.savez_compressed(os.path.join(args.saida, f"dist_{conj}.npz"), chaves=np.array(rot["chaves"]),
                            tamanhos=tam, distancia=d)
        faixa = indice_faixa(d)
        resumo[conj] = {"pixels": int(d.size), "parcela_por_faixa": {FAIXAS[k]: float((faixa == k).mean())
                                                                     for k in range(len(FAIXAS))}}
        log(f"{conj}: {len(rot['chaves'])} units, {d.size:,} pixels; labels"
            + (" and boundary flags" if rot["borda"] is not None else "") + f" equal to labels_{conj}.npz")
    with open(os.path.join(args.saida, "distancias.json"), "w", encoding="utf-8") as f:
        json.dump(resumo, f, indent=1, ensure_ascii=False)
    log("distances done")


# ----------------------------------------------------------------------------------------------
# Step 2: analysis
# ----------------------------------------------------------------------------------------------

def indice_faixa(d):
    """Band index 0..4; a pixel with no pixel of the other class (d = -1) goes to the last band."""
    k = np.searchsorted(np.array(LIMITES), d, side="left").astype(np.int8)
    k[d == SEM_OUTRA] = len(FAIXAS) - 1
    return k


def ler_distancias(pasta, conj, rot):
    z = np.load(os.path.join(pasta, f"dist_{conj}.npz"), allow_pickle=False)
    if [str(c) for c in z["chaves"]] != rot["chaves"] or not np.array_equal(z["tamanhos"], rot["tamanhos"]):
        raise ValueError(f"dist_{conj}.npz: units or sizes differ from labels_{conj}.npz")
    d = z["distancia"]
    if rot["borda"] is not None and not np.array_equal((d >= 1) & (d <= 2), rot["borda"]):
        raise ValueError(f"dist_{conj}.npz: pixels with 1 <= d <= 2 differ from the boundary flag")
    return d


def metricas_faixas(m, y, tau, faixa, grupos):
    """For each group (name -> pixel mask or None) and band: pixels, errors, flagged, unflagged errors, AUROC_e."""
    y = np.asarray(y).astype(bool)
    a = np.abs(m)
    err = (m > 0) != y
    sinal = expit(a.astype(np.float64)) < tau
    out = {}
    for nome, mask in grupos.items():
        out[nome] = []
        for k in range(len(FAIXAS)):
            sel = faixa == k if mask is None else mask & (faixa == k)
            e, s = err[sel], sinal[sel]
            n, ne = int(sel.sum()), int(e.sum())
            auc = float(N.auroc_exact(a[sel][e], a[sel][~e])) if 0 < ne < n else float("nan")
            out[nome].append({"n": n, "erro": ne, "sinalizado": int(s.sum()), "nao_sinalizado": int((e & ~s).sum()),
                              "auroc": auc})
    return out


def por_unidade(m, tau, faixa_u_interior, unit_id, n_units):
    """Per unit: empty-set rate, and pixels and empty-set rate among the pixels with d > 10."""
    sinal = expit(np.abs(m).astype(np.float64)) < tau
    n = np.bincount(unit_id, minlength=n_units).astype(np.float64)
    s = np.bincount(unit_id[sinal], minlength=n_units).astype(np.float64)
    ni = np.bincount(unit_id[faixa_u_interior], minlength=n_units).astype(np.float64)
    si = np.bincount(unit_id[faixa_u_interior & sinal], minlength=n_units).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return s / n, si / ni, ni


def spearman_parcial(x, y, z):
    """Pearson correlation of the residuals of rank(x) and rank(y) after a linear regression on rank(z)."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    A = np.column_stack([np.ones_like(rz), rz])
    res_x = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    res_y = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    return float(np.corrcoef(res_x, res_y)[0, 1])


def _taxa(c, num, den):
    return c[num] / c[den] if c[den] > 0 else float("nan")


def figura_faixas(tab, caminho):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    metricas = (("taxa_erro", "Taxa de erro"), ("conjunto_vazio", "Taxa de conjunto vazio"),
                ("erro_nao_sinalizado", "Erros não sinalizados / erros"))
    cores = {"R1": "#4a6fa5", "R2": "#8c8c8c", "R3": "#c0504d"}
    estilos = {"teste": ("--", "teste da fonte"), "alvo": ("-", "alvo")}
    fig, eixos = plt.subplots(1, 3, figsize=(11, 3.8))
    x = np.arange(len(FAIXAS))
    for eixo, (chave, titulo) in zip(eixos, metricas):
        for reg in sorted({k[0] for k in tab}):
            for conj, (ls, nome) in estilos.items():
                if (reg, conj) in tab:
                    eixo.plot(x, [tab[(reg, conj)][f][chave]["media"] for f in range(len(FAIXAS))], ls=ls, marker="o",
                              ms=3.5, lw=1.2, color=cores.get(reg, "#333333"), label=f"{reg}, {nome}")
        eixo.set_xticks(x)
        eixo.set_xticklabels(FAIXAS, fontsize=8)
        eixo.set_xlabel("Distância à borda do rótulo (pixels)", fontsize=8)
        eixo.set_title(titulo, fontsize=9)
        eixo.yaxis.set_major_formatter(FuncFormatter(X._virgula))
        eixo.tick_params(labelsize=8)
        eixo.grid(color="#eeeeee", lw=0.6)
    alcas, nomes = eixos[0].get_legend_handles_labels()
    fig.legend(alcas, nomes, loc="lower center", ncol=len(nomes), fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(caminho, dpi=200)
    plt.close(fig)


def passo_analise(args):
    log("E3 analysis start")
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
    regimes = [r for r in AC.REGIMES if r in reg2cod]
    rot = ctx["rot"]
    dist = {c: ler_distancias(args.distancias, c, rot[c]) for c in CONJ_DIST}
    log("distance files match the label files and the boundary flags")
    uni, px = X.universo_primario(ctx["alvo"])
    fonte = ctx["fonte"]
    faixa = {"val": indice_faixa(dist["val"]), "teste": indice_faixa(dist["teste"]),
             "alvo": indice_faixa(np.concatenate([dist["bolivia"], dist["t2"]])[px])}
    grupos = {"val": {"val": None}, "teste": {"teste": None},
              "alvo": {"alvo": None, **{f"alvo|{e}": (uni.evento == e)[uni.unit_id] for e in X.EVENTOS}}}
    interior = {"teste": faixa["teste"] >= 3, "alvo": faixa["alvo"] >= 3}     # bands 11-20 and >20: d > 10
    if LIMITES[2] != INTERIOR:
        raise ValueError("The interior band must start above 10")

    taus = {(e["codigo"], int(e["semente"])): e["tau"] for e in res_c["execucoes"]}
    est = {}
    for reg in regimes:
        cod = reg2cod[reg]
        for sem in sorted(corridas[cod]):
            arq = corridas[cod][sem]
            m_v = AC.ler_margens(arq["val"], rot["val"])
            _, tau = N.conformal_threshold(m_v, rot["val"]["rotulo"])
            X.igual(tau, taus[(cod, sem)], f"tau of {reg} s{sem}")
            r = {"faixas": metricas_faixas(m_v, rot["val"]["rotulo"], tau, faixa["val"], grupos["val"])}
            del m_v
            m_f = AC.margens_universo(arq, ctx, "fonte")
            r["faixas"].update(metricas_faixas(m_f, fonte.rotulo, tau, faixa["teste"], grupos["teste"]))
            r["unid_teste"] = por_unidade(m_f, tau, interior["teste"], fonte.unit_id, fonte.n)
            del m_f
            m = AC.margens_universo(arq, ctx, "alvo")[px]
            r["faixas"].update(metricas_faixas(m, uni.rotulo, tau, faixa["alvo"], grupos["alvo"]))
            r["unid_alvo"] = por_unidade(m, tau, interior["alvo"], uni.unit_id, uni.n)
            del m
            est[(reg, sem)] = r
            log(f"  run {reg} s{sem} done")

    # per band, mean over seeds
    tab = {}
    linhas_f = []
    for reg in regimes:
        sementes = sorted(s for g, s in est if g == reg)
        for nome in ["val", "teste", "alvo"] + [f"alvo|{e}" for e in X.EVENTOS]:
            tab[(reg, nome)] = []
            for k in range(len(FAIXAS)):
                cs = [est[(reg, s)]["faixas"][nome][k] for s in sementes]
                tot = sum(c["n"] for c in est[(reg, sementes[0])]["faixas"][nome])
                d = {"parcela_pixels": cs[0]["n"] / tot if tot else float("nan")}
                for chave, f in (("taxa_erro", lambda c: _taxa(c, "erro", "n")),
                                 ("conjunto_vazio", lambda c: _taxa(c, "sinalizado", "n")),
                                 ("erro_nao_sinalizado", lambda c: _taxa(c, "nao_sinalizado", "erro")),
                                 ("auroc_e", lambda c: c["auroc"])):
                    v = [f(c) for c in cs]
                    d[chave] = AC.estatistica_sementes(v) if not np.isnan(v).any() else {"media": float("nan"), "dp": None}
                tab[(reg, nome)].append(d)
                linhas_f.append([reg, nome, FAIXAS[k], AC.fmt(d["parcela_pixels"])]
                                + [AC.fmt(d[c]["media"]) for c in ("taxa_erro", "conjunto_vazio", "erro_nao_sinalizado",
                                                                    "auroc_e")])

    # per unit
    linhas_u, corr = [], {}
    for nome, u_obj, chave_u in (("fonte", fonte, "unid_teste"), ("alvo", uni, "unid_alvo")):
        fr = u_obj.agua / u_obj.tamanhos
        borda = np.bincount(u_obj.unit_id, weights=(faixa["teste" if nome == "fonte" else "alvo"] == 0),
                            minlength=u_obj.n) / u_obj.tamanhos
        taxa, taxa_int, n_int = {}, {}, None
        for reg in regimes:
            sementes = sorted(s for g, s in est if g == reg)
            taxa[reg] = np.mean([est[(reg, s)][chave_u][0] for s in sementes], axis=0)
            with np.errstate(invalid="ignore"):
                taxa_int[reg] = np.mean([est[(reg, s)][chave_u][1] for s in sementes], axis=0)
            n_int = est[(reg, sementes[0])][chave_u][2]
            ok = n_int >= MIN_INTERIOR
            rho = float(spearmanr(fr, taxa[reg])[0])
            X.igual(rho, res_c["rq3_spearman"][f"{nome}_{reg}"]["spearman"], f"RQ3 Spearman of {reg} on {nome}")
            corr[(nome, reg)] = {"spearman": rho, "parcial_dado_borda": spearman_parcial(fr, taxa[reg], borda),
                                 "spearman_interior": float(spearmanr(fr[ok], taxa_int[reg][ok])[0]) if ok.sum() >= 3
                                 else float("nan"), "unidades": int(u_obj.n), "unidades_interior": int(ok.sum()),
                                 "spearman_borda_vazio": float(spearmanr(borda, taxa[reg])[0]),
                                 "spearman_borda_agua": float(spearmanr(borda, fr)[0])}
        for u in range(u_obj.n):
            linhas_u.append([nome, u_obj.ids[u], u_obj.evento[u], int(u_obj.tamanhos[u]), AC.fmt(float(fr[u])),
                             AC.fmt(float(borda[u])), int(n_int[u])]
                            + [AC.fmt(float(taxa[r][u])) for r in regimes]
                            + [AC.fmt(float(taxa_int[r][u])) if n_int[u] > 0 else "" for r in regimes])
    log("checks against the confirmatory run: tau and the RQ3 Spearman values match")

    os.makedirs(os.path.join(saida, "tabelas"), exist_ok=True)
    os.makedirs(os.path.join(saida, "figuras"), exist_ok=True)
    tabd = os.path.join(saida, "tabelas")
    AC.escrever_csv(os.path.join(tabd, "e3_faixas.csv"),
                    ["regime", "conjunto", "faixa", "parcela_pixels", "taxa_erro", "conjunto_vazio",
                     "erro_nao_sinalizado", "auroc_e"], linhas_f)
    AC.escrever_csv(os.path.join(tabd, "e3_unidades.csv"),
                    ["conjunto", "unidade", "evento", "pixels", "fracao_agua", "parcela_borda", "pixels_interior"]
                    + [f"taxa_vazio_{r}" for r in regimes] + [f"taxa_vazio_interior_{r}" for r in regimes], linhas_u)
    AC.escrever_csv(os.path.join(tabd, "e3_correlacoes.csv"),
                    ["conjunto", "regime", "spearman_agua_vazio", "parcial_dado_borda", "spearman_agua_vazio_interior",
                     "unidades", "unidades_interior", "spearman_borda_vazio", "spearman_borda_agua"],
                    [[c, r, AC.fmt(v["spearman"], 3), AC.fmt(v["parcial_dado_borda"], 3),
                      AC.fmt(v["spearman_interior"], 3), v["unidades"], v["unidades_interior"],
                      AC.fmt(v["spearman_borda_vazio"], 3), AC.fmt(v["spearman_borda_agua"], 3)]
                     for (c, r), v in corr.items()])
    figura_faixas({k: v for k, v in tab.items() if k[1] in ("teste", "alvo")},
                  os.path.join(saida, "figuras", "e3_faixas.png"))
    meta = {"plano": "prereg/plano_exploratorio.md, E3", "faixas": FAIXAS, "interior": f"d > {INTERIOR}",
            "min_interior": MIN_INTERIOR, "congelados_sha256": shas,
            "e3_sha256": AC.sha256_arquivo(os.path.abspath(__file__)), "chave": reg2cod,
            "versoes": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
            "conferencias": "tau e Spearman do RQ3 iguais aos da rodada confirmatoria (tolerancia 1e-9); "
                            "distancias conferidas contra rotulos e borda de labels_<conjunto>.npz"}
    with open(os.path.join(saida, "e3.json"), "w", encoding="utf-8") as f:
        json.dump(AC.para_json({"meta": meta,
                                "faixas": {f"{r}|{c}": v for (r, c), v in tab.items()},
                                "correlacoes": {f"{c}|{r}": v for (c, r), v in corr.items()}}),
                  f, indent=1, ensure_ascii=False)
    log("done")
    return {"faixas": tab, "correlacoes": corr}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="passo", required=True)
    a = sub.add_parser("distancias", help="step 1")
    a.add_argument("--repo", required=True)
    a.add_argument("--margens", required=True, help="folder with labels_<set>.npz")
    a.add_argument("--s1f11-rotulos", required=True, help="folder with the Sen1Floods11 *_LabelHand.tif")
    a.add_argument("--janelas", required=True, help="folder of the T2 windows (with label/)")
    a.add_argument("--wf2", required=True, help="WorldFloods v2 folder (with test/gt/)")
    a.add_argument("--saida", required=True)
    b = sub.add_parser("analise", help="step 2")
    b.add_argument("--repo", required=True)
    b.add_argument("--margens", required=True)
    b.add_argument("--chave", required=True)
    b.add_argument("--confirmatoria", required=True)
    b.add_argument("--distancias", required=True)
    b.add_argument("--saida", required=True)
    args = ap.parse_args(argv)
    return passo_distancias(args) if args.passo == "distancias" else passo_analise(args)


if __name__ == "__main__":
    main()
