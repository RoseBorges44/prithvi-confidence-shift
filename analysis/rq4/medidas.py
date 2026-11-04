#!/usr/bin/env python3
"""Measures of RQ4 (prereg/plano_rq4.md, section 3): disagreement between the model and the references.

Nothing here is an error rate: the references are two independent maps of the flood, not ground truth, so a pixel where
the model and the consensus differ is called a disagreement. The functions reuse analysis/nucleo.py (frozen): the
"error" of nucleo is the disagreement here, the confidence is |m| and MSP = sigmoid(|m|); a pixel is flagged when
MSP < tau, with the tau of the same run in the confirmatory analysis (execucoes.csv).

Pixels counted, for a distance d of the UFRGS mapped area: the pixels with code 0 or 1 at d (scored) that lie in a
window that enters at d (rq4_janelas.csv, column `kept`). The pixels are split in six disjoint cells,
stratum (1 built-up, 2 permanent water, 3 other) x number of covering references (1 or 2); the groups of the plan
(all, each stratum, each number of references) are sums of cells, so counts and histograms are exact for every group
and can be pooled over tiles.

Files read (written by notebooks/semana9_rq4_inferencia.ipynb): rotulos_<tile>.npz and run_<R>_s<seed>_<tile>.npz.
"""
import csv
import os
import sys

import numpy as np
from scipy.special import expit

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
sys.path.insert(0, os.path.join(AQUI, ".."))
import nucleo as N  # noqa: E402
import inferencia as I  # noqa: E402
import mascaras as M  # noqa: E402

N_CELULAS = 6
TILES = ("22JDM", "22JCN", "22JDN")
AGREGADO = "3 tiles"
DISTANCIA_PRIMARIA = 1000
SEMENTE_DOS_MAPAS = 1                # the seed drawn for the maps, fixed here before any map is made
NOMES_ESTRATO = {1: "construído", 2: "água permanente", 3: "outro"}
BLOCO_MAPA = 10                      # maps: blocks of 10 x 10 pixels (100 m)

# group -> cells of the six (cell = (stratum - 1) * 2 + (references - 1))
GRUPOS = [("todos", "todos", list(range(N_CELULAS)))]
GRUPOS += [("estrato", NOMES_ESTRATO[e], [(e - 1) * 2, (e - 1) * 2 + 1]) for e in (1, 2, 3)]
GRUPOS += [("referencias", f"{r} referência" + ("s" if r > 1 else ""), [(e - 1) * 2 + (r - 1) for e in (1, 2, 3)])
           for r in (1, 2)]

MEDIDAS = ("taxa_discordancia", "auroc_d", "discordancia_nao_sinalizada", "conjunto_vazio",
           "discordancia_entre_sinalizados", "discordancia_entre_nao_sinalizados")
# measure of the plan -> name of the measure in regimes.csv of the confirmatory analysis (error there, disagreement here)
MEDIDA_NO_ALVO = {"taxa_discordancia": "taxa_erro", "auroc_d": "auroc",
                  "discordancia_nao_sinalizada": "erro_nao_sinalizado", "conjunto_vazio": "conjunto_vazio"}


# ----------------------------------------------------------------------------------------------
# Selection of the pixels of one tile at one distance
# ----------------------------------------------------------------------------------------------

def janelas_que_entram(linhas, tile, distancia):
    """Set of window ids of `tile` that enter at `distancia`, from the rows of rq4_janelas.csv."""
    return {r["window_id"] for r in linhas
            if r["tile"] == tile and int(r["distancia_m"]) == int(distancia) and str(r["kept"]) == "True"}


def selecionar(rot, k, entram):
    """Pixels counted at distance index k of the labels file `rot`.

    entram: set of window ids that enter at that distance. Returns a dict with
    idx (position in the margin array of the file, which holds the pixels scored at some distance), y (reference
    water), celula (0..5) and n_janelas."""
    cod = rot["codigo"][k]
    pont = rot["pontuado"]
    ids = [str(w) for w in rot["janelas"]]
    ok_janela = np.array([w in entram for w in ids], dtype=bool)
    janela_do_pixel = np.repeat(np.arange(len(ids)), rot["tamanhos"])
    sel = (cod <= M.AGUA) & ok_janela[janela_do_pixel]
    if not pont[sel].all():
        raise ValueError("a scored pixel is missing from the margin array")
    pos = np.cumsum(pont) - 1
    nref = rot["nrefs"][k][sel].astype(np.int64)
    est = rot["estrato"][sel].astype(np.int64)
    if not np.isin(nref, (1, 2)).all():
        raise ValueError(f"scored pixels with {np.unique(nref).tolist()} covering references")
    if not np.isin(est, (1, 2, 3)).all():
        raise ValueError(f"strata {np.unique(est).tolist()}")
    return {"idx": pos[sel], "y": cod[sel] == M.AGUA, "celula": (est - 1) * 2 + (nref - 1),
            "n_janelas": int(ok_janela.sum())}


# ----------------------------------------------------------------------------------------------
# Counts and measures
# ----------------------------------------------------------------------------------------------

def contar(margem, sel, tau, bins):
    """Counts (6 cells x nucleo.CONTAGENS) and histograms of |m| (disagreement, agreement) of one run."""
    m = margem[sel["idx"]]
    y = sel["y"]
    disc = (m > 0) != y
    cont = N.unit_counts(sel["celula"], m, y, tau, N_CELULAS)
    he, hc = N.unit_histograms(sel["celula"], np.abs(m), disc, N_CELULAS, bins)
    return cont, he, hc


def auroc_exato(margem, sel):
    """Exact AUROC_d of all counted pixels, to check the histogram value."""
    m = margem[sel["idx"]]
    a = np.abs(m)
    disc = (m > 0) != sel["y"]
    return N.auroc_exact(a[disc], a[~disc])


def somar(partes):
    """Sum counts and histograms of several tiles (same bins)."""
    return (sum(p[0] for p in partes), sum(p[1] for p in partes), sum(p[2] for p in partes))


def _div(a, b):
    return float(a / b) if b > 0 else float("nan")


def medidas_do_grupo(cont, he, hc, celulas):
    """Measures of one group (sum of cells). Counts are returned with the measures."""
    c = cont[celulas].sum(0)
    n, disc, unf, sin = c[0], c[1], c[2], c[3]
    auroc = float(N.auroc_from_hist(he[celulas].sum(0), hc[celulas].sum(0)))
    return {"pixels": int(n), "discordancias": int(disc), "sinalizados": int(sin),
            "discordancias_nao_sinalizadas": int(unf),
            "taxa_discordancia": _div(disc, n), "auroc_d": auroc,
            "discordancia_nao_sinalizada": _div(unf, disc), "conjunto_vazio": _div(sin, n),
            "discordancia_entre_sinalizados": _div(disc - unf, sin),
            "discordancia_entre_nao_sinalizados": _div(unf, n - sin)}


def calcular(obter_margem, tau, tiles, rotulos, janelas_linhas, regimes, sementes, bins, distancias=None,
             conferir_exato=True, aviso=print):
    """All counts of the runs.

    obter_margem(regime, semente, tile) -> float32 margins of the pixels scored at some distance;
    tau[(regime, semente)]; rotulos[tile] -> labels file as a dict; janelas_linhas: rows of rq4_janelas.csv.
    Returns {"contagens": {(regime, semente, tile, distancia): (cont, he, hc)}, "dif_hist": max |histogram - exact|
    AUROC_d at the primary distance, "max_abs_m": largest |m| seen}."""
    distancias = list(distancias or [int(d) for d in rotulos[tiles[0]]["distancias"]])
    out, dif_max, max_abs = {}, 0.0, 0.0
    for tile in tiles:
        rot = rotulos[tile]
        ds = [int(d) for d in rot["distancias"]]
        sels = {d: selecionar(rot, ds.index(d), janelas_que_entram(janelas_linhas, tile, d)) for d in distancias}
        aviso(f"{tile}: " + ", ".join(f"{d} m {s['idx'].size:,} pixels em {s['n_janelas']} janelas" for d, s in sels.items()))
        for regime in regimes:
            for sem in sementes:
                margem = obter_margem(regime, sem, tile)
                max_abs = max(max_abs, float(np.abs(margem).max()))
                for d, sel in sels.items():
                    out[(regime, sem, tile, d)] = contar(margem, sel, tau[(regime, sem)], bins)
                if conferir_exato and DISTANCIA_PRIMARIA in sels:
                    cont, he, hc = out[(regime, sem, tile, DISTANCIA_PRIMARIA)]
                    h = float(N.auroc_from_hist(he.sum(0), hc.sum(0)))
                    e = auroc_exato(margem, sels[DISTANCIA_PRIMARIA])
                    if not (np.isnan(h) and np.isnan(e)):
                        dif_max = max(dif_max, abs(h - e))
                del margem
    return {"contagens": out, "dif_hist": dif_max, "max_abs_m": max_abs}


# ----------------------------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------------------------

def tabela_execucoes(contagens, tiles, regimes, sementes, distancias):
    """One row per run, scope (tile or the three pooled), distance and group."""
    linhas = []
    for regime in regimes:
        for sem in sementes:
            for d in distancias:
                por_escopo = {t: contagens[(regime, sem, t, d)] for t in tiles}
                por_escopo[AGREGADO] = somar(list(por_escopo.values()))
                for escopo, (cont, he, hc) in por_escopo.items():
                    for tipo, nome, cel in GRUPOS:
                        linhas.append({"regime": regime, "semente": sem, "escopo": escopo, "distancia_m": d,
                                       "tipo_grupo": tipo, "grupo": nome, **medidas_do_grupo(cont, he, hc, cel)})
    return linhas


def estatistica(v):
    v = np.asarray(v, dtype=np.float64)
    ok = v[~np.isnan(v)]
    if ok.size == 0:
        return {"media": float("nan"), "dp": float("nan"), "minimo": float("nan"), "maximo": float("nan"), "n": 0}
    return {"media": float(ok.mean()), "dp": float(ok.std(ddof=1)) if ok.size > 1 else float("nan"),
            "minimo": float(ok.min()), "maximo": float(ok.max()), "n": int(ok.size)}


def ler_alvo(linhas_regimes):
    """Target values of the confirmatory analysis from the rows of regimes.csv.

    Returns {(regime, measure of this plan): {"agregado": mean over seeds of the pooled target, "eventos": {event: mean}}}."""
    out = {}
    for r in linhas_regimes:
        conj = r["conjunto"]
        if conj != "alvo" and not conj.startswith("alvo|"):
            continue
        for nosso, deles in MEDIDA_NO_ALVO.items():
            if r["metrica"] == deles:
                d = out.setdefault((r["regime"], nosso), {"agregado": None, "eventos": {}})
                v = float(r["media_sementes"])
                if conj == "alvo":
                    d["agregado"] = v
                else:
                    d["eventos"][conj.split("|", 1)[1]] = v
    return out


def posicao(valor, eventos):
    """Where the seed mean falls against the range of the target events (described, not tested)."""
    if not eventos or np.isnan(valor):
        return ""
    lo, hi = min(eventos.values()), max(eventos.values())
    return "abaixo da faixa" if valor < lo else "acima da faixa" if valor > hi else "dentro da faixa"


def tabela_regimes(linhas_exec, alvo=None):
    """Mean, standard deviation and range over the seeds of each regime, next to the target values."""
    chaves = {}
    for r in linhas_exec:
        chaves.setdefault((r["regime"], r["escopo"], r["distancia_m"], r["tipo_grupo"], r["grupo"]), []).append(r)
    linhas = []
    for (regime, escopo, d, tipo, grupo), rs in chaves.items():
        for med in MEDIDAS:
            e = estatistica([r[med] for r in rs])
            lin = {"regime": regime, "escopo": escopo, "distancia_m": d, "tipo_grupo": tipo, "grupo": grupo,
                   "medida": med, "pixels": rs[0]["pixels"],
                   "discordancias_media": float(np.mean([r["discordancias"] for r in rs])),
                   "media_sementes": e["media"], "dp_sementes": e["dp"], "minimo": e["minimo"],
                   "maximo": e["maximo"], "sementes": e["n"], "alvo_agregado": "", "alvo_eventos_min": "",
                   "alvo_eventos_max": "", "posicao_nos_eventos": ""}
            a = (alvo or {}).get((regime, med)) if tipo == "todos" else None
            if a and a["agregado"] is not None:
                lin["alvo_agregado"] = a["agregado"]
                if a["eventos"]:
                    lin["alvo_eventos_min"] = min(a["eventos"].values())
                    lin["alvo_eventos_max"] = max(a["eventos"].values())
                    lin["posicao_nos_eventos"] = posicao(e["media"], a["eventos"])
            linhas.append(lin)
    return linhas


def tabela_celulas(contagens, tiles, regimes, sementes, distancias):
    """Counts of each cell (stratum x references) of each run, to rebuild any pooling."""
    linhas = []
    for (regime, sem, tile, d), (cont, _, _) in sorted(contagens.items()):
        for cel in range(N_CELULAS):
            linhas.append({"regime": regime, "semente": sem, "tile": tile, "distancia_m": d,
                           "estrato": NOMES_ESTRATO[cel // 2 + 1], "referencias": cel % 2 + 1,
                           "pixels": int(cont[cel, 0]), "discordancias": int(cont[cel, 1]),
                           "discordancias_nao_sinalizadas": int(cont[cel, 2]), "sinalizados": int(cont[cel, 3])})
    return linhas


def escrever_csv(caminho, linhas):
    if not linhas:
        raise ValueError(f"{caminho}: no rows")
    cab = list(linhas[0].keys())
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cab)
        for r in linhas:
            w.writerow([("" if isinstance(r[k], float) and np.isnan(r[k]) else
                         (f"{r[k]:.8g}" if isinstance(r[k], float) else r[k])) for k in cab])


# ----------------------------------------------------------------------------------------------
# Maps
# ----------------------------------------------------------------------------------------------

def grade_do_tile(valores, rot, altura, largura, preencher):
    """Put the per-pixel `valores` (windows' assigned pixels in file order) on the tile grid."""
    janelas = [(str(w), int(l), int(c)) for w, l, c in zip(rot["janelas"], rot["linha"], rot["coluna"])]
    novos = I.atribuidos(altura, largura, janelas)
    g = np.full((altura, largura), preencher, dtype=valores.dtype)
    ini = 0
    for (_, l, c), novo in zip(janelas, novos):
        n = int(novo.sum())
        sub = g[l:l + M.TAM, c:c + M.TAM]
        sub[novo] = valores[ini:ini + n]
        ini += n
    if ini != valores.size:
        raise ValueError("pixels left over when placing the values on the grid")
    return g


def medias_por_bloco(valor, valido, f=BLOCO_MAPA, linhas_por_vez=60):
    """Mean of `valor` over the valid pixels of each f x f block (NaN where a block has none)."""
    h, w = valor.shape
    if h % f or w % f:
        raise ValueError("the grid is not a multiple of the block")
    out = np.full((h // f, w // f), np.nan, np.float32)
    passo = f * linhas_por_vez
    for i in range(0, h, passo):
        v = valor[i:i + passo].astype(np.float32)
        ok = valido[i:i + passo]
        b = lambda x: x.reshape(x.shape[0] // f, f, w // f, f).sum((1, 3))   # noqa: E731
        s, n = b(np.where(ok, v, 0.0)), b(ok.astype(np.float32))
        with np.errstate(invalid="ignore", divide="ignore"):
            out[i // f:(i + v.shape[0]) // f] = np.where(n > 0, s / n, np.nan)
    return out


def blocos_do_tile(margem_q, margem, rot, tau, altura=10980, largura=10980):
    """Block summaries of one run on one tile: mean MSP, fraction flagged and fraction in disagreement.

    margem_q: int8 margins (round(4 m)) of every assigned pixel; margem: float32 margins of the pixels scored at some
    distance (they replace the quantized ones there). The disagreement is at the primary distance."""
    m = margem_q.astype(np.float32) / I.ESCALA_Q
    m[rot["pontuado"]] = margem
    grade = grade_do_tile(m, rot, altura, largura, np.float32(np.nan))
    tem = ~np.isnan(grade)
    msp = expit(np.abs(np.where(tem, grade, 0.0)))
    k = [int(d) for d in rot["distancias"]].index(DISTANCIA_PRIMARIA)
    cod = grade_do_tile(rot["codigo"][k], rot, altura, largura, np.uint8(255))
    pont = cod <= M.AGUA
    disc = pont & ((np.where(tem, grade, 0.0) > 0) != (cod == M.AGUA))
    return {"msp": medias_por_bloco(msp, tem), "sinalizado": medias_por_bloco(msp < tau, tem),
            "discordancia": medias_por_bloco(disc, pont)}
