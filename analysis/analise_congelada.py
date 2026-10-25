#!/usr/bin/env python3
"""Frozen analysis of prereg/preregistro.md.

Reads the logit margins of the confirmatory runs, computes the metrics of section 5, the stratified
bootstrap of section 6, the decision categories of section 1, the key secondary outcomes with Holm
correction, the preconditions of section 8 and the sensitivity and secondary analyses of section 9.

Usage
    python analysis/analise_congelada.py --repo REPO --margens DIR --chave DIR/chave_codigos.json --saida OUT

Files in --margens (written by the inference notebook)
    labels_<set>.npz      chaves, tamanhos, rotulo (uint8), borda (bool, only target sets and source test)
    run_<code>_s<seed>_<set>.npz   chaves, tamanhos, margem (float32), margem_tta (float32, optional)
    sets: val, teste (source test), bolivia, t2. <code> is the random regime code (letters and digits).
Pixels of a unit are the scored pixels in row-major order; margem = z_water - z_land, 32-bit inference.
The key file (a JSON object code -> "R1" | "R2" | "R3") is read after every code has been processed, only to
assign the codes to regimes in the contrasts and to label the tables.

Memory: about 10 GB at the peak for the full set of windows. Time: about two hours on a laptop.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import sys
import time
from collections import OrderedDict, defaultdict

import numpy as np
import scipy

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import categorias as C  # noqa: E402
import nucleo as N  # noqa: E402
import reamostragem as R  # noqa: E402

REGIMES = ("R1", "R2", "R3")
N_SEMENTES = 5
CONJUNTOS = ("val", "teste", "bolivia", "t2")
CHAVE_RX = re.compile(r"([A-Za-z-]+_\d+)")
ARQ_RX = re.compile(r"^run_(?P<cod>[A-Za-z0-9]+)_s(?P<sem>\d+)_(?P<conj>val|teste|bolivia|t2)\.npz$")
NUVEM_050 = ("EMSR286_08ITUANGONORTH_DEL_MONIT02_v1", "EMSR286_09ITUANGOSOUTH_DEL_MONIT02_v1",
             "EMSR347_07ZOMBA_DEL_MONIT01_v1")
NUVEM_MAX_050 = 0.50
EVENTOS_PRIMARIOS = ("Bolivia", "EMSR333", "EMSR342", "EMSR347", "EMSR438")
JANELAS_PRIMARIAS = {"EMSR333": 2, "EMSR342": 92, "EMSR347": 141, "EMSR438": 7}
MARGEM_S2_PP = 5.0
# pooled selections whose histogram AUROC_e is compared with the exact value (section 6, "in any run and set")
CHECADOS = ("todos", "bolivia", "t2", "prim", "sens", "c50", "full")
MAX_REFINOS = 3
SEMENTES_PILOTO = {0, 101, 102}
TAMANHO_BLOCO = 2


def agora():
    return time.strftime("%H:%M:%S")


def log(*partes):
    print(agora(), *partes, flush=True)


# ----------------------------------------------------------------------------------------------
# Input
# ----------------------------------------------------------------------------------------------

def chave_de(nome):
    achado = CHAVE_RX.search(os.path.basename(str(nome)))
    if not achado:
        raise ValueError(f"Chip name without a recognizable key: {nome!r}")
    return achado.group(1)


def ler_lista(repo, nome):
    with open(os.path.join(repo, "splits", nome), encoding="utf-8") as f:
        return [chave_de(linha.strip()) for linha in f if linha.strip()]


def ler_csv(caminho):
    with open(caminho, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def booleano(s):
    if s in ("True", "true", "1"):
        return True
    if s in ("False", "false", "0"):
        return False
    raise ValueError(f"Boolean expected, found {s!r}")


def listar_corridas(pasta):
    """{code: {seed: {set: path}}} from the file names."""
    achados = defaultdict(lambda: defaultdict(dict))
    for nome in sorted(os.listdir(pasta)):
        m = ARQ_RX.match(nome)
        if m:
            achados[m["cod"]][int(m["sem"])][m["conj"]] = os.path.join(pasta, nome)
        elif nome.startswith("run_") or (nome.startswith("labels_") and nome not in
                                         {f"labels_{c}.npz" for c in CONJUNTOS}):
            raise ValueError(f"File name not recognized: {nome}")
    return {c: dict(s) for c, s in achados.items()}


def ler_rotulos(pasta, conjunto):
    caminho = os.path.join(pasta, f"labels_{conjunto}.npz")
    z = np.load(caminho, allow_pickle=False)
    if z["rotulo"].dtype.kind not in "iub":
        raise ValueError(f"{caminho}: rotulo must be integer or boolean, found {z['rotulo'].dtype}")
    if "borda" in z.files and z["borda"].dtype != np.bool_:
        raise ValueError(f"{caminho}: borda must be boolean, found {z['borda'].dtype}")
    r = {"chaves": [str(c) for c in z["chaves"]], "tamanhos": z["tamanhos"].astype(np.int64),
         "rotulo": z["rotulo"].astype(np.uint8),
         "borda": z["borda"] if "borda" in z.files else None}
    if r["rotulo"].size != r["tamanhos"].sum():
        raise ValueError(f"{caminho}: label count differs from the sizes")
    if not np.isin(r["rotulo"], (0, 1)).all():
        raise ValueError(f"{caminho}: labels other than 0 and 1")
    if r["borda"] is not None and r["borda"].size != r["rotulo"].size:
        raise ValueError(f"{caminho}: boundary flags and labels differ in size")
    if r["borda"] is not None and not r["borda"].any():
        raise ValueError(f"{caminho}: no pixel is flagged as boundary")
    return r


def ler_margens(caminho, rot, campo="margem"):
    z = np.load(caminho, allow_pickle=False)
    if [str(c) for c in z["chaves"]] != rot["chaves"] or not np.array_equal(z["tamanhos"], rot["tamanhos"]):
        raise ValueError(f"{caminho}: units or sizes differ from the labels file")
    if "rotulo" in z.files and not np.array_equal(z["rotulo"], rot["rotulo"]):
        raise ValueError(f"{caminho}: labels differ from the labels file")
    if campo not in z.files:
        return None
    m = z[campo]
    if m.dtype != np.float32 or m.size != rot["tamanhos"].sum():
        raise ValueError(f"{caminho}: {campo} must be float32 with one value per scored pixel")
    if not np.isfinite(m).all():
        raise ValueError(f"{caminho}: {campo} has non-finite values")
    if m.size and float(np.ptp(m)) == 0.0:
        raise ValueError(f"{caminho}: {campo} is constant")
    return m


# ----------------------------------------------------------------------------------------------
# Context: units, strata, selections
# ----------------------------------------------------------------------------------------------

class Universo:
    """Units of one universe (source test, or the union of Bolivia and the T2 windows) and their pixels."""

    def __init__(self, ids, evento, tamanhos, rotulo, borda, atributos=None):
        self.ids = list(ids)
        self.evento = np.array(evento)
        self.tamanhos = np.asarray(tamanhos, dtype=np.int64)
        self.n = len(self.ids)
        self.rotulo = rotulo
        self.borda = borda
        self.unit_id = np.repeat(np.arange(self.n, dtype=np.int32), self.tamanhos)
        self.inicio = np.concatenate([[0], np.cumsum(self.tamanhos)])
        self.attr = atributos or {}
        self.agua = np.bincount(self.unit_id[rotulo == 1], minlength=self.n)

    def mascara(self, **condicoes):
        m = np.ones(self.n, dtype=bool)
        for k, v in condicoes.items():
            m &= self.attr[k] == v
        return m


def cortar(rot, manter):
    """Drop the units without scored pixels. Returns the arrays of the kept units in order."""
    return [c for c, t in zip(rot["chaves"], rot["tamanhos"]) if t > 0], rot["tamanhos"][rot["tamanhos"] > 0]


def montar_contexto(repo, pasta, usar_nuvem050=True):
    rot = {c: ler_rotulos(pasta, c) for c in CONJUNTOS}
    for conj in ("teste", "bolivia", "t2"):
        if rot[conj]["borda"] is None:
            raise ValueError(f"labels_{conj}.npz has no boundary flags (borda)")
    ctx = {"rot": rot}
    for conj, lista in (("val", "sen1floods11_val.txt"), ("teste", "sen1floods11_test.txt"),
                        ("bolivia", "sen1floods11_bolivia.txt")):
        esperado = ler_lista(repo, lista)
        if sorted(rot[conj]["chaves"]) != sorted(esperado):
            raise ValueError(f"Units of {conj} differ from splits/{lista}")

    # source test
    ids_s, tam_s = cortar(rot["teste"], None)
    ctx["fonte"] = Universo(ids_s, [i.split("_")[0] for i in ids_s], tam_s, rot["teste"]["rotulo"],
                            rot["teste"]["borda"])

    # target: Bolivia then T2 windows
    ids_b, tam_b = cortar(rot["bolivia"], None)
    janelas = {r["window_id"]: r for r in ler_csv(os.path.join(repo, "splits", "t2_janelas.csv"))}
    caminho_nuvem = os.path.join(repo, "splits", "t2_janelas_nuvem050.csv")
    if usar_nuvem050 and os.path.exists(caminho_nuvem):
        for r in ler_csv(caminho_nuvem):
            if r["window_id"] in janelas:
                raise ValueError(f"Window {r['window_id']} is in both window tables")
            if r["map_id"] not in NUVEM_050:
                raise ValueError(f"Window of a map outside the cloud 0.50 sensitivity analysis: {r['map_id']}")
            janelas[r["window_id"]] = r
    eventos = {r["map_id"]: r for r in ler_csv(os.path.join(repo, "splits", "t2_eventos.csv"))}
    nuvem = {}
    for mapa, r in eventos.items():
        nuvem[mapa] = (booleano(r["c1_country"]) and booleano(r["c2_aligned"])
                       and float(r["cloud_frac"]) <= NUVEM_MAX_050)
    adicionados = tuple(sorted(m for m, r in eventos.items() if nuvem[m] and not booleano(r["core"])))
    if adicionados != tuple(sorted(NUVEM_050)):
        raise ValueError(f"Maps added by the cloud 0.50 rule differ from the announced ones: {adicionados}")
    mapas_usados = {m for m, r in eventos.items()
                    if booleano(r["core"]) or booleano(r["full_set"]) or (usar_nuvem050 and nuvem[m])}
    esperado_t2 = sorted(w for w, r in janelas.items()
                         if r["map_id"] in mapas_usados and booleano(r["kept_sensitivity"]))
    if usar_nuvem050:
        faltam = [m for m in NUVEM_050 if not any(r["map_id"] == m for r in janelas.values())]
        if faltam:
            raise ValueError("No windows cut for the maps of the cloud 0.50 sensitivity analysis (expected in "
                             "splits/t2_janelas_nuvem050.csv): " + ", ".join(faltam))
    if sorted(c for c, t in zip(rot["t2"]["chaves"], rot["t2"]["tamanhos"]) if t > 0) != esperado_t2:
        raise ValueError("Windows of the T2 set differ from the windows expected from splits/t2_janelas.csv")
    ids_t, tam_t = cortar(rot["t2"], None)
    for w, t in zip(ids_t, tam_t):
        if int(janelas[w]["scored_pixels"]) != int(t):
            raise ValueError(f"Window {w}: scored pixels differ from splits/t2_janelas.csv")

    # grid rank of each window in its scene, from all rows of the table
    linhas = defaultdict(set)
    colunas = defaultdict(set)
    for r in janelas.values():
        linhas[r["map_id"]].add(int(r["row_off"]))
        colunas[r["map_id"]].add(int(r["col_off"]))
    rank_l = {m: {v: i for i, v in enumerate(sorted(s))} for m, s in linhas.items()}
    rank_c = {m: {v: i for i, v in enumerate(sorted(s))} for m, s in colunas.items()}

    ids = ids_b + ids_t
    n_b = len(ids_b)
    evento, mapa, conjunto, prim, sens, c50, full, bloco = [], [], [], [], [], [], [], []
    for i in ids_b:
        evento.append("Bolivia"); mapa.append(""); conjunto.append("bolivia")
        prim.append(True); sens.append(True); c50.append(True); full.append(True); bloco.append(i)
    for w in ids_t:
        r = janelas[w]
        e = eventos[r["map_id"]]
        evento.append(r["event"]); mapa.append(r["map_id"]); conjunto.append("t2")
        kp = booleano(r["kept_primary"])
        prim.append(booleano(e["core"]) and kp)
        sens.append(booleano(e["core"]) and booleano(r["kept_sensitivity"]))
        c50.append(nuvem[r["map_id"]] and kp)
        full.append(booleano(e["full_set"]) and kp)
        ml, mc = rank_l[r["map_id"]][int(r["row_off"])], rank_c[r["map_id"]][int(r["col_off"])]
        bloco.append(f"{r['map_id']}|{ml // TAMANHO_BLOCO}|{mc // TAMANHO_BLOCO}")
    # check the labels against the counts of the window table
    rotulo_t = np.concatenate([rot["bolivia"]["rotulo"], rot["t2"]["rotulo"]])
    tam_todos = np.concatenate([tam_b, tam_t])
    borda_t = np.concatenate([rot["bolivia"]["borda"], rot["t2"]["borda"]]) if (
        rot["bolivia"]["borda"] is not None and rot["t2"]["borda"] is not None) else None
    alvo = Universo(ids, evento, tam_todos, rotulo_t, borda_t,
                    {"mapa": np.array(mapa), "conjunto": np.array(conjunto), "prim": np.array(prim),
                     "sens": np.array(sens), "c50": np.array(c50), "full": np.array(full),
                     "bloco": np.array(bloco)})
    for k, w in enumerate(ids_t):
        u = n_b + k
        agua = int(alvo.agua[u])
        if agua != int(janelas[w]["water_pixels"]) or int(alvo.tamanhos[u]) - agua != int(janelas[w]["land_pixels"]):
            raise ValueError(f"Window {w}: label counts differ from splits/t2_janelas.csv")
    ctx["alvo"] = alvo
    # primary set must be the registered one
    contagem = defaultdict(int)
    for e, p in zip(alvo.evento, alvo.attr["prim"]):
        if p and e != "Bolivia":
            contagem[e] += 1
    if dict(contagem) != JANELAS_PRIMARIAS:
        raise ValueError(f"Primary windows differ from the registered ones: {dict(contagem)}")
    if int(alvo.attr["prim"][:n_b].sum()) != 15:
        raise ValueError("Bolivia must have 15 chips with scored pixels")
    if len(ids_s) != 89 or len(set(fonte_ev := [i.split("_")[0] for i in ids_s])) != 10:
        raise ValueError(f"Source test must have 89 units with scored pixels in 10 events, found {len(ids_s)}")
    if sum(1 for t in rot["val"]["tamanhos"] if t > 0) != 86:
        raise ValueError("Validation must have 86 chips with scored pixels")
    ctx["usar_nuvem050"] = usar_nuvem050
    ctx["val_chaves"] = rot["val"]
    return ctx


def selecoes_alvo(alvo, usar_nuvem050=True):
    """Boolean masks over the target units. Names: pools 'prim', 'sens', 'c50', 'full'; events
    'prim|E', 'sens|E', 'c50|E', 'full|E'; leave-one-event-out 'sem:E'."""
    sel = OrderedDict()
    variantes = ["prim", "sens", "full"] + (["c50"] if usar_nuvem050 else [])
    for v in variantes:
        base = alvo.attr[v]
        sel[v] = base.copy()
        for e in sorted(set(alvo.evento[base])):
            sel[f"{v}|{e}"] = base & (alvo.evento == e)
    for e in EVENTOS_PRIMARIOS:
        sel[f"sem:{e}"] = alvo.attr["prim"] & (alvo.evento != e)
    return sel


def pedidos_exatos(alvo, usar_nuvem050=True, reduzido=False):
    """List of (name, unit mask, pixel filter) for which exact values are computed on the target."""
    sel = selecoes_alvo(alvo, usar_nuvem050)
    pedidos = [("todos", np.ones(alvo.n, dtype=bool), "tudo")]
    if not reduzido:
        for conj in ("bolivia", "t2"):
            pedidos.append((conj, alvo.attr["conjunto"] == conj, "tudo"))
    for nome, mask in sel.items():
        if reduzido and not (nome == "prim" or nome.startswith("prim|")):
            continue
        pedidos.append((nome, mask, "tudo"))
    if alvo.borda is not None and not reduzido:
        for nome, mask in sel.items():
            if nome == "prim" or nome.startswith("prim|"):
                pedidos.append((nome, mask, "sem_borda"))
    return pedidos


# ----------------------------------------------------------------------------------------------
# Statistics of one run on one universe
# ----------------------------------------------------------------------------------------------

class OrdemGlobal:
    """Pixels of a universe in increasing order of the score a (stable), with the tie groups of a.

    Exact AUROC_e, AUGRC and E-AURC of any subset of units come from the counts per tie group, so the
    sort is done once per run and universe."""

    def __init__(self, a, err, unit_id, borda, m=None, y=None):
        if a.size == 0:
            raise ValueError("Empty universe")
        ordem = np.argsort(a, kind="stable")
        a_s = a[ordem]
        self.inicio = np.concatenate([[0], np.flatnonzero(a_s[1:] != a_s[:-1]) + 1])
        del a_s
        self.err_s = err[ordem]
        self.unit_s = unit_id[ordem].astype(np.int32)
        self.borda_s = borda[ordem] if borda is not None else None
        self.m_s = m[ordem] if m is not None else None
        self.y_s = y[ordem] if y is not None else None
        del ordem

    def _selecionar(self, mask_u, filtro):
        sel = mask_u[self.unit_s]
        if filtro == "sem_borda":
            if self.borda_s is None:
                raise ValueError("Boundary flags are needed for the filter sem_borda")
            sel &= ~self.borda_s
        return sel

    def metricas(self, mask_u, filtro="tudo"):
        sel = self._selecionar(mask_u, filtro)
        e = sel & self.err_s
        c = sel & ~self.err_s
        eg = np.add.reduceat(e, self.inicio, dtype=np.int64)
        cg = np.add.reduceat(c, self.inicio, dtype=np.int64)
        return {"auroc": float(N.auroc_from_hist(eg, cg)), "augrc": float(N.augrc_from_hist(eg, cg)),
                "eaurc": float(N.eaurc_from_hist(eg, cg)), "ne": int(eg.sum()), "nc": int(cg.sum())}

    def calibracao(self, mask_u, filtro, temperatura):
        sel = self._selecionar(mask_u, filtro)
        m, y = self.m_s[sel], self.y_s[sel]
        return {"antes": N.calibration_sorted(m, y, 1.0), "depois": N.calibration_sorted(m, y, temperatura)}


def unit_exact_auroc(a, err, inicio):
    n = len(inicio) - 1
    auc = np.full(n, np.nan)
    ne = np.zeros(n, dtype=np.int64)
    nc = np.zeros(n, dtype=np.int64)
    for u in range(n):
        au = a[inicio[u]:inicio[u + 1]]
        eu = err[inicio[u]:inicio[u + 1]]
        ne[u] = int(eu.sum())
        nc[u] = eu.size - ne[u]
        if ne[u] > 0 and nc[u] > 0:
            auc[u] = N.auroc_exact(au[eu], au[~eu])
    return auc, ne, nc


def compactar(he, hc):
    """Drop the bins that are empty in every unit. AUROC_e, AUGRC and E-AURC do not change when bins that
    are empty in every unit and every replicate are removed."""
    cols = np.flatnonzero((he.sum(0, dtype=np.int64) + hc.sum(0, dtype=np.int64)) > 0)
    return np.ascontiguousarray(he[:, cols]), np.ascontiguousarray(hc[:, cols])


def estatisticas_universo(uni, m, tau, bins, pedidos, temperatura=None, calib=(), com_hist=True):
    """Everything the analysis needs from one run on one universe.

    uni: Universo; m: float32 margins; tau: conformal threshold of this run; pedidos: list of
    (name, unit mask, filter) for the exact values; calib: names of the requested selections whose
    calibration is computed (needs temperatura)."""
    y = uni.rotulo.astype(bool)
    a = np.abs(m)
    err = (m > 0) != y
    out = {"max_abs_m": float(a.max())}
    if com_hist:
        out["he"], out["hc"] = N.unit_histograms(uni.unit_id, a, err, uni.n, bins)
        if uni.borda is not None:
            sem = ~uni.borda
            out["he_b"], out["hc_b"] = N.unit_histograms(uni.unit_id, a, err, uni.n, bins, mascara=sem)
    out["cont"] = N.unit_counts(uni.unit_id, m, y, tau, uni.n)
    if uni.borda is not None:
        sem = ~uni.borda
        out["cont_b"] = N.unit_counts(uni.unit_id[sem], m[sem], y[sem], tau, uni.n)
    out["auc_unid"], out["ne_unid"], out["nc_unid"] = unit_exact_auroc(a, err, uni.inicio)
    ordem = OrdemGlobal(a, err, uni.unit_id, uni.borda, m, y)
    out["exato"] = {(nome, filtro): ordem.metricas(mask, filtro) for nome, mask, filtro in pedidos}
    pedidos_d = {(n, f): mk for n, mk, f in pedidos}
    dif = 0.0
    dif_outros = 0.0
    if com_hist:
        for (nome, filtro), ex in out["exato"].items():
            he, hc = (out["he_b"], out["hc_b"]) if filtro == "sem_borda" else (out["he"], out["hc"])
            mask = pedidos_d[(nome, filtro)]
            h = float(N.auroc_from_hist(he[mask].sum(0, dtype=np.int64), hc[mask].sum(0, dtype=np.int64)))
            if not (math.isnan(h) and math.isnan(ex["auroc"])):
                if nome in CHECADOS and filtro == "tudo":
                    dif = max(dif, abs(h - ex["auroc"]))
                else:
                    dif_outros = max(dif_outros, abs(h - ex["auroc"]))
    out["dif_hist"] = dif
    out["dif_hist_outros"] = dif_outros
    if com_hist:
        out["he"], out["hc"] = compactar(out["he"], out["hc"])
        if "he_b" in out:
            out["he_b"], out["hc_b"] = compactar(out["he_b"], out["hc_b"])
    out["calib"] = {}
    if temperatura is not None:
        for nome in calib:
            out["calib"][nome] = ordem.calibracao(pedidos_d[(nome, "tudo")], "tudo", temperatura)
    return out


# ----------------------------------------------------------------------------------------------
# One run, one code, all runs
# ----------------------------------------------------------------------------------------------

def margens_universo(arquivos, ctx, qual, campo="margem"):
    """Concatenated margins of a universe ('fonte' or 'alvo') for one run. None if the field is absent."""
    rot = ctx["rot"]
    if qual == "fonte":
        return ler_margens(arquivos["teste"], rot["teste"], campo)
    mb = ler_margens(arquivos["bolivia"], rot["bolivia"], campo)
    mt = ler_margens(arquivos["t2"], rot["t2"], campo)
    if (mb is None) != (mt is None):
        raise ValueError(f"Field {campo} present in only one of the target files of {arquivos['t2']}")
    return None if mb is None else np.concatenate([mb, mt])


def universo_validacao(ctx):
    rv = ctx["rot"]["val"]
    return Universo(["val"], ["val"], [int(rv["tamanhos"].sum())], rv["rotulo"], None)


def processar_corrida(cod, sem, arquivos, ctx, bins, pedidos, com_tta):
    """Statistics of one run (code, seed). pedidos: dict with the requests for 'fonte' and 'alvo'."""
    t0 = time.time()
    uni_v = universo_validacao(ctx)
    m_v = ler_margens(arquivos["val"], ctx["rot"]["val"])
    y_v = ctx["rot"]["val"]["rotulo"]
    q_hat, tau = N.conformal_threshold(m_v, y_v)
    T = N.fit_temperature(m_v, y_v)
    todos_v = [("todos", np.ones(1, dtype=bool), "tudo")]
    val = estatisticas_universo(uni_v, m_v, tau, bins, todos_v, T, calib=("todos",))
    taxa_val = float(val["cont"][0][1] / val["cont"][0][0])
    if taxa_val >= 0.5:
        raise ValueError(f"Run {cod} s{sem}: validation error rate {taxa_val:.3f}; margins look inverted or broken")
    r = {"cod": cod, "sem": sem, "q_hat": q_hat, "tau": tau, "T": T, "val": val,
         "digest": {"val": hashlib.sha256(m_v.tobytes()).hexdigest()}}
    del m_v
    for qual in ("fonte", "alvo"):
        uni = ctx[qual]
        m = margens_universo(arquivos, ctx, qual)
        r[qual] = estatisticas_universo(uni, m, tau, bins, pedidos[qual][0], T, calib=pedidos[qual][1])
        r["digest"][qual] = hashlib.sha256(m.tobytes()).hexdigest()
        del m
        if com_tta:
            mt = margens_universo(arquivos, ctx, qual, "margem_tta")
            if mt is not None:
                r["tta_" + qual] = estatisticas_universo(uni, mt, tau, bins, pedidos[qual][2])
            del mt
    log(f"  run {cod} s{sem}: tau {tau:.4f}, T {T:.3f}, {time.time() - t0:.0f}s")
    return r


def margem_conjunto(lista_m, bloco=10_000_000):
    """Ensemble over seeds. Returns (effective margin of the mean probability, variance score), both float32.

    The variance score is -log10(variance between seeds of the water probability + 1e-30), so that a larger
    variance gives a smaller value, the same direction as |m|."""
    from scipy.special import expit
    n = lista_m[0].size
    m_ens = np.empty(n, dtype=np.float32)
    a_var = np.empty(n, dtype=np.float32)
    for i in range(0, n, bloco):
        pilha = np.stack([x[i:i + bloco] for x in lista_m])
        m_ens[i:i + bloco] = N.mean_probability_margin(pilha)
        v = expit(pilha.astype(np.float64)).var(axis=0)
        a_var[i:i + bloco] = -np.log10(v + 1e-30)
    return m_ens, a_var


def processar_conjunto(cod, sementes, corridas_cod, ctx, bins, pedidos):
    """Ensemble of the seeds of one code on the source test and on the target."""
    out = {}
    for qual in ("fonte", "alvo"):
        uni = ctx[qual]
        pilha = [margens_universo(corridas_cod[s], ctx, qual) for s in sementes]
        m_ens, a_var = margem_conjunto(pilha)
        del pilha
        out[qual] = estatisticas_universo(uni, m_ens, 2.0, bins, pedidos[qual][2])
        y = uni.rotulo.astype(bool)
        err = (m_ens > 0) != y
        ordem = OrdemGlobal(a_var, err, uni.unit_id, uni.borda)
        out["var_" + qual] = {(n, f): ordem.metricas(mk, f) for n, mk, f in pedidos[qual][2]}
        del m_ens, a_var, ordem
    return out


def verificar_corridas(corridas):
    codigos = sorted(corridas)
    if not codigos:
        raise ValueError("No run files found")
    for cod in codigos:
        sementes = sorted(corridas[cod])
        if set(sementes) & SEMENTES_PILOTO:
            raise ValueError(f"Code {cod} uses a pilot seed: {sorted(set(sementes) & SEMENTES_PILOTO)}")
        if len(sementes) != N_SEMENTES:
            raise ValueError(f"Code {cod} has {len(sementes)} seeds, {N_SEMENTES} expected")
        for s in sementes:
            faltam = [c for c in CONJUNTOS if c not in corridas[cod][s]]
            if faltam:
                raise ValueError(f"Code {cod} seed {s}: missing sets {faltam}")
    return codigos


def montar_pedidos(ctx):
    alvo, fonte = ctx["alvo"], ctx["fonte"]
    ped_alvo = pedidos_exatos(alvo, ctx["usar_nuvem050"], reduzido=False)
    ped_alvo_red = pedidos_exatos(alvo, ctx["usar_nuvem050"], reduzido=True)
    ped_fonte = [("todos", np.ones(fonte.n, dtype=bool), "tudo")]
    if fonte.borda is not None:
        ped_fonte.append(("todos", np.ones(fonte.n, dtype=bool), "sem_borda"))
    calib_alvo = ["todos", "prim"] + [f"prim|{e}" for e in EVENTOS_PRIMARIOS]
    return {"fonte": (ped_fonte, ["todos"], [ped_fonte[0]]),
            "alvo": (ped_alvo, calib_alvo, ped_alvo_red)}


def processar_tudo(ctx, corridas, bins, com_tta, est_diag=None):
    est_diag = est_diag if est_diag is not None else {}
    pedidos = montar_pedidos(ctx)
    est = {}
    ens = {}
    for cod in sorted(corridas):
        for sem in sorted(corridas[cod]):
            est[(cod, sem)] = processar_corrida(cod, sem, corridas[cod][sem], ctx, bins, pedidos, com_tta)
        ens[cod] = processar_conjunto(cod, sorted(corridas[cod]), corridas[cod], ctx, bins, pedidos)
        log(f"  ensemble {cod} done")
    for qual in ("val", "fonte", "alvo"):
        vistos = {}
        for chave, r in est.items():
            d = r["digest"][qual]
            if d in vistos:
                raise ValueError(f"Runs {vistos[d]} and {chave} have identical {qual} margins")
            vistos[d] = chave
    maximo = 0.0
    dif = 0.0
    dif_extra = 0.0
    for r in est.values():
        for chave in ("val", "fonte", "alvo"):
            maximo = max(maximo, r[chave]["max_abs_m"])
            dif = max(dif, r[chave]["dif_hist"])
        for chave in ("tta_fonte", "tta_alvo"):
            if chave in r:
                maximo = max(maximo, r[chave]["max_abs_m"])
                dif_extra = max(dif_extra, r[chave]["dif_hist"])
    for e in ens.values():
        for chave in ("fonte", "alvo"):
            maximo = max(maximo, e[chave]["max_abs_m"])
            dif_extra = max(dif_extra, e[chave]["dif_hist"])
    dif_outros = max(r[c]["dif_hist_outros"] for r in est.values() for c in ("fonte", "alvo"))
    log(f"  histogram difference of the pooled sets of the runs {dif:.2e}; of events, leave-one-event-out and "
        f"boundary subsets {dif_outros:.2e}; of the TTA and ensemble variants {dif_extra:.2e} (the last two only reported)")
    est_diag["dif_outros"] = dif_outros
    est_diag["dif_extra"] = dif_extra
    return est, ens, dif, maximo


# ----------------------------------------------------------------------------------------------
# Bootstrap scenarios
# ----------------------------------------------------------------------------------------------

def especificacoes(ctx):
    """Scenarios of section 6 (primary) and section 9 (sensitivity), in the order they are run."""
    base = {"filtro": "tudo", "esquema": "estrato", "extras": False, "variantes": False}
    e = OrderedDict()
    e["primaria"] = dict(base, sel="prim", variante="prim", extras=True, variantes=True)
    e["janelas_sensiveis"] = dict(base, sel="sens", variante="sens")
    if ctx["usar_nuvem050"]:
        e["nuvem_050"] = dict(base, sel="c50", variante="c50")
    e["conjunto_completo"] = dict(base, sel="full", variante="full")
    e["blocos_espaciais"] = dict(base, sel="prim", variante="prim", esquema="bloco")
    e["reamostra_eventos"] = dict(base, sel="prim", variante="prim", esquema="dois_estagios")
    e["sem_fronteira"] = dict(base, sel="prim", variante="prim", filtro="sem_borda")
    for ev in EVENTOS_PRIMARIOS:
        e[f"sem_evento_{ev}"] = dict(base, sel=f"sem:{ev}", variante="prim")
    return e


def _chave_hist(filtro):
    return ("he_b", "hc_b", "cont_b") if filtro == "sem_borda" else ("he", "hc", "cont")


def preparar_cenario(ctx, esp, selecoes):
    fonte, alvo = ctx["fonte"], ctx["alvo"]
    ev_s = sorted(set(fonte.evento))
    P = {"est_s": [np.flatnonzero(fonte.evento == e) for e in ev_s], "n_s": fonte.n}
    idx = np.flatnonzero(selecoes[esp["sel"]])
    P["idx"] = idx
    if esp["esquema"] == "bloco":
        blocos = alvo.attr["bloco"][idx]
        nomes = sorted(set(blocos))
        pos = {b: i for i, b in enumerate(nomes)}
        grupo = np.array([pos[b] for b in blocos])
        Ag = np.zeros((len(nomes), idx.size))
        Ag[grupo, np.arange(idx.size)] = 1.0
        ev_agg = np.empty(len(nomes), dtype=object)
        for i, g in enumerate(grupo):
            ev_agg[g] = alvo.evento[idx[i]]
        P["Ag"] = Ag
    else:
        P["Ag"] = None
        ev_agg = alvo.evento[idx].astype(object)
    P["eventos"] = sorted(set(alvo.evento[idx]))
    P["est_t"] = [np.flatnonzero(ev_agg == e) for e in P["eventos"]]
    P["n_t"] = len(ev_agg)
    return P


def _agregar(P, est_alvo, filtro):
    kh, kc, kn = _chave_hist(filtro)
    he = est_alvo[kh][P["idx"]].astype(np.float64)
    hc = est_alvo[kc][P["idx"]].astype(np.float64)
    cn = est_alvo[kn][P["idx"]]
    if P["Ag"] is not None:
        he, hc, cn = P["Ag"] @ he, P["Ag"] @ hc, P["Ag"] @ cn
    return he, hc, cn


def elegiveis(est, qual, minimo=N.MIN_PER_CLASS):
    """Units with at least `minimo` errors and `minimo` correct pixels in every run."""
    ok = None
    for r in est.values():
        e = r[qual]
        este = (e["ne_unid"] >= minimo) & (e["nc_unid"] >= minimo)
        ok = este if ok is None else ok & este
    return ok


def replicas_corrida(P, ctx, r_fonte, r_alvo, cnt_s, cnt_t, filtro, extras, elig_s=None, elig_t=None):
    out = {}
    kh, kc, kn = _chave_hist(filtro)
    out["src_auc"], out["src_augrc"] = R.hist_metrics(cnt_s, r_fonte[kh], r_fonte[kc])
    he, hc, cn = _agregar(P, r_alvo, filtro)
    out["tgt_auc"], out["tgt_augrc"] = R.hist_metrics(cnt_t, he, hc)
    for k, v in R.count_metrics_replicates(cnt_s, r_fonte[kn]).items():
        out["src_" + k] = v
    for k, v in R.count_metrics_replicates(cnt_t, cn).items():
        out["tgt_" + k] = v
    if extras:
        ev_auc, ev_ne, ev_nc = [], [], []
        for idx_e in P["est_t"]:
            ce = cnt_t[:, idx_e]
            a, _ = R.hist_metrics(ce, he[idx_e], hc[idx_e])
            ev_auc.append(a)
            ev_ne.append(ce.astype(np.float64) @ he[idx_e].sum(1))
            ev_nc.append(ce.astype(np.float64) @ hc[idx_e].sum(1))
        ev_auc, ev_ne, ev_nc = np.array(ev_auc), np.array(ev_ne), np.array(ev_nc)
        out["ev_auc"] = ev_auc
        w = ev_ne * ev_nc
        with np.errstate(invalid="ignore", divide="ignore"):
            out["tgt_igual"] = np.nanmean(ev_auc, axis=0)
            out["tgt_pares"] = np.nansum(np.where(np.isnan(ev_auc), 0.0, ev_auc * w), axis=0) / np.where(
                np.isnan(ev_auc), 0.0, w).sum(axis=0)
        for qual, cnt, a_u, el in (("src", cnt_s, r_fonte["auc_unid"], elig_s),
                                   ("tgt", cnt_t, r_alvo["auc_unid"][P["idx"]], elig_t)):
            c = cnt.astype(np.float64) * el[None, :]
            with np.errstate(invalid="ignore", divide="ignore"):
                out[qual + "_macro"] = (c @ np.where(el, a_u, 0.0)) / c.sum(axis=1)
    return out


def rodar_cenario(ctx, est, ens, codigos, reg2cod, esp, selecoes, B, elig_s=None, elig_t=None):
    """Replicates of one scenario. Returns the seed draws and, per code, arrays (K, B) per metric."""
    t0 = time.time()
    P = preparar_cenario(ctx, esp, selecoes)
    if esp["extras"]:
        if P["Ag"] is not None:
            raise ValueError("Extras are only defined without blocks")
        el_t = elig_t[P["idx"]]
        el_s = elig_s
    else:
        el_t = el_s = None
    rng = R.make_rng()
    S = {reg: R.seed_draw(rng, B, N_SEMENTES) for reg in ("R1", "R3")}
    sortear = R.two_stage_counts if esp["esquema"] == "dois_estagios" else R.stratified_counts
    cnt_s = sortear(rng, P["est_s"], P["n_s"], B)
    cnt_t = sortear(rng, P["est_t"], P["n_t"], B)
    if "R2" in reg2cod:
        S["R2"] = R.seed_draw(rng, B, N_SEMENTES)
    S = {reg2cod[reg]: v for reg, v in S.items()}
    Rb = defaultdict(lambda: defaultdict(list))
    for cod in codigos:
        for sem in sorted(k[1] for k in est if k[0] == cod):
            r = est[(cod, sem)]
            rep = replicas_corrida(P, ctx, r["fonte"], r["alvo"], cnt_s, cnt_t, esp["filtro"], esp["extras"],
                                   el_s, el_t)
            for k, v in rep.items():
                Rb[cod][k].append(v)
            if esp["variantes"] and "tta_fonte" in r:
                rep = replicas_corrida(P, ctx, r["tta_fonte"], r["tta_alvo"], cnt_s, cnt_t, esp["filtro"], False)
                for k, v in rep.items():
                    Rb["tta:" + cod][k].append(v)
        if esp["variantes"]:
            rep = replicas_corrida(P, ctx, ens[cod]["fonte"], ens[cod]["alvo"], cnt_s, cnt_t, esp["filtro"], False)
            for k, v in rep.items():
                Rb["ens:" + cod][k].append(v)
    Rb = {g: {k: np.array(v) for k, v in d.items()} for g, d in Rb.items()}
    log(f"  scenario {esp['sel']}/{esp['esquema']}/{esp['filtro']}: {time.time() - t0:.0f}s")
    return {"S": S, "P": P, "Rb": Rb, "B": B}


def media_replicas(Rb_g, chave, S_cod):
    """Seed-averaged replicates of one group. Rb_g[chave]: (K, B) or (K, E, B) with the draws S_cod (B, K)."""
    v = Rb_g[chave]
    if v.shape[0] == 1:                      # ensemble: one pseudo-seed
        return v[0]
    if v.ndim == 2:
        return R.seed_mean(v, S_cod)
    # (K, E, B): average per event
    return np.stack([R.seed_mean(v[:, e, :], S_cod) for e in range(v.shape[1])])


# ----------------------------------------------------------------------------------------------
# Exact point estimates of a scenario
# ----------------------------------------------------------------------------------------------

def sementes_de(est, cod):
    return sorted(k[1] for k in est if k[0] == cod)


def pontos_cenario(est, ens, codigos, esp, ctx, selecoes, elig_s, elig_t):
    """Exact values (seed by seed) of a scenario. Returns {code or 'tta:code' or 'ens:code': {metric: array (K, ...)}}."""
    alvo = ctx["alvo"]
    idx = np.flatnonzero(selecoes[esp["sel"]])
    filtro = esp["filtro"]
    kn = "cont_b" if filtro == "sem_borda" else "cont"
    eventos = sorted(set(alvo.evento[idx]))
    out = {}
    for cod in codigos:
        acc = defaultdict(list)
        for sem in sementes_de(est, cod):
            r = est[(cod, sem)]
            ex_t = r["alvo"]["exato"][(esp["sel"], filtro)]
            ex_s = r["fonte"]["exato"][("todos", filtro)]
            for k in ("auroc", "augrc", "eaurc"):
                acc["tgt_" + {"auroc": "auc", "augrc": "augrc", "eaurc": "eaurc"}[k]].append(ex_t[k])
                acc["src_" + {"auroc": "auc", "augrc": "augrc", "eaurc": "eaurc"}[k]].append(ex_s[k])
            for k, v in N.count_metrics(r["alvo"][kn][idx].sum(0)).items():
                acc["tgt_" + k].append(float(v))
            for k, v in N.count_metrics(r["fonte"][kn].sum(0)).items():
                acc["src_" + k].append(float(v))
            ev = [r["alvo"]["exato"][(f"{esp['variante']}|{e}", filtro)] for e in eventos]
            a = np.array([x["auroc"] for x in ev])
            acc["ev_auc"].append(a)
            if esp["extras"]:
                w = np.array([float(x["ne"]) * float(x["nc"]) for x in ev])
                ok = ~np.isnan(a)
                acc["tgt_igual"].append(float(np.mean(a[ok])))
                acc["tgt_pares"].append(float((a[ok] * w[ok]).sum() / w[ok].sum()))
                au_t = r["alvo"]["auc_unid"][idx][elig_t[idx]]
                au_s = r["fonte"]["auc_unid"][elig_s]
                acc["tgt_macro"].append(float(au_t.mean()) if au_t.size else float("nan"))
                acc["src_macro"].append(float(au_s.mean()) if au_s.size else float("nan"))
        out[cod] = {k: np.array(v) for k, v in acc.items()}
        if esp["variantes"]:
            acc_t = defaultdict(list)
            for sem in sementes_de(est, cod):
                r = est[(cod, sem)]
                if "tta_alvo" in r:
                    et = r["tta_alvo"]["exato"][("prim", "tudo")]
                    es = r["tta_fonte"]["exato"][("todos", "tudo")]
                    acc_t["tgt_auc"].append(et["auroc"]); acc_t["tgt_augrc"].append(et["augrc"])
                    acc_t["tgt_eaurc"].append(et["eaurc"])
                    acc_t["src_auc"].append(es["auroc"]); acc_t["src_augrc"].append(es["augrc"])
                    acc_t["src_eaurc"].append(es["eaurc"])
            if acc_t:
                out["tta:" + cod] = {k: np.array(v) for k, v in acc_t.items()}
            e = ens[cod]
            et, es = e["alvo"]["exato"][("prim", "tudo")], e["fonte"]["exato"][("todos", "tudo")]
            out["ens:" + cod] = {"tgt_auc": np.array([et["auroc"]]), "tgt_augrc": np.array([et["augrc"]]),
                                 "tgt_eaurc": np.array([et["eaurc"]]), "src_auc": np.array([es["auroc"]]),
                                 "src_augrc": np.array([es["augrc"]]), "src_eaurc": np.array([es["eaurc"]])}
    return out


# ----------------------------------------------------------------------------------------------
# Contrasts, categories and summaries
# ----------------------------------------------------------------------------------------------

def intervalos(x):
    return {"ic90": list(R.quantile_interval(x, 0.90)), "ic95": list(R.quantile_interval(x, 0.95)),
            "ep": float(np.nanstd(x, ddof=1)) if np.sum(~np.isnan(x)) > 1 else float("nan"),
            "replicas_nan": int(np.isnan(x).sum())}


def valor(ponto, rep):
    d = {"ponto": float(ponto)}
    d.update(intervalos(rep))
    return d


def contraste(pa, ra, pb, rb, escala=1.0):
    ponto = (pa - pb) * escala
    rep = (np.asarray(ra) - np.asarray(rb)) * escala
    d = {"ponto": float(ponto), "p": R.p_value(rep), "mde": R.mde(rep)}
    d.update(intervalos(rep))
    return d


def contar_sinais(ev_r3, ev_r1):
    """Per-event sign of Delta_reg. ev_*: arrays (K, E) of exact per-event AUROC_e per seed (NaN if undefined).
    An event with an undefined value in any run counts in neither number."""
    definido = ~(np.isnan(ev_r3).any(axis=0) | np.isnan(ev_r1).any(axis=0))
    dif = np.where(definido, ev_r3.mean(axis=0) - ev_r1.mean(axis=0), np.nan)
    return int(np.sum(definido & (dif < 0))), int(np.sum(definido & (dif > 0))), dif


def analisar_cenario(res, pt, codigos, reg2cod, esp, eventos):
    """Estimates, intervals, p-values and categories of one scenario."""
    S, Rb = res["S"], res["Rb"]

    def rep(grupo, chave):
        return media_replicas(Rb[grupo], chave, S[grupo.split(":")[-1]])

    def ponto(grupo, chave):
        return float(np.mean(pt[grupo][chave], axis=0))

    regimes = [r for r in REGIMES if r in reg2cod]
    out = {"agregadores": {}, "eventos": {}}
    agregadores = {"pool": ("tgt_auc", "src_auc")}
    if esp["extras"]:
        agregadores.update({"igual_peso": ("tgt_igual", "src_auc"), "pares_intra_evento": ("tgt_pares", "src_auc"),
                            "macro": ("tgt_macro", "src_macro")})
    n_neg = n_pos = None
    if "R3" in reg2cod and "R1" in reg2cod:
        n_neg, n_pos, dif_ev = contar_sinais(pt[reg2cod["R3"]]["ev_auc"], pt[reg2cod["R1"]]["ev_auc"])
        out["sinais"] = {"n_eventos": len(eventos), "exigido": R.ceil_two_thirds(len(eventos)),
                         "negativos": n_neg, "positivos": n_pos,
                         "dif_por_evento": {e: (None if np.isnan(d) else float(d)) for e, d in zip(eventos, dif_ev)}}
    for nome, (kt, ks) in agregadores.items():
        a = {"auroc": {}, "delta_shift": {}, "delta_reg": {}}
        for reg in regimes:
            c = reg2cod[reg]
            a["auroc"][reg] = {"alvo": valor(ponto(c, kt), rep(c, kt)), "fonte": valor(ponto(c, ks), rep(c, ks))}
            a["delta_shift"][reg] = contraste(ponto(c, kt), rep(c, kt), ponto(c, ks), rep(c, ks))
        if "R3" in reg2cod:
            d = a["delta_shift"]["R3"]
            d["categoria"] = C.category_h1_h2(d["ponto"], d["ic90"][0], d["ic90"][1])
        pares = [("R3", "R1"), ("R2", "R1"), ("R3", "R2")]
        for x, y in pares:
            if x in reg2cod and y in reg2cod:
                cx, cy = reg2cod[x], reg2cod[y]
                a["delta_reg"][f"{x}-{y}"] = contraste(ponto(cx, kt), rep(cx, kt), ponto(cy, kt), rep(cy, kt))
        if "R3" in reg2cod and "R1" in reg2cod:
            c3, c1 = reg2cod["R3"], reg2cod["R1"]
            a["delta_reg_fonte"] = contraste(ponto(c3, ks), rep(c3, ks), ponto(c1, ks), rep(c1, ks))
            s1_ponto = (ponto(c3, kt) - ponto(c1, kt)) - (ponto(c3, ks) - ponto(c1, ks))
            s1_rep = (rep(c3, kt) - rep(c1, kt)) - (rep(c3, ks) - rep(c1, ks))
            a["s1"] = {"ponto": float(s1_ponto), "p": R.p_value(s1_rep), "mde": R.mde(s1_rep)}
            a["s1"].update(intervalos(s1_rep))
            d = a["delta_reg"]["R3-R1"]
            d["categoria"] = C.category_h3(d["ponto"], d["ic90"][0], d["ic90"][1], s1_ponto, n_neg, n_pos,
                                           len(eventos))
        out["agregadores"][nome] = a
    if "R3" in reg2cod and "R1" in reg2cod:
        c3, c1 = reg2cod["R3"], reg2cod["R1"]
        # S2 (unflagged error rate, percentage points) and S3 (AUGRC)
        s2 = contraste(ponto(c3, "tgt_erro_nao_sinalizado"), rep(c3, "tgt_erro_nao_sinalizado"),
                       ponto(c1, "tgt_erro_nao_sinalizado"), rep(c1, "tgt_erro_nao_sinalizado"), 100.0)
        s2["equivalencia_5pp"] = bool(s2["ic90"][0] > -MARGEM_S2_PP and s2["ic90"][1] < MARGEM_S2_PP)
        out["s2"] = s2
        out["s3"] = contraste(ponto(c3, "tgt_augrc"), rep(c3, "tgt_augrc"), ponto(c1, "tgt_augrc"),
                              rep(c1, "tgt_augrc"))
    if esp["extras"]:
        # per-event estimates from the chip bootstrap inside each event
        for j, e in enumerate(eventos):
            d = {"auroc": {}, "delta_shift": {}}
            for reg in regimes:
                c = reg2cod[reg]
                ev_rep = rep(c, "ev_auc")[j]
                ev_pt = float(np.mean(pt[c]["ev_auc"][:, j]))
                d["auroc"][reg] = valor(ev_pt, ev_rep)
                d["delta_shift"][reg] = contraste(ev_pt, ev_rep, ponto(c, "src_auc"), rep(c, "src_auc"))
            if "R3" in reg2cod and "R1" in reg2cod:
                c3, c1 = reg2cod["R3"], reg2cod["R1"]
                d["delta_reg"] = contraste(float(np.mean(pt[c3]["ev_auc"][:, j])), rep(c3, "ev_auc")[j],
                                           float(np.mean(pt[c1]["ev_auc"][:, j])), rep(c1, "ev_auc")[j])
            out["eventos"][e] = d
    return out


# ----------------------------------------------------------------------------------------------
# Leave one chip out, preconditions, descriptive tables
# ----------------------------------------------------------------------------------------------

def deixar_um_de_fora(est, codigos, reg2cod, selecoes):
    """Delta_reg (R3 - R1) with each primary target unit left out in turn, from the histograms."""
    idx = np.flatnonzero(selecoes["prim"])
    valores = {}
    for reg in ("R1", "R3"):
        cod = reg2cod[reg]
        por_semente = []
        for sem in sementes_de(est, cod):
            e = est[(cod, sem)]["alvo"]
            he = e["he"][idx].astype(np.float64)
            hc = e["hc"][idx].astype(np.float64)
            por_semente.append(N.auroc_from_hist(he.sum(0)[None, :] - he, hc.sum(0)[None, :] - hc))
        valores[reg] = np.mean(por_semente, axis=0)
    d = valores["R3"] - valores["R1"]
    return {"minimo": float(d.min()), "maximo": float(d.max()), "mediana": float(np.median(d)),
            "n_unidades": int(idx.size)}


def precondicoes(est, ens, reg2cod, ctx, selecoes, r0):
    out = {}
    # 1. alpha larger than the validation error rate in every run
    taxas = {}
    for (cod, sem), r in est.items():
        c = r["val"]["cont"][0]
        taxas[f"{cod}_s{sem}"] = float(c[1] / c[0])
    out["p1_taxa_erro_validacao"] = {"alfa": float(N.ALPHA), "maxima": max(taxas.values()),
                                      "cumprida": bool(max(taxas.values()) < float(N.ALPHA)), "por_corrida": taxas}
    # 2. R0 inside the range of the R3 seeds on Bolivia
    idx_b = np.flatnonzero(selecoes["prim|Bolivia"])
    cod3 = reg2cod["R3"]
    metricas = defaultdict(list)
    for sem in sementes_de(est, cod3):
        r = est[(cod3, sem)]["alvo"]
        cm = N.count_metrics(r["cont"][idx_b].sum(0))
        metricas["auroc_e"].append(r["exato"][("prim|Bolivia", "tudo")]["auroc"])
        metricas["erro_nao_sinalizado"].append(float(cm["erro_nao_sinalizado"]))
        metricas["taxa_erro"].append(float(cm["taxa_erro"]))
        metricas["miou"].append(float(cm["miou"]))
    p2 = {}
    for k, v in metricas.items():
        ref = r0["bolivia"][k]
        p2[k] = {"r0": ref, "minimo_r3": float(min(v)), "maximo_r3": float(max(v)),
                 "dentro": bool(min(v) <= ref <= max(v))}
    out["p2_r0_na_bolivia"] = p2
    # 3. source-test mIoU of R1 and R3
    miou = {}
    for reg in ("R1", "R3"):
        cod = reg2cod[reg]
        miou[reg] = float(np.mean([float(N.count_metrics(est[(cod, s)]["fonte"]["cont"].sum(0))["miou"])
                                    for s in sementes_de(est, cod)]))
    dif = abs(miou["R1"] - miou["R3"])
    out["p3_miou_fonte"] = {"R1": miou["R1"], "R3": miou["R3"], "diferenca_pp": dif * 100.0,
                            "cumprida": bool(dif <= 0.015)}
    return out


def estatistica_sementes(v):
    v = np.asarray(v, dtype=np.float64)
    return {"media": float(v.mean()), "dp": float(v.std(ddof=1)) if v.size > 1 else None,
            "minimo": float(v.min()), "maximo": float(v.max())}


def tabela_regimes(est, reg2cod, ctx, selecoes):
    """Seed mean, standard deviation across seeds and range of every metric, per regime and set."""
    alvo = ctx["alvo"]
    conjuntos = OrderedDict()
    conjuntos["val"] = ("val", None, ("todos", "tudo"))
    conjuntos["teste"] = ("fonte", None, ("todos", "tudo"))
    conjuntos["alvo"] = ("alvo", selecoes["prim"], ("prim", "tudo"))
    for e in EVENTOS_PRIMARIOS:
        conjuntos[f"alvo|{e}"] = ("alvo", selecoes[f"prim|{e}"], (f"prim|{e}", "tudo"))
    out = {}
    for reg, cod in reg2cod.items():
        out[reg] = {}
        for nome, (qual, mask, chave_ex) in conjuntos.items():
            acc = defaultdict(list)
            for sem in sementes_de(est, cod):
                r = est[(cod, sem)][qual]
                cont = r["cont"][0] if qual == "val" else r["cont"][mask if mask is not None else slice(None)].sum(0)
                for k, v in N.count_metrics(cont).items():
                    acc[k].append(float(v))
                ex = r["exato"][chave_ex]
                for k in ("auroc", "augrc", "eaurc"):
                    acc[k].append(ex[k])
                chave_cal = "todos" if qual in ("val", "fonte") else chave_ex[0]
                if chave_cal in r["calib"]:
                    for fase in ("antes", "depois"):
                        for k, v in r["calib"][chave_cal][fase].items():
                            acc[f"{k}_{fase}"].append(v)
            out[reg][nome] = {k: estatistica_sementes(v) for k, v in acc.items()}
    return out


def rq3_por_unidade(est, reg2cod, ctx, selecoes):
    """Empty-set rate per unit against the water fraction of the unit, and its Spearman correlation."""
    from scipy.stats import spearmanr
    alvo, fonte = ctx["alvo"], ctx["fonte"]
    linhas = []
    resumo = {}
    for nome, uni, idx in (("fonte", fonte, np.arange(fonte.n)), ("alvo", alvo, np.flatnonzero(selecoes["prim"]))):
        frac_agua = (uni.agua[idx] / uni.tamanhos[idx]).astype(float)
        taxas = {}
        for reg, cod in reg2cod.items():
            vals = []
            for sem in sementes_de(est, cod):
                c = est[(cod, sem)][nome]["cont"][idx]
                vals.append(c[:, 3] / c[:, 0])
            taxas[reg] = np.mean(vals, axis=0)
            rho, p = spearmanr(frac_agua, taxas[reg])
            resumo[f"{nome}_{reg}"] = {"spearman": float(rho), "p": float(p), "unidades": int(idx.size)}
        for j, u in enumerate(idx):
            linhas.append([nome, uni.ids[u], uni.evento[u], int(uni.tamanhos[u]), float(frac_agua[j])]
                          + [float(taxas[r][j]) for r in REGIMES if r in reg2cod])
    return linhas, resumo


def linhas_de_base(ctx, selecoes):
    out = {}
    alvo, fonte = ctx["alvo"], ctx["fonte"]
    for nome, uni, mask in (("teste", fonte, np.ones(fonte.n, dtype=bool)), ("alvo", alvo, selecoes["prim"])):
        n = float(uni.tamanhos[mask].sum())
        agua = float(uni.agua[mask].sum())
        out[nome] = {"auroc_e_confianca_constante": 0.5, "sempre_terra_taxa_erro": agua / n,
                     "sempre_terra_miou": (n - agua) / n / 2.0, "fracao_agua": agua / n}
    return out


def tabela_execucoes(est):
    linhas = []
    for (cod, sem), r in sorted(est.items()):
        c = r["val"]["cont"][0]
        cm = N.count_metrics(c)
        linhas.append({"codigo": cod, "semente": sem, "q_hat": r["q_hat"], "tau": r["tau"], "temperatura": r["T"],
                       "val_taxa_erro": float(cm["taxa_erro"]), "val_miou": float(cm["miou"]),
                       "val_auroc_e": r["val"]["exato"][("todos", "tudo")]["auroc"]})
    return linhas


# ----------------------------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------------------------

def para_json(x):
    if isinstance(x, dict):
        return {(k if isinstance(k, str) else "|".join(map(str, k))): para_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [para_json(v) for v in x]
    if isinstance(x, np.ndarray):
        return para_json(x.tolist())
    if isinstance(x, (np.floating, float)):
        return float(x) if math.isfinite(float(x)) else None
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def escrever_csv(caminho, cabecalho, linhas):
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cabecalho)
        w.writerows(linhas)


def fmt(x, casas=6):
    if x is None:
        return ""
    if isinstance(x, (float, np.floating)):
        return "" if not math.isfinite(float(x)) else f"{float(x):.{casas}f}"
    return x


def linha_contraste(nome, d, extra=("", "")):
    return [nome, fmt(d["ponto"]), fmt(d["ic90"][0]), fmt(d["ic90"][1]), fmt(d["ic95"][0]), fmt(d["ic95"][1]),
            fmt(d["p"], 4), extra[0], d.get("categoria", ""), fmt(d.get("mde")), extra[1]]


CAB_CONTRASTE = ["analise", "estimativa", "ic90_inf", "ic90_sup", "ic95_inf", "ic95_sup", "p", "p_holm", "categoria",
                 "mde", "observacao"]


def sha256_arquivo(caminho):
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def escrever_saidas(saida, resultados, meta):
    os.makedirs(os.path.join(saida, "tabelas"), exist_ok=True)
    with open(os.path.join(saida, "resultados.json"), "w", encoding="utf-8") as f:
        json.dump(para_json({"meta": meta, **resultados}), f, indent=1, ensure_ascii=False)
    tab = os.path.join(saida, "tabelas")
    prim = resultados["cenarios"]["primaria"]
    pool = prim["agregadores"]["pool"]
    holm = resultados["holm"]
    linhas = [linha_contraste("H1/H2: Delta_shift R3", pool["delta_shift"]["R3"])]
    for reg in ("R1", "R2"):
        if reg in pool["delta_shift"]:
            linhas.append(linha_contraste(f"Delta_shift {reg} (secundaria)", pool["delta_shift"][reg]))
    linhas.append(linha_contraste("H3: Delta_reg R3-R1", pool["delta_reg"]["R3-R1"],
                                  ("", f"sinais negativos {prim['sinais']['negativos']}, positivos "
                                       f"{prim['sinais']['positivos']} de {prim['sinais']['n_eventos']}")))
    linhas.append(linha_contraste("S1: Delta_reg alvo - Delta_reg fonte", pool["s1"], (fmt(holm["S1"], 4), "")))
    linhas.append(linha_contraste("S2: erro nao sinalizado R3-R1 (pontos percentuais)", prim["s2"],
                                  (fmt(holm["S2"], 4), "equivalencia em +-5 pp: " + ("sim" if prim["s2"]["equivalencia_5pp"] else "nao")
                                   + " (IC90 dentro de +-5 pp; nao ajustada por multiplicidade)")))
    linhas.append(linha_contraste("S3: AUGRC R3-R1", prim["s3"], (fmt(holm["S3"], 4), "")))
    for par in ("R2-R1", "R3-R2"):
        if par in pool["delta_reg"]:
            linhas.append(linha_contraste(f"Delta_reg {par} (secundaria)", pool["delta_reg"][par]))
    escrever_csv(os.path.join(tab, "principal.csv"), CAB_CONTRASTE, linhas)

    linhas = []
    for cen, r in resultados["cenarios"].items():
        for agr, a in r["agregadores"].items():
            ds = a["delta_shift"].get("R3")
            if ds:
                linhas.append(linha_contraste(f"{cen} | {agr} | Delta_shift R3", ds))
            dr = a["delta_reg"].get("R3-R1")
            if dr:
                linhas.append(linha_contraste(f"{cen} | {agr} | Delta_reg R3-R1", dr))
            if "s1" in a:
                linhas.append(linha_contraste(f"{cen} | {agr} | S1", a["s1"]))
    escrever_csv(os.path.join(tab, "sensibilidades.csv"), CAB_CONTRASTE, linhas)

    linhas = []
    for e, d in prim["eventos"].items():
        for reg, v in d["auroc"].items():
            linhas.append([e, f"AUROC_e {reg}", fmt(v["ponto"]), fmt(v["ic90"][0]), fmt(v["ic90"][1]),
                           fmt(v["ic95"][0]), fmt(v["ic95"][1])])
        for reg, v in d["delta_shift"].items():
            linhas.append([e, f"Delta_shift {reg}", fmt(v["ponto"]), fmt(v["ic90"][0]), fmt(v["ic90"][1]),
                           fmt(v["ic95"][0]), fmt(v["ic95"][1])])
        if "delta_reg" in d:
            v = d["delta_reg"]
            linhas.append([e, "Delta_reg R3-R1", fmt(v["ponto"]), fmt(v["ic90"][0]), fmt(v["ic90"][1]),
                           fmt(v["ic95"][0]), fmt(v["ic95"][1])])
    escrever_csv(os.path.join(tab, "eventos.csv"),
                 ["evento", "medida", "estimativa", "ic90_inf", "ic90_sup", "ic95_inf", "ic95_sup"], linhas)

    linhas = []
    for reg, por_conj in resultados["regimes"].items():
        for conj, mets in por_conj.items():
            for k, v in mets.items():
                linhas.append([reg, conj, k, fmt(v["media"]), fmt(v["dp"]), fmt(v["minimo"]), fmt(v["maximo"])])
    escrever_csv(os.path.join(tab, "regimes.csv"),
                 ["regime", "conjunto", "metrica", "media_sementes", "dp_sementes", "minimo", "maximo"], linhas)

    linhas = []
    pc = resultados["precondicoes"]
    linhas.append(["1", "maior taxa de erro na validacao", fmt(pc["p1_taxa_erro_validacao"]["maxima"]),
                   "alfa " + fmt(pc["p1_taxa_erro_validacao"]["alfa"], 2),
                   "sim" if pc["p1_taxa_erro_validacao"]["cumprida"] else "nao"])
    for k, v in pc["p2_r0_na_bolivia"].items():
        linhas.append(["2", f"R0 na Bolivia: {k}", fmt(v["r0"]),
                       f"R3 de {fmt(v['minimo_r3'])} a {fmt(v['maximo_r3'])}", "sim" if v["dentro"] else "nao"])
    p3 = pc["p3_miou_fonte"]
    linhas.append(["3", "mIoU no teste da fonte, R1 e R3", f"{fmt(p3['R1'])} e {fmt(p3['R3'])}",
                   f"diferenca {fmt(p3['diferenca_pp'], 3)} pp (limite 1.5)", "sim" if p3["cumprida"] else "nao"])
    escrever_csv(os.path.join(tab, "precondicoes.csv"), ["numero", "item", "valor", "referencia", "cumprida"], linhas)

    escrever_csv(os.path.join(tab, "execucoes.csv"),
                 ["codigo", "semente", "q_hat", "tau", "temperatura", "val_taxa_erro", "val_miou", "val_auroc_e"],
                 [[r["codigo"], r["semente"]] + [fmt(r[k], 8) for k in
                                                  ("q_hat", "tau", "temperatura", "val_taxa_erro", "val_miou",
                                                   "val_auroc_e")] for r in resultados["execucoes"]])
    escrever_csv(os.path.join(tab, "rq3_por_unidade.csv"),
                 ["conjunto", "unidade", "evento", "pixels", "fracao_agua"] + [f"taxa_vazio_{r}" for r in meta["regimes"]],
                 [[l[0], l[1], l[2], l[3], fmt(l[4])] + [fmt(x) for x in l[5:]] for l in resultados["rq3_linhas"]])


def ler_chave(caminho, codigos):
    """Regime of each code from the key file. Returns {regime: code}."""
    with open(caminho, encoding="utf-8") as f:
        chave = json.load(f)
    if sorted(chave) != sorted(codigos) or not set(chave.values()) <= set(REGIMES):
        raise ValueError("Key file must map exactly the codes found to R1, R2 or R3")
    reg2cod = {}
    for cod, reg in chave.items():
        if reg in reg2cod:
            raise ValueError(f"Regime {reg} assigned to more than one code")
        reg2cod[reg] = cod
    for reg in ("R1", "R3"):
        if reg not in reg2cod:
            raise ValueError(f"Regime {reg} missing from the key file")
    return reg2cod


def carregar_r0():
    caminho = os.path.join(AQUI, "r0_referencia.json")
    with open(caminho, encoding="utf-8") as f:
        return json.load(f)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="repository root (reads splits/)")
    ap.add_argument("--margens", required=True, help="directory with the run and label files")
    ap.add_argument("--chave", required=True, help="JSON file code -> regime")
    ap.add_argument("--saida", required=True, help="output directory")
    ap.add_argument("--B", type=int, default=N.BOOT_B, help="bootstrap replicates (registered: 10000)")
    ap.add_argument("--sem-nuvem050", action="store_true",
                    help="skip the cloud 0.50 sensitivity analysis (windows not cut); recorded in the output")
    ap.add_argument("--sem-tta", action="store_true", help="ignore the margem_tta fields")
    args = ap.parse_args(argv)

    log("analysis start")
    ctx = montar_contexto(args.repo, args.margens, usar_nuvem050=not args.sem_nuvem050)
    corridas = listar_corridas(args.margens)
    codigos = verificar_corridas(corridas)
    log(f"codes: {len(codigos)}; source units {ctx['fonte'].n}, target units {ctx['alvo'].n}")

    bins = N.Bins()
    diag = {}
    for tentativa in range(MAX_REFINOS + 1):
        est, ens, dif, maximo = processar_tudo(ctx, corridas, bins, not args.sem_tta, diag)
        log(f"histogram check: largest difference {dif:.2e} with {bins}; largest |m| {maximo:.2f}")
        if dif <= N.HIST_TOL:
            break
        bins = N.refine_bins(bins, maximo)
        log(f"difference above {N.HIST_TOL}: new bins {bins}")
    else:
        raise RuntimeError(f"Histogram AUROC_e still differs from the exact value after {MAX_REFINOS} refinements "
                           "of the bins; memory doubles with each one, so decide how to proceed")

    reg2cod = ler_chave(args.chave, codigos)
    log("key read: " + ", ".join(f"{r}={reg2cod[r]}" for r in REGIMES if r in reg2cod))

    selecoes = selecoes_alvo(ctx["alvo"], ctx["usar_nuvem050"])
    el_s, el_t = elegiveis(est, "fonte"), elegiveis(est, "alvo")
    resultados = {"cenarios": OrderedDict()}
    for nome, esp in especificacoes(ctx).items():
        log(f"scenario {nome}")
        res = rodar_cenario(ctx, est, ens, codigos, reg2cod, esp, selecoes, args.B, el_s, el_t)
        pt = pontos_cenario(est, ens, codigos, esp, ctx, selecoes, el_s, el_t)
        resultados["cenarios"][nome] = analisar_cenario(res, pt, codigos, reg2cod, esp, res["P"]["eventos"])
        if nome == "primaria":
            resultados["variantes"] = variantes(res, pt, codigos, reg2cod, ens)
        del res
    prim = resultados["cenarios"]["primaria"]
    p = [prim["agregadores"]["pool"]["s1"]["p"], prim["s2"]["p"], prim["s3"]["p"]]
    ajust = R.holm(p)
    resultados["holm"] = {"S1": float(ajust[0]), "S2": float(ajust[1]), "S3": float(ajust[2]),
                          "bruto": {"S1": p[0], "S2": p[1], "S3": p[2]}}
    resultados["deixar_um_de_fora"] = deixar_um_de_fora(est, codigos, reg2cod, selecoes)
    resultados["precondicoes"] = precondicoes(est, ens, reg2cod, ctx, selecoes, carregar_r0())
    resultados["regimes"] = tabela_regimes(est, reg2cod, ctx, selecoes)
    linhas, resumo = rq3_por_unidade(est, reg2cod, ctx, selecoes)
    resultados["rq3_linhas"] = linhas
    resultados["rq3_spearman"] = resumo
    resultados["linhas_de_base"] = linhas_de_base(ctx, selecoes)
    resultados["execucoes"] = tabela_execucoes(est)
    meta = {"B": args.B, "B_registrado": args.B == N.BOOT_B, "bins": repr(bins), "regimes": [r for r in REGIMES if r in reg2cod],
            "chave": reg2cod, "sem_nuvem050": args.sem_nuvem050, "sem_tta": args.sem_tta,
            "scripts_sha256": {f: sha256_arquivo(os.path.join(AQUI, f)) for f in
                               ("analise_congelada.py", "nucleo.py", "reamostragem.py", "categorias.py")},
            "versoes": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
            "histograma_dif_maxima": dif, "histograma_dif_outros": diag.get("dif_outros"),
            "histograma_dif_variantes": diag.get("dif_extra")}
    escrever_saidas(args.saida, resultados, meta)
    log("done")
    return resultados


def variantes(res, pt, codigos, reg2cod, ens):
    """Single-model MSP against the ensemble of the seeds and against test-time augmentation (secondary)."""
    S, Rb = res["S"], res["Rb"]
    out = {}
    for reg, cod in reg2cod.items():
        d = {}
        for rotulo, grupo in (("unico", cod), ("tta", "tta:" + cod), ("conjunto", "ens:" + cod)):
            if grupo not in pt:
                continue
            ponto = {k: float(np.mean(pt[grupo][k])) for k in ("tgt_auc", "tgt_augrc", "tgt_eaurc", "src_auc")}
            reps = {k: media_replicas(Rb[grupo], k, S[cod]) for k in ("tgt_auc", "tgt_augrc", "src_auc")}
            d[rotulo] = {"auroc_e_alvo": valor(ponto["tgt_auc"], reps["tgt_auc"]),
                         "augrc_alvo": valor(ponto["tgt_augrc"], reps["tgt_augrc"]),
                         "eaurc_alvo": ponto["tgt_eaurc"],
                         "auroc_e_fonte": valor(ponto["src_auc"], reps["src_auc"])}
            if rotulo != "unico":
                d[rotulo]["diferenca_auroc_e_alvo_menos_unico"] = contraste(
                    ponto["tgt_auc"], reps["tgt_auc"], float(np.mean(pt[cod]["tgt_auc"])),
                    media_replicas(Rb[cod], "tgt_auc", S[cod]))
        d["variancia_entre_sementes_auroc_e_alvo"] = ens[cod]["var_alvo"][("prim", "tudo")]["auroc"]
        out[reg] = d
    return out


if __name__ == "__main__":
    main()
