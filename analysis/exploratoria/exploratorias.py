#!/usr/bin/env python3
"""Exploratory analyses E1, E2, E4 and E6 of prereg/plano_exploratorio.md.

Nothing here is confirmatory. The frozen scripts in analysis/ are imported read-only, and their SHA-256 is
checked against analysis/README.md before anything else. Outputs go to a folder of their own. The outputs of
the confirmatory run are only read.

Usage
    python analysis/exploratoria/exploratorias.py --repo REPO --margens DIR --chave DIR/chave_codigos.json \
        --confirmatoria DIR --saida OUT

--confirmatoria is the output folder of the single confirmatory run (the one with resultados.json). E1 takes
its numbers from there. It also serves as a check that this script reads the margins as the frozen run did:
tau of every run, the single-model AUROC_e of every event, the unflagged error rate and the pooled ensemble
values must match, or the script stops.

Only the primary target units (Bolivia and the core T2 windows) are used.
Memory: about 4 GB at the peak. Time: about one hour on Colab, CPU only.
"""
import argparse
import json
import os
import platform
import re
import sys

import numpy as np
import scipy
from scipy.special import expit
from scipy.stats import spearmanr

AQUI = os.path.dirname(os.path.abspath(__file__))
ANALISE = os.path.dirname(AQUI)
sys.path.insert(0, ANALISE)
import analise_congelada as AC  # noqa: E402
import nucleo as N  # noqa: E402
import reamostragem as R  # noqa: E402

CONGELADOS = ("analise_congelada.py", "nucleo.py", "reamostragem.py", "categorias.py")
EVENTOS = AC.EVENTOS_PRIMARIOS
B_PADRAO = 10_000
SEMENTE_BOOT = 1             # plan, E4
MIN_UNIDADES_E6 = 10         # plan, E6
TOL = 1e-9
CLASSES = ("falsa_agua", "falsa_terra")
NOME_FIGURA = {"Bolivia": "Bolívia"}
log = AC.log


# ----------------------------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------------------------

def conferir_congelados():
    """SHA-256 of the frozen scripts against the table of analysis/README.md."""
    with open(os.path.join(ANALISE, "README.md"), encoding="utf-8") as f:
        texto = f.read()
    registrados = dict(re.findall(r"\|\s*`([\w.]+\.py)`\s*\|\s*`([0-9a-f]{64})`\s*\|", texto))
    for nome in CONGELADOS:
        if nome not in registrados:
            raise ValueError(f"{nome} has no SHA-256 in analysis/README.md")
        if AC.sha256_arquivo(os.path.join(ANALISE, nome)) != registrados[nome]:
            raise ValueError(f"{nome}: SHA-256 differs from analysis/README.md; the frozen scripts were changed")
    return {n: registrados[n] for n in CONGELADOS}


def igual(a, b, o_que):
    if not (abs(float(a) - float(b)) <= TOL):
        raise ValueError(f"Check failed, {o_que}: {a!r} here against {b!r} in the confirmatory run")


# ----------------------------------------------------------------------------------------------
# Units
# ----------------------------------------------------------------------------------------------

def universo_primario(alvo):
    """Universe of the primary target units only, and the pixel mask that selects them from the target."""
    idx = np.flatnonzero(alvo.attr["prim"])
    px = alvo.attr["prim"][alvo.unit_id]
    uni = AC.Universo([alvo.ids[i] for i in idx], alvo.evento[idx], alvo.tamanhos[idx], alvo.rotulo[px], None)
    return uni, px


def contagem_classe(m, y, tau, unit_id, n_units):
    """Per unit: false water, false land, and each of them with MSP >= tau (unflagged). Shape (n_units, 4).

    MSP and the flag are as in nucleo.unit_counts."""
    y = np.asarray(y).astype(bool)
    pred = m > 0
    nao_sinalizado = expit(np.abs(m).astype(np.float64)) >= tau
    fa = pred & ~y
    ft = ~pred & y
    unit_id = np.asarray(unit_id, dtype=np.int64)
    out = np.zeros((n_units, 4), dtype=np.float64)
    for j, mask in enumerate((fa, ft, fa & nao_sinalizado, ft & nao_sinalizado)):
        out[:, j] = np.bincount(unit_id[mask], minlength=n_units)
    return out


def metricas_classe(c):
    """From summed counts (fa, ft, fa unflagged, ft unflagged): per class, share of all errors, share of the
    unflagged errors, and P(MSP >= tau | error of that class). Also the overall unflagged error rate."""
    fa, ft, fa_n, ft_n = (float(x) for x in c)
    err, err_n = fa + ft, fa_n + ft_n

    def div(a, b):
        return a / b if b > 0 else float("nan")

    return {"falsa_agua": {"parcela_erros": div(fa, err), "parcela_nao_sinalizados": div(fa_n, err_n),
                           "erro_nao_sinalizado": div(fa_n, fa)},
            "falsa_terra": {"parcela_erros": div(ft, err), "parcela_nao_sinalizados": div(ft_n, err_n),
                            "erro_nao_sinalizado": div(ft_n, ft)},
            "total": div(err_n, err)}


# ----------------------------------------------------------------------------------------------
# One run
# ----------------------------------------------------------------------------------------------

def processar_corrida(arquivos, ctx, uni, px):
    rv = ctx["rot"]["val"]
    m_v = AC.ler_margens(arquivos["val"], rv)
    _, tau = N.conformal_threshold(m_v, rv["rotulo"])
    out = {"tau": tau, "classe": {}}
    out["classe"]["val"] = contagem_classe(m_v, rv["rotulo"], tau, np.zeros(m_v.size, dtype=np.int64), 1)[0]
    del m_v
    fonte = ctx["fonte"]
    m_f = AC.margens_universo(arquivos, ctx, "fonte")
    out["classe"]["teste"] = contagem_classe(m_f, fonte.rotulo, tau, fonte.unit_id, fonte.n).sum(0)
    del m_f
    m = AC.margens_universo(arquivos, ctx, "alvo")[px]
    y = uni.rotulo.astype(bool)
    out["classe_unid"] = contagem_classe(m, y, tau, uni.unit_id, uni.n)
    a = np.abs(m)
    err = (m > 0) != y
    del m
    ordem = AC.OrdemGlobal(a, err, uni.unit_id, None)
    out["auroc_pool"] = ordem.metricas(np.ones(uni.n, dtype=bool))["auroc"]
    out["auroc_evento"] = {e: ordem.metricas(uni.evento == e)["auroc"] for e in EVENTOS}
    del ordem
    out["auc_unid"], out["ne_unid"], out["nc_unid"] = AC.unit_exact_auroc(a, err, uni.inicio)
    return out


# ----------------------------------------------------------------------------------------------
# E4: ensemble of one regime
# ----------------------------------------------------------------------------------------------

def processar_conjunto(corridas_cod, ctx, uni, px, bins):
    pilha = [AC.margens_universo(corridas_cod[s], ctx, "alvo")[px] for s in sorted(corridas_cod)]
    m_ens, a_var = AC.margem_conjunto(pilha)
    del pilha
    y = uni.rotulo.astype(bool)
    err = (m_ens > 0) != y
    a = np.abs(m_ens)
    del m_ens
    out = {}
    for nome, escore in (("conjunto", a), ("variancia", a_var)):
        ordem = AC.OrdemGlobal(escore, err, uni.unit_id, None)
        out[nome] = {"pool": ordem.metricas(np.ones(uni.n, dtype=bool))["auroc"]}
        out[nome].update({e: ordem.metricas(uni.evento == e)["auroc"] for e in EVENTOS})
        del ordem
    out["he"], out["hc"] = N.unit_histograms(uni.unit_id, a, err, uni.n, bins)
    out["dif_hist"] = abs(float(N.auroc_from_hist(out["he"].sum(0, dtype=np.int64),
                                                  out["hc"].sum(0, dtype=np.int64))) - out["conjunto"]["pool"])
    return out


def bootstrap_r3_menos_r1(uni, ens, reg2cod, B):
    """Ensemble R3 minus ensemble R1, pooled and per event: units resampled within events, seeds not."""
    estratos = [np.flatnonzero(uni.evento == e) for e in EVENTOS]
    cnt = R.stratified_counts(np.random.default_rng(SEMENTE_BOOT), estratos, uni.n, B)
    rep = {}
    for reg in ("R1", "R3"):
        e = ens[reg2cod[reg]]
        rep[reg] = {"pool": R.hist_metrics(cnt, e["he"], e["hc"])[0]}
        for ev, idx in zip(EVENTOS, estratos):
            rep[reg][ev] = R.hist_metrics(cnt[:, idx], e["he"][idx], e["hc"][idx])[0]
    out = {}
    for chave in ("pool",) + EVENTOS:
        ponto = ens[reg2cod["R3"]]["conjunto"][chave] - ens[reg2cod["R1"]]["conjunto"][chave]
        x = rep["R3"][chave] - rep["R1"][chave]
        out[chave] = {"ponto": float(ponto), "ic90": list(R.quantile_interval(x, 0.90)),
                      "replicas_nan": int(np.isnan(x).sum())}
    return out


# ----------------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------------

def _virgula(x, _=None):
    return f"{x:g}".replace(".", ",").replace("-", "−")


def figura_e1(linhas, caminho):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    nomes = [l["evento"] for l in linhas]
    ys = np.arange(len(linhas))[::-1]
    fig, eixos = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)
    for eixo, (chave, titulo) in zip(eixos, (("delta_shift_r3", "Δ_shift (R3)"),
                                             ("delta_reg", "Δ_reg (R3 − R1)"))):
        for y, l in zip(ys, linhas):
            p, lo, hi = l[chave], l[chave + "_ic90_inf"], l[chave + "_ic90_sup"]
            cor = "#1f1f1f" if l["evento"] == "Agregado" else "#4a6fa5"
            eixo.errorbar(p, y, xerr=[[p - lo], [hi - p]], fmt="D" if l["evento"] == "Agregado" else "o",
                          color=cor, ecolor=cor, capsize=3, ms=5, lw=1.2)
        eixo.axvline(0.0, color="#888888", lw=0.8)
        for d in (-N.DELTA, N.DELTA):
            eixo.axvline(d, color="#bbbbbb", lw=0.8, ls="--")
        eixo.set_title(titulo, fontsize=10)
        eixo.xaxis.set_major_formatter(FuncFormatter(_virgula))
        eixo.grid(axis="x", color="#eeeeee", lw=0.6)
        eixo.set_axisbelow(True)
    rotulos = [f"{NOME_FIGURA.get(n, n)}  ({_virgula(round(100 * l['parcela_pixels'], 1))}%)" if n != "Agregado"
               else n for n, l in zip(nomes, linhas)]
    eixos[0].set_yticks(ys)
    eixos[0].set_yticklabels(rotulos, fontsize=9)
    fig.text(0.5, 0.01, "Estimativa e IC de 90%. Linhas tracejadas: ±δ = ±0,02. "
             "Entre parênteses: parcela dos pixels do alvo.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(caminho, dpi=200)
    plt.close(fig)


def figura_e6(uni, dreg, elegivel, fracao, resumo, caminho):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    eventos = [e for e in EVENTOS if resumo[e]["spearman"] is not None]
    if not eventos:
        return False
    fig, eixos = plt.subplots(1, len(eventos), figsize=(3.4 * len(eventos), 3.2), squeeze=False)
    for eixo, e in zip(eixos[0], eventos):
        sel = elegivel & (uni.evento == e)
        eixo.scatter(fracao[sel], dreg[sel], s=12, color="#4a6fa5", alpha=0.8, lw=0)
        eixo.axhline(0.0, color="#888888", lw=0.8)
        rho = resumo[e]["spearman"]
        eixo.set_title(f"{NOME_FIGURA.get(e, e)}: n = {resumo[e]['unidades']}, ρ = {_virgula(round(rho, 2))}", fontsize=9)
        eixo.set_xlabel("Fração de água da unidade", fontsize=8)
        eixo.xaxis.set_major_formatter(FuncFormatter(_virgula))
        eixo.yaxis.set_major_formatter(FuncFormatter(_virgula))
        eixo.tick_params(labelsize=8)
    eixos[0][0].set_ylabel("Δ_reg da unidade (R3 − R1)", fontsize=8)
    fig.tight_layout()
    fig.savefig(caminho, dpi=200)
    plt.close(fig)
    return True


# ----------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="repository root (reads splits/)")
    ap.add_argument("--margens", required=True, help="directory with the run and label files")
    ap.add_argument("--chave", required=True, help="JSON file code -> regime")
    ap.add_argument("--confirmatoria", required=True, help="output folder of the confirmatory run")
    ap.add_argument("--saida", required=True, help="output folder of these analyses (must be new or empty)")
    ap.add_argument("--B", type=int, default=B_PADRAO, help="bootstrap replicates of E4 (plan: 10000)")
    args = ap.parse_args(argv)

    log("exploratory analyses start")
    shas = conferir_congelados()
    log("frozen scripts: SHA-256 as in analysis/README.md")
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
    uni, px = universo_primario(ctx["alvo"])
    log(f"primary target units: {uni.n}; pixels {int(uni.tamanhos.sum())}")

    # runs
    est = {}
    for reg in regimes:
        cod = reg2cod[reg]
        for sem in sorted(corridas[cod]):
            est[(reg, sem)] = processar_corrida(corridas[cod][sem], ctx, uni, px)
            log(f"  run {reg} s{sem}: tau {est[(reg, sem)]['tau']:.6f}")

    # checks against the confirmatory run
    taus = {(e["codigo"], int(e["semente"])): e["tau"] for e in res_c["execucoes"]}
    ev_c = res_c["cenarios"]["primaria"]["eventos"]
    for reg in regimes:
        cod = reg2cod[reg]
        sementes = sorted(corridas[cod])
        for sem in sementes:
            igual(est[(reg, sem)]["tau"], taus[(cod, sem)], f"tau of {reg} s{sem}")
        for e in EVENTOS:
            igual(np.mean([est[(reg, s)]["auroc_evento"][e] for s in sementes]), ev_c[e]["auroc"][reg]["ponto"],
                  f"AUROC_e of {reg} on {e}")
        for conj, contagem in (("val", lambda r: r["classe"]["val"]), ("teste", lambda r: r["classe"]["teste"]),
                               ("alvo", lambda r: r["classe_unid"].sum(0))):
            v = np.mean([metricas_classe(contagem(est[(reg, s)]))["total"] for s in sementes])
            igual(v, res_c["regimes"][reg][conj]["erro_nao_sinalizado"]["media"], f"unflagged error of {reg} on {conj}")
    log("checks against the confirmatory run: tau, AUROC_e per event and unflagged error rate match")

    # E4
    bins = N.Bins()
    ens = {}
    for reg in regimes:
        cod = reg2cod[reg]
        ens[cod] = processar_conjunto(corridas[cod], ctx, uni, px, bins)
        var_c = res_c["variantes"][reg]
        igual(ens[cod]["conjunto"]["pool"], var_c["conjunto"]["auroc_e_alvo"]["ponto"], f"ensemble AUROC_e of {reg}")
        igual(ens[cod]["variancia"]["pool"], var_c["variancia_entre_sementes_auroc_e_alvo"],
              f"seed-variance AUROC_e of {reg}")
        log(f"  ensemble {reg}: AUROC_e {ens[cod]['conjunto']['pool']:.6f}; histogram difference "
            f"{ens[cod]['dif_hist']:.1e}")
    log("checks against the confirmatory run: pooled ensemble values match")
    boot = bootstrap_r3_menos_r1(uni, ens, reg2cod, args.B)

    os.makedirs(os.path.join(saida, "tabelas"), exist_ok=True)
    os.makedirs(os.path.join(saida, "figuras"), exist_ok=True)
    tab, fig = os.path.join(saida, "tabelas"), os.path.join(saida, "figuras")
    fmt = AC.fmt
    saida_json = {}

    # E1
    pix = {e: int(uni.tamanhos[uni.evento == e].sum()) for e in EVENTOS}
    total = sum(pix.values())
    pool_c = res_c["cenarios"]["primaria"]["agregadores"]["pool"]
    linhas_e1 = []
    for e in EVENTOS + ("Agregado",):
        ds = pool_c["delta_shift"]["R3"] if e == "Agregado" else ev_c[e]["delta_shift"]["R3"]
        dr = pool_c["delta_reg"]["R3-R1"] if e == "Agregado" else ev_c[e]["delta_reg"]
        linhas_e1.append({"evento": e, "parcela_pixels": 1.0 if e == "Agregado" else pix[e] / total,
                          "delta_shift_r3": ds["ponto"], "delta_shift_r3_ic90_inf": ds["ic90"][0],
                          "delta_shift_r3_ic90_sup": ds["ic90"][1], "delta_reg": dr["ponto"],
                          "delta_reg_ic90_inf": dr["ic90"][0], "delta_reg_ic90_sup": dr["ic90"][1]})
    cab = list(linhas_e1[0].keys())
    AC.escrever_csv(os.path.join(tab, "e1_eventos.csv"), cab, [[fmt(l[k]) for k in cab] for l in linhas_e1])
    figura_e1(linhas_e1, os.path.join(fig, "e1_floresta.png"))
    saida_json["e1"] = linhas_e1

    # E2
    linhas_e2, e2 = [], {}
    conjuntos = [("val", None), ("teste", None), ("alvo", None)] + [(f"alvo|{e}", e) for e in EVENTOS]
    for reg in regimes:
        sementes = sorted(s for r, s in est if r == reg)
        e2[reg] = {}
        for conj, ev in conjuntos:
            ms = []
            for s in sementes:
                r = est[(reg, s)]
                if conj in ("val", "teste"):
                    c = r["classe"][conj]
                else:
                    sel = np.ones(uni.n, dtype=bool) if ev is None else uni.evento == ev
                    c = r["classe_unid"][sel].sum(0)
                ms.append(metricas_classe(c))
            e2[reg][conj] = {}
            for cl in CLASSES:
                d = {k: AC.estatistica_sementes([m[cl][k] for m in ms])
                     for k in ("parcela_erros", "parcela_nao_sinalizados", "erro_nao_sinalizado")}
                e2[reg][conj][cl] = d
                linhas_e2.append([reg, conj, cl] + [fmt(d[k][q]) for k in ("parcela_erros", "parcela_nao_sinalizados",
                                                                       "erro_nao_sinalizado") for q in ("media", "dp")])
    AC.escrever_csv(os.path.join(tab, "e2_erros_por_classe.csv"),
                    ["regime", "conjunto", "classe", "parcela_erros_media", "parcela_erros_dp",
                     "parcela_nao_sinalizados_media", "parcela_nao_sinalizados_dp", "erro_nao_sinalizado_media",
                     "erro_nao_sinalizado_dp"], linhas_e2)
    saida_json["e2"] = e2

    # E4
    linhas_e4, e4 = [], {}
    for reg in regimes:
        cod = reg2cod[reg]
        sementes = sorted(s for r, s in est if r == reg)
        e4[reg] = {}
        for chave in ("pool",) + EVENTOS:
            unico = float(np.mean([est[(reg, s)]["auroc_pool"] if chave == "pool" else
                                   est[(reg, s)]["auroc_evento"][chave] for s in sementes]))
            d = {"unico": unico, "conjunto": ens[cod]["conjunto"][chave],
                 "conjunto_menos_unico": ens[cod]["conjunto"][chave] - unico,
                 "variancia_entre_sementes": ens[cod]["variancia"][chave]}
            e4[reg][chave] = d
            linhas_e4.append([reg, "Agregado" if chave == "pool" else chave] +
                             [fmt(d[k]) for k in ("unico", "conjunto", "conjunto_menos_unico", "variancia_entre_sementes")])
    AC.escrever_csv(os.path.join(tab, "e4_conjunto.csv"),
                    ["regime", "evento", "auroc_e_modelo_unico", "auroc_e_conjunto", "conjunto_menos_unico",
                     "auroc_e_variancia_entre_sementes"], linhas_e4)
    AC.escrever_csv(os.path.join(tab, "e4_conjunto_r3_menos_r1.csv"),
                    ["evento", "estimativa", "ic90_inf", "ic90_sup", "replicas_nan"],
                    [["Agregado" if k == "pool" else k, fmt(v["ponto"]), fmt(v["ic90"][0]), fmt(v["ic90"][1]),
                      v["replicas_nan"]] for k, v in boot.items()])
    saida_json["e4"] = {"por_regime": e4, "conjunto_r3_menos_r1": boot,
                        "dif_hist": {reg: ens[reg2cod[reg]]["dif_hist"] for reg in regimes}}

    # E6
    fracao = uni.agua / uni.tamanhos
    elegivel = np.ones(uni.n, dtype=bool)
    auc = {}
    for reg in ("R1", "R3"):
        sementes = sorted(s for r, s in est if r == reg)
        for s in sementes:
            r = est[(reg, s)]
            elegivel &= (r["ne_unid"] >= N.MIN_PER_CLASS) & (r["nc_unid"] >= N.MIN_PER_CLASS)
        auc[reg] = np.mean([est[(reg, s)]["auc_unid"] for s in sementes], axis=0)
    dreg = np.where(elegivel, auc["R3"] - auc["R1"], np.nan)
    resumo = {}
    for e in EVENTOS:
        sel = elegivel & (uni.evento == e)
        n = int(sel.sum())
        rho = float(spearmanr(fracao[sel], dreg[sel])[0]) if n >= MIN_UNIDADES_E6 else None
        resumo[e] = {"unidades": n, "spearman": rho}
    AC.escrever_csv(os.path.join(tab, "e6_unidades.csv"),
                    ["unidade", "evento", "pixels", "fracao_agua", "elegivel", "auroc_e_R1", "auroc_e_R3", "delta_reg"],
                    [[uni.ids[u], uni.evento[u], int(uni.tamanhos[u]), fmt(float(fracao[u])), bool(elegivel[u]),
                      fmt(float(auc["R1"][u])) if elegivel[u] else "", fmt(float(auc["R3"][u])) if elegivel[u] else "",
                      fmt(float(dreg[u])) if elegivel[u] else ""] for u in range(uni.n)])
    AC.escrever_csv(os.path.join(tab, "e6_resumo.csv"), ["evento", "unidades_elegiveis", "spearman"],
                    [[e, d["unidades"], fmt(d["spearman"], 3) if d["spearman"] is not None else ""]
                     for e, d in resumo.items()])
    figura_e6(uni, dreg, elegivel, fracao, resumo, os.path.join(fig, "e6_dispersao.png"))
    saida_json["e6"] = resumo

    meta = {"plano": "prereg/plano_exploratorio.md", "B": args.B, "semente_bootstrap": SEMENTE_BOOT,
            "congelados_sha256": shas,
            "exploratorias_sha256": AC.sha256_arquivo(os.path.abspath(__file__)),
            "chave": reg2cod, "unidades_alvo": uni.n,
            "versoes": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
            "conferencias": "tau, AUROC_e por evento, erro nao sinalizado e valores agregados do conjunto iguais "
                            "aos da rodada confirmatoria (tolerancia 1e-9)"}
    with open(os.path.join(saida, "exploratorias.json"), "w", encoding="utf-8") as f:
        json.dump(AC.para_json({"meta": meta, **saida_json}), f, indent=1, ensure_ascii=False)
    log("done")
    return saida_json


if __name__ == "__main__":
    main()
