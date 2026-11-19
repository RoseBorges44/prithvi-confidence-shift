#!/usr/bin/env python3
"""Exploratory analysis E8 of prereg/plano_exploratorio_2.md: mixed logistic model of error pixels.

Among the error pixels of the source test and of the pooled target, the outcome is 1 when the error is unflagged
(MSP >= tau of the run). Fixed effects: regime, domain and their interaction, error class (false water, false land),
distance band to the label boundary (from the distance files of E3) and the standardized water fraction of the unit.
Random intercepts: unit, event (units are nested in events because unit names are unique) and run.

    python analysis/exploratoria/e8_misto.py --repo REPO --margens DIR --chave KEY --confirmatoria DIR \
        --distancias DIR_DIST_E3 --saida OUT

Nothing here is confirmatory. The frozen scripts are imported read-only and their SHA-256 is checked first.
"""
import argparse
import json
import os
import platform
import sys

import numpy as np
import scipy
from scipy import sparse
from scipy.special import expit

AQUI = os.path.dirname(os.path.abspath(__file__))
ANALISE = os.path.dirname(AQUI)
sys.path.insert(0, ANALISE)
sys.path.insert(0, AQUI)
import analise_congelada as AC  # noqa: E402
import e3_borda as E3  # noqa: E402
import exploratorias as X  # noqa: E402
import nucleo as N  # noqa: E402

MAX_POR_UNIDADE = 200        # plan
SEMENTE_AMOSTRA = 2          # plan
SEMENTE_VB = 4               # plan amendment of 2026-11-19: starting values of fit_vb
FAIXAS = ("ate2", "3a10", "mais10")
DOMINIOS = ("teste", "alvo")
CLASSES = ("falsa_agua", "falsa_terra")
log = AC.log


# ----------------------------------------------------------------------------------------------
# Rows
# ----------------------------------------------------------------------------------------------

def faixa3(d):
    """Band of the plan from the Chebyshev distance of E3: 0 (d <= 2), 1 (3 to 10), 2 (more than 10 or no other class)."""
    k = np.full(d.shape, 2, dtype=np.int8)
    k[(d >= 1) & (d <= 2)] = 0
    k[(d >= 3) & (d <= 10)] = 1
    return k


def amostrar(rng, m, y, tau, unit_id, inicio, faixa, max_por_unidade=MAX_POR_UNIDADE):
    """Error pixels of one run and set, at most `max_por_unidade` per unit, drawn without replacement in unit order.

    Returns a dict of arrays: unit, unflagged, class (0 false water, 1 false land), band, weight."""
    pred = m > 0
    y = np.asarray(y).astype(bool)
    erro = pred != y
    out = {k: [] for k in ("unidade", "nao_sinalizado", "classe", "faixa", "peso")}
    for u in range(len(inicio) - 1):
        a, b = inicio[u], inicio[u + 1]
        idx = a + np.flatnonzero(erro[a:b])
        if idx.size == 0:
            continue
        k = idx.size
        if k > max_por_unidade:
            idx = np.sort(rng.choice(idx, size=max_por_unidade, replace=False))
        out["unidade"].append(np.full(idx.size, u, dtype=np.int32))
        out["nao_sinalizado"].append((expit(np.abs(m[idx]).astype(np.float64)) >= tau).astype(np.int8))
        out["classe"].append((~pred[idx]).astype(np.int8))
        out["faixa"].append(faixa[idx])
        out["peso"].append(np.full(idx.size, k / idx.size))
    return {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in out.items()}


# ----------------------------------------------------------------------------------------------
# Design and fits
# ----------------------------------------------------------------------------------------------

def matriz_fixa(dados, termos):
    """Dense design of the fixed effects with treatment coding. Returns (matrix, names)."""
    cols, nomes = [np.ones(len(dados["y"]))], ["intercepto"]
    for termo in termos:
        if termo == "agua_z":
            cols.append(dados["agua_z"])
            nomes.append("agua_z")
            continue
        if ":" in termo:
            a, b = termo.split(":")
            for va in dados["niveis"][a][1:]:
                for vb in dados["niveis"][b][1:]:
                    cols.append(((dados[a] == va) & (dados[b] == vb)).astype(np.float64))
                    nomes.append(f"{a}[{va}]:{b}[{vb}]")
            continue
        for v in dados["niveis"][termo][1:]:
            cols.append((dados[termo] == v).astype(np.float64))
            nomes.append(f"{termo}[{v}]")
    return np.column_stack(cols), nomes


def matriz_aleatoria(dados, efeitos):
    """Sparse indicators of the random intercepts and the ident vector of BinomialBayesMixedGLM."""
    blocos, ident, nomes = [], [], []
    for j, ef in enumerate(efeitos):
        codigos, niveis = np.unique(dados[ef], return_inverse=True)
        n = len(dados["y"])
        blocos.append(sparse.csr_matrix((np.ones(n), (np.arange(n), niveis)), shape=(n, len(codigos))))
        ident.append(np.full(len(codigos), j, dtype=np.int64))
        nomes.append(ef)
    return sparse.hstack(blocos).tocsr(), np.concatenate(ident), nomes


def ajustar_misto(dados, termos, efeitos):
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    exog, nomes = matriz_fixa(dados, termos)
    exog_vc, ident, nomes_vc = matriz_aleatoria(dados, efeitos)
    modelo = BinomialBayesMixedGLM(dados["y"].astype(np.float64), exog, exog_vc, ident,
                                   fep_names=nomes, vcp_names=nomes_vc)
    # Plan amendment of 2026-11-19: fixed seed for the starting values, and a second pass started
    # from the first solution; the second pass is the reported fit.
    passadas = []
    r = modelo.fit_vb(rng=np.random.default_rng(SEMENTE_VB))
    passadas.append(r)
    r = modelo.fit_vb(mean=r.params.copy(), sd=np.sqrt(r._cov_params), rng=np.random.default_rng(SEMENTE_VB))
    passadas.append(r)
    convergencia = [{"sucesso": bool(p.optim_retvals.success), "mensagem": str(p.optim_retvals.message),
                     "iteracoes": int(p.optim_retvals.nit),
                     "gradiente_max": float(np.max(np.abs(p.optim_retvals.jac)))} for p in passadas]
    mudanca = float(np.max(np.abs(passadas[1].fe_mean - passadas[0].fe_mean)))
    log(f"  VB pass 1: {convergencia[0]['mensagem']} ({convergencia[0]['iteracoes']} iterations); "
        f"pass 2: {convergencia[1]['mensagem']} ({convergencia[1]['iteracoes']} iterations); "
        f"largest change of a fixed effect between passes {mudanca:.2e} (log odds)")
    efeitos_fixos = [{"termo": n, "estimativa": float(b), "erro": float(s), "or": float(np.exp(b)),
                      "ic95": [float(np.exp(b - 1.96 * s)), float(np.exp(b + 1.96 * s))]}
                     for n, b, s in zip(nomes, r.fe_mean, r.fe_sd)]
    dp = {n: float(np.exp(v)) for n, v in zip(nomes_vc, r.vcp_mean)}
    return {"metodo": "BinomialBayesMixedGLM.fit_vb, duas passadas", "fixos": efeitos_fixos, "dp_aleatorios": dp,
            "linhas": int(len(dados["y"])), "convergencia": convergencia,
            "mudanca_max_fixos_entre_passadas": mudanca}


def ajustar_gee(dados, termos, pesos=False):
    import statsmodels.api as sm
    exog, nomes = matriz_fixa(dados, termos)
    grupos = np.unique(dados["unidade_global"], return_inverse=True)[1]
    ordem = np.argsort(grupos, kind="stable")
    if pesos:
        r = sm.GEE(dados["y"][ordem].astype(np.float64), exog[ordem], groups=grupos[ordem],
                   family=sm.families.Binomial(), cov_struct=sm.cov_struct.Independence(),
                   weights=dados["peso"][ordem]).fit()
        metodo = "GEE logistico com os pesos das linhas, correlacao independente, erros robustos por unidade"
    else:
        r = sm.GEE(dados["y"][ordem].astype(np.float64), exog[ordem], groups=grupos[ordem],
                   family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
        metodo = "GEE logistico, correlacao permutavel dentro da unidade"
    efeitos_fixos = [{"termo": n, "estimativa": float(b), "erro": float(s), "or": float(np.exp(b)),
                      "ic95": [float(np.exp(b - 1.96 * s)), float(np.exp(b + 1.96 * s))]}
                     for n, b, s in zip(nomes, r.params, r.bse)]
    return {"metodo": metodo, "fixos": efeitos_fixos, "linhas": int(len(dados["y"]))}


def subconjunto(dados, sel):
    out = {k: (v[sel] if isinstance(v, np.ndarray) and v.shape[:1] == dados["y"].shape else v)
           for k, v in dados.items()}
    return out


# ----------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------

def montar_dados(args, res_c, reg2cod):
    ctx = AC.montar_contexto(args.repo, args.margens, usar_nuvem050=not res_c["meta"]["sem_nuvem050"])
    corridas = AC.listar_corridas(args.margens)
    codigos = AC.verificar_corridas(corridas)
    if AC.ler_chave(args.chave, codigos) != reg2cod:
        raise ValueError("The key differs from the one recorded by the confirmatory run")
    rot = ctx["rot"]
    dist = {c: E3.ler_distancias(args.distancias, c, rot[c]) for c in ("teste", "bolivia", "t2")}
    log("distance files of E3 match the label files and the boundary flags")
    uni, px = X.universo_primario(ctx["alvo"])
    fonte = ctx["fonte"]
    faixa = {"teste": faixa3(dist["teste"]), "alvo": faixa3(np.concatenate([dist["bolivia"], dist["t2"]])[px])}
    universos = {"teste": fonte, "alvo": uni}

    # standardized water fraction over the units of both domains
    fr = {d: u.agua / u.tamanhos for d, u in universos.items()}
    todas = np.concatenate([fr["teste"], fr["alvo"]])
    media, dp = float(todas.mean()), float(todas.std(ddof=1))

    taus = {(e["codigo"], int(e["semente"])): e["tau"] for e in res_c["execucoes"]}
    rng = np.random.default_rng(SEMENTE_AMOSTRA)
    partes = []
    for reg in AC.REGIMES:
        cod = reg2cod[reg]
        for sem in sorted(corridas[cod]):
            arq = corridas[cod][sem]
            m_v = AC.ler_margens(arq["val"], rot["val"])
            _, tau = N.conformal_threshold(m_v, rot["val"]["rotulo"])
            del m_v
            X.igual(tau, taus[(cod, sem)], f"tau of {reg} s{sem}")
            for dom in DOMINIOS:
                u_obj = universos[dom]
                m = AC.margens_universo(arq, ctx, "fonte") if dom == "teste" else AC.margens_universo(arq, ctx, "alvo")[px]
                a = amostrar(rng, m, u_obj.rotulo, tau, u_obj.unit_id, u_obj.inicio, faixa[dom])
                del m
                n = a["unidade"].size
                partes.append({"y": a["nao_sinalizado"], "regime": np.full(n, reg), "dominio": np.full(n, dom),
                               "classe": np.array(CLASSES)[a["classe"]], "faixa": np.array(FAIXAS)[a["faixa"]],
                               "agua_z": (fr[dom][a["unidade"]] - media) / dp,
                               "evento": np.array([f"{dom}|{e}" for e in u_obj.evento])[a["unidade"]],
                               "unidade_global": np.array([f"{dom}|{i}" for i in u_obj.ids])[a["unidade"]],
                               "execucao": np.full(n, f"{reg}|{sem}"), "peso": a["peso"]})
            log(f"  run {reg} s{sem}: rows drawn")
    dados = {k: np.concatenate([p[k] for p in partes]) for k in partes[0]}
    dados["niveis"] = {"regime": list(AC.REGIMES), "dominio": list(DOMINIOS), "classe": list(CLASSES),
                       "faixa": list(FAIXAS),
                       "evento_alvo": ["alvo|" + e for e in X.EVENTOS]}
    dados["evento_alvo"] = dados["evento"]
    return dados


def descritivo(dados):
    linhas = []
    for reg in AC.REGIMES:
        for dom in DOMINIOS:
            for cl in CLASSES:
                for fx in FAIXAS:
                    sel = ((dados["regime"] == reg) & (dados["dominio"] == dom) & (dados["classe"] == cl)
                           & (dados["faixa"] == fx))
                    n = int(sel.sum())
                    linhas.append([reg, dom, cl, fx, n, AC.fmt(float(dados["y"][sel].mean())) if n else ""])
    return linhas


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--margens", required=True)
    ap.add_argument("--chave", required=True)
    ap.add_argument("--confirmatoria", required=True)
    ap.add_argument("--distancias", required=True, help="folder with dist_<set>.npz written by E3")
    ap.add_argument("--saida", required=True)
    args = ap.parse_args(argv)

    log("E8 start")
    shas = X.conferir_congelados()
    conf, saida = os.path.abspath(args.confirmatoria), os.path.abspath(args.saida)
    if saida == conf or saida.startswith(conf + os.sep):
        raise ValueError("--saida must be outside the folder of the confirmatory run")
    if os.path.isdir(saida) and os.listdir(saida):
        raise ValueError(f"{saida} is not empty; choose a new folder")
    with open(os.path.join(conf, "resultados.json"), encoding="utf-8") as f:
        res_c = json.load(f)
    reg2cod = res_c["meta"]["chave"]
    dados = montar_dados(args, res_c, reg2cod)
    log(f"rows: {len(dados['y']):,}; units: {len(np.unique(dados['unidade_global']))}; "
        f"share unflagged: {dados['y'].mean():.4f}")

    termos = ["regime", "dominio", "regime:dominio", "classe", "faixa", "agua_z"]
    efeitos = ["unidade_global", "evento", "execucao"]
    ajustes = {}
    ajustes["principal"] = ajustar_misto(dados, termos, efeitos)
    log("main model fitted")
    ajustes["a_gee"] = ajustar_gee(dados, termos)
    log("sensitivity (a) fitted")
    ajustes["b_pesos"] = ajustar_gee(dados, termos, pesos=True)
    log("sensitivity (b) fitted")
    alvo = subconjunto(dados, dados["dominio"] == "alvo")
    ajustes["c_so_alvo"] = ajustar_misto(alvo, ["regime", "evento_alvo", "classe", "faixa", "agua_z"],
                                         ["unidade_global", "execucao"])
    log("sensitivity (c) fitted")

    os.makedirs(os.path.join(saida, "tabelas"), exist_ok=True)
    tab = os.path.join(saida, "tabelas")
    AC.escrever_csv(os.path.join(tab, "e8_descritivo.csv"),
                    ["regime", "dominio", "classe", "faixa", "linhas", "parcela_nao_sinalizada"], descritivo(dados))
    linhas = []
    for nome, a in ajustes.items():
        for e in a["fixos"]:
            linhas.append([nome, e["termo"], AC.fmt(e["estimativa"]), AC.fmt(e["erro"]), AC.fmt(e["or"]),
                           AC.fmt(e["ic95"][0]), AC.fmt(e["ic95"][1]), a["linhas"]])
    AC.escrever_csv(os.path.join(tab, "e8_efeitos.csv"),
                    ["ajuste", "termo", "estimativa_log_odds", "erro", "razao_de_chances", "ic95_inf", "ic95_sup",
                     "linhas"], linhas)
    meta = {"plano": "prereg/plano_exploratorio_2.md, E8", "max_por_unidade": MAX_POR_UNIDADE,
            "semente_amostra": SEMENTE_AMOSTRA, "faixas": {"ate2": "d <= 2", "3a10": "3 <= d <= 10",
                                                           "mais10": "d > 10 ou sem a outra classe"},
            "congelados_sha256": shas, "e8_sha256": AC.sha256_arquivo(os.path.abspath(__file__)), "chave": reg2cod,
            "versoes": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                        "statsmodels": __import__("statsmodels").__version__},
            "conferencias": "tau igual ao da rodada confirmatoria; distancias do E3 conferidas contra os rotulos"}
    with open(os.path.join(saida, "e8.json"), "w", encoding="utf-8") as f:
        json.dump(AC.para_json({"meta": meta, "ajustes": ajustes}), f, indent=1, ensure_ascii=False)
    log("done")
    return ajustes


if __name__ == "__main__":
    main()
