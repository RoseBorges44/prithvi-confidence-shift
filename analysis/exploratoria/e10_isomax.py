#!/usr/bin/env python3
"""E10: exploratory IsoMax+ arm in R1 (prereg/plano_isomax.md).

Compares the R1 runs trained with the IsoMax+ loss (seeds 1 to 3) with the confirmatory R1 runs of the same
seeds (cross-entropy). For each run: tau on validation (nucleo.conformal_threshold), coverage, empty-set
rate, error rate, mIoU, unflagged error rate and AUROC_e of the MSP on the validation set, the source test,
the pooled primary target and each target event; for the IsoMax+ runs, also AUROC_e of the minimum distance
score. Contrasts with 90% and 95% intervals from a bootstrap of units within events (B = 10,000,
numpy.random.default_rng(5), seeds not resampled, the same draws for both arms).

The frozen scripts are imported read-only and their SHA-256 is checked first. For the confirmatory R1 runs,
tau and the exact validation AUROC_e must equal those of the confirmatory run.

Inputs:
  --margens      folder with the week-6 label files and margin files (run_<code>_s<seed>_<set>.npz)
  --isomax       folder with iso_s<seed>_<set>.npz (fields chaves, tamanhos, margem, mds; sets val, teste,
                 bolivia, t2), in the order of the week-6 label files
  --chave, --confirmatoria, --repo as in the other exploratory scripts

Usage: python analysis/exploratoria/e10_isomax.py --repo . --margens M --isomax I --chave K \
           --confirmatoria C --saida S [--B 10000]
"""
import argparse
import json
import os
import platform
import sys

import numpy as np
import scipy

AQUI = os.path.dirname(os.path.abspath(__file__))
ANALISE = os.path.dirname(AQUI)
sys.path.insert(0, ANALISE)
sys.path.insert(0, AQUI)
import analise_congelada as AC  # noqa: E402
import exploratorias as X  # noqa: E402
import nucleo as N  # noqa: E402
import reamostragem as R  # noqa: E402

SEMENTES = (1, 2, 3)            # plan
SEMENTE_BOOT = 5                # plan
B_PADRAO = 10_000               # plan
N_BINS_MDS = 8000
MAX_REFINO = 6
CONJ_ISO = ("val", "teste", "bolivia", "t2")
BRACOS = ("R1", "ISO")

log = AC.log


# ----------------------------------------------------------------------------------------------
# Histograms for any confidence score
# ----------------------------------------------------------------------------------------------

class BinsValores:
    """Bins with fixed increasing edges for a confidence score (higher means more confident).

    Same interface as nucleo.Bins (attribute n and method index), so nucleo.unit_histograms accepts it."""

    def __init__(self, bordas):
        self.bordas = np.unique(np.asarray(bordas, dtype=np.float64))
        self.n = self.bordas.size + 1

    def index(self, v):
        return np.searchsorted(self.bordas, np.asarray(v, dtype=np.float64), side="right")


def bins_por_quantis(valores, n):
    """Equal-frequency edges of the pooled values."""
    q = np.linspace(0.0, 1.0, n + 1)[1:-1]
    return BinsValores(np.quantile(np.asarray(valores, dtype=np.float64), q))


def histogramas(uni, conf, err, bins_ini, refinar):
    """Per-unit histograms of a confidence score with the bins refined until the pooled histogram AUROC_e is
    within nucleo.HIST_TOL of the exact value. Returns he, hc, exact AUROC_e, histogram AUROC_e, bins."""
    exato = N.auroc_exact(conf[err], conf[~err])
    bins = bins_ini
    for _ in range(MAX_REFINO + 1):
        he, hc = N.unit_histograms(uni.unit_id, conf, err, uni.n, bins)
        h = float(N.auroc_from_hist(he.sum(0, dtype=np.int64), hc.sum(0, dtype=np.int64)))
        if np.isnan(exato) and np.isnan(h):
            break
        if abs(h - exato) <= N.HIST_TOL:
            break
        bins = refinar(bins)
    else:
        raise ValueError(f"Histogram AUROC_e still differs from the exact value after {MAX_REFINO} refinements")
    he, hc = AC.compactar(he, hc)
    return he, hc, float(exato), h, bins


def refinar_msp(maximo):
    return lambda bins: N.refine_bins(bins, maximo)


def refinar_mds(valores):
    return lambda bins: bins_por_quantis(valores, 2 * (bins.n - 1))


# ----------------------------------------------------------------------------------------------
# One run on one universe
# ----------------------------------------------------------------------------------------------

def estatisticas(uni, m, tau, mds=None, eventos=()):
    """Counts per unit, histograms and exact AUROC_e of the MSP (and of the minimum distance score if given),
    pooled and per event."""
    y = uni.rotulo.astype(bool)
    err = (m > 0) != y
    a = np.abs(m)
    out = {"cont": N.unit_counts(uni.unit_id, m, y, tau, uni.n)}
    maximo = float(a.max())
    out["he"], out["hc"], out["auroc"], out["auroc_hist"], _ = histogramas(
        uni, a, err, N.Bins(), refinar_msp(maximo))
    if mds is not None:
        out["he_d"], out["hc_d"], out["auroc_d"], out["auroc_d_hist"], bd = histogramas(
            uni, mds, err, bins_por_quantis(mds, N_BINS_MDS), refinar_mds(mds))
        out["bins_mds"] = int(bd.n)
    out["eventos"] = {}
    ev_px = uni.evento[uni.unit_id]
    for e in eventos:
        sel = ev_px == e
        d = {"cont": out["cont"][uni.evento == e].sum(0),
             "auroc": N.auroc_exact(a[sel & err], a[sel & ~err])}
        if mds is not None:
            d["auroc_d"] = N.auroc_exact(mds[sel & err], mds[sel & ~err])
        out["eventos"][e] = d
    return out


def processar_corrida(arq, ctx, uni, px, com_mds):
    rot = ctx["rot"]
    m_v = AC.ler_margens(arq["val"], rot["val"])
    q_hat, tau = N.conformal_threshold(m_v, rot["val"]["rotulo"])
    val_u = AC.universo_validacao(ctx)
    d_v = AC.ler_margens(arq["val"], rot["val"], "mds") if com_mds else None
    out = {"tau": tau, "q_hat": q_hat, "val": estatisticas(val_u, m_v, tau, d_v)}
    del m_v, d_v
    fonte = ctx["fonte"]
    m_f = AC.margens_universo(arq, ctx, "fonte")
    d_f = AC.margens_universo(arq, ctx, "fonte", "mds") if com_mds else None
    out["teste"] = estatisticas(fonte, m_f, tau, d_f, sorted(set(fonte.evento)))
    del m_f, d_f
    m_a = AC.margens_universo(arq, ctx, "alvo")[px]
    d_a = AC.margens_universo(arq, ctx, "alvo", "mds")[px] if com_mds else None
    out["alvo"] = estatisticas(uni, m_a, tau, d_a, X.EVENTOS)
    return out


def arquivos_iso(pasta, sem):
    arq = {c: os.path.join(pasta, f"iso_s{sem}_{c}.npz") for c in CONJ_ISO}
    for c, p in arq.items():
        if not os.path.isfile(p):
            raise FileNotFoundError(p)
    return arq


# ----------------------------------------------------------------------------------------------
# Summaries
# ----------------------------------------------------------------------------------------------

def metricas_contagem(c):
    return {k: float(v) for k, v in N.count_metrics(np.asarray(c).sum(0) if np.ndim(c) == 2 else c).items()}


def linha_metricas(braco, semente, conjunto, unidades, cm, auroc, auroc_d):
    return [braco, semente, conjunto, unidades, AC.fmt(cm["taxa_erro"]), AC.fmt(cm["miou"]),
            AC.fmt(cm["cobertura"]), AC.fmt(cm["conjunto_vazio"]), AC.fmt(cm["erro_nao_sinalizado"]),
            AC.fmt(auroc), AC.fmt(auroc_d) if auroc_d is not None else ""]


def intervalo(rep, ponto, escala=1.0):
    rep = np.asarray(rep, dtype=np.float64) * escala
    return {"estimativa": float(ponto) * escala,
            "ic90": [float(v) for v in R.quantile_interval(rep, 0.90)],
            "ic95": [float(v) for v in R.quantile_interval(rep, 0.95)],
            "replicas_nan": int(np.isnan(rep).sum())}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--margens", required=True)
    ap.add_argument("--isomax", required=True)
    ap.add_argument("--chave", required=True)
    ap.add_argument("--confirmatoria", required=True)
    ap.add_argument("--saida", required=True)
    ap.add_argument("--B", type=int, default=B_PADRAO)
    args = ap.parse_args(argv)

    log("E10 start")
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
    uni, px = X.universo_primario(ctx["alvo"])
    fonte = ctx["fonte"]
    execs = {(e["codigo"], int(e["semente"])): e for e in res_c["execucoes"]}

    est = {b: {} for b in BRACOS}
    cod1 = reg2cod["R1"]
    for sem in SEMENTES:
        r = processar_corrida(corridas[cod1][sem], ctx, uni, px, com_mds=False)
        X.igual(r["tau"], execs[(cod1, sem)]["tau"], f"tau of R1 s{sem}")
        X.igual(r["val"]["auroc"], execs[(cod1, sem)]["val_auroc_e"], f"validation AUROC_e of R1 s{sem}")
        est["R1"][sem] = r
        log(f"  R1 s{sem}: tau and validation AUROC_e equal the confirmatory run")
    for sem in SEMENTES:
        est["ISO"][sem] = processar_corrida(arquivos_iso(args.isomax, sem), ctx, uni, px, com_mds=True)
        log(f"  IsoMax+ s{sem}: tau {est['ISO'][sem]['tau']:.4f}")

    # bootstrap draws, fixed order: source test, then target
    rng = np.random.default_rng(SEMENTE_BOOT)
    estratos_f = [np.flatnonzero(fonte.evento == e) for e in sorted(set(fonte.evento))]
    estratos_a = [np.flatnonzero(uni.evento == e) for e in X.EVENTOS]
    cnt = {"teste": R.stratified_counts(rng, estratos_f, fonte.n, args.B),
           "alvo": R.stratified_counts(rng, estratos_a, uni.n, args.B)}

    rep, pts = {}, {}
    for b in BRACOS:
        for conj in ("teste", "alvo"):
            au = [R.hist_metrics(cnt[conj], est[b][s][conj]["he"], est[b][s][conj]["hc"])[0] for s in SEMENTES]
            rep[(b, conj, "auroc")] = np.mean(au, axis=0)
            pts[(b, conj, "auroc")] = float(np.mean([est[b][s][conj]["auroc"] for s in SEMENTES]))
            un = [R.count_metrics_replicates(cnt[conj], est[b][s][conj]["cont"])["erro_nao_sinalizado"]
                  for s in SEMENTES]
            rep[(b, conj, "nao_sinalizado")] = np.mean(un, axis=0)
            pts[(b, conj, "nao_sinalizado")] = float(np.mean(
                [metricas_contagem(est[b][s][conj]["cont"])["erro_nao_sinalizado"] for s in SEMENTES]))
            if b == "ISO":
                ad = [R.hist_metrics(cnt[conj], est[b][s][conj]["he_d"], est[b][s][conj]["hc_d"])[0]
                      for s in SEMENTES]
                rep[(b, conj, "auroc_d")] = np.mean(ad, axis=0)
                pts[(b, conj, "auroc_d")] = float(np.mean([est[b][s][conj]["auroc_d"] for s in SEMENTES]))

    def dif(k1, k2, escala=1.0):
        return intervalo(rep[k1] - rep[k2], pts[k1] - pts[k2], escala)

    contrastes = {
        "iso_menos_r1_auroc_alvo": dif(("ISO", "alvo", "auroc"), ("R1", "alvo", "auroc")),
        "iso_menos_r1_auroc_teste": dif(("ISO", "teste", "auroc"), ("R1", "teste", "auroc")),
        "iso_menos_r1_nao_sinalizado_alvo_pp": dif(("ISO", "alvo", "nao_sinalizado"),
                                                   ("R1", "alvo", "nao_sinalizado"), 100.0),
        "iso_distancia_menos_msp_alvo": dif(("ISO", "alvo", "auroc_d"), ("ISO", "alvo", "auroc")),
        "iso_distancia_menos_msp_teste": dif(("ISO", "teste", "auroc_d"), ("ISO", "teste", "auroc")),
        "delta_desl_iso": dif(("ISO", "alvo", "auroc"), ("ISO", "teste", "auroc")),
        "delta_desl_r1_sementes_1a3": dif(("R1", "alvo", "auroc"), ("R1", "teste", "auroc")),
    }
    for nome, c in contrastes.items():
        log(f"  {nome}: {c['estimativa']:+.4f} [{c['ic90'][0]:+.4f}; {c['ic90'][1]:+.4f}]")

    # tables
    tab = os.path.join(saida, "tabelas")
    os.makedirs(tab, exist_ok=True)
    linhas_x = []
    for b in BRACOS:
        for s in SEMENTES:
            r = est[b][s]
            cm = metricas_contagem(r["val"]["cont"])
            linhas_x.append([b, s, AC.fmt(r["tau"]), AC.fmt(cm["taxa_erro"]), AC.fmt(cm["miou"]),
                             AC.fmt(r["val"]["auroc"]), AC.fmt(r["val"]["auroc_d"]) if b == "ISO" else ""])
    AC.escrever_csv(os.path.join(tab, "e10_execucoes.csv"),
                    ["braco", "semente", "tau", "val_taxa_erro", "val_miou", "val_auroc_msp", "val_auroc_distancia"],
                    linhas_x)

    linhas_m = []
    for b in BRACOS:
        por_conj = {}
        for s in SEMENTES:
            r = est[b][s]
            for conj, u_n in (("val", 1), ("teste", fonte.n), ("alvo", uni.n)):
                cm = metricas_contagem(r[conj]["cont"])
                ad = r[conj].get("auroc_d")
                linhas_m.append(linha_metricas(b, s, conj, u_n, cm, r[conj]["auroc"], ad))
                por_conj.setdefault(conj, []).append((cm, r[conj]["auroc"], ad))
            for e, d in r["alvo"]["eventos"].items():
                cm = N.count_metrics(d["cont"])
                cm = {k: float(v) for k, v in cm.items()}
                n_e = int((uni.evento == e).sum())
                linhas_m.append(linha_metricas(b, s, e, n_e, cm, d["auroc"], d.get("auroc_d")))
                por_conj.setdefault(e, []).append((cm, d["auroc"], d.get("auroc_d")))
        for conj, lista in por_conj.items():
            cm = {k: float(np.mean([x[0][k] for x in lista])) for k in lista[0][0]}
            au = float(np.mean([x[1] for x in lista]))
            ad = float(np.mean([x[2] for x in lista])) if lista[0][2] is not None else None
            n_c = 1 if conj == "val" else (fonte.n if conj == "teste" else
                                           (uni.n if conj == "alvo" else int((uni.evento == conj).sum())))
            linhas_m.append(linha_metricas(b, "media", conj, n_c, cm, au, ad))
    AC.escrever_csv(os.path.join(tab, "e10_metricas.csv"),
                    ["braco", "semente", "conjunto", "unidades", "taxa_erro", "miou", "cobertura", "conjunto_vazio",
                     "erro_nao_sinalizado", "auroc_msp", "auroc_distancia"], linhas_m)

    AC.escrever_csv(os.path.join(tab, "e10_contrastes.csv"),
                    ["contraste", "estimativa", "ic90_inf", "ic90_sup", "ic95_inf", "ic95_sup", "replicas_nan"],
                    [[n, AC.fmt(c["estimativa"]), AC.fmt(c["ic90"][0]), AC.fmt(c["ic90"][1]), AC.fmt(c["ic95"][0]),
                      AC.fmt(c["ic95"][1]), c["replicas_nan"]] for n, c in contrastes.items()])

    meta = {"plano": "prereg/plano_isomax.md", "sementes": list(SEMENTES), "B": args.B,
            "semente_bootstrap": SEMENTE_BOOT, "congelados_sha256": shas,
            "e10_sha256": AC.sha256_arquivo(os.path.abspath(__file__)), "chave": reg2cod,
            "iso_sha256": {os.path.basename(p): AC.sha256_arquivo(p)
                           for s in SEMENTES for p in arquivos_iso(args.isomax, s).values()},
            "versoes": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
            "conferencias": "tau e AUROC_e da validacao do R1 (sementes 1 a 3) iguais aos da rodada confirmatoria",
            "bins_distancia": {f"s{s}|{c}": est["ISO"][s][c]["bins_mds"] for s in SEMENTES
                               for c in ("val", "teste", "alvo")}}
    taus = {f"{b}|{s}": est[b][s]["tau"] for b in BRACOS for s in SEMENTES}
    with open(os.path.join(saida, "e10.json"), "w", encoding="utf-8") as f:
        json.dump(AC.para_json({"meta": meta, "taus": taus, "contrastes": contrastes}), f, indent=1,
                  ensure_ascii=False)
    log("done")
    return {"contrastes": contrastes, "est": est}


if __name__ == "__main__":
    main()
