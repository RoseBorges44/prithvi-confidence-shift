#!/usr/bin/env python3
"""Exploratory analysis E5 of prereg/plano_exploratorio.md: the released checkpoint (R0) through the inference
pipeline of week 6.

The margins of R0 are inferred by notebooks/semana8_e5_r0.ipynb with the reading, preprocessing and 32-bit
inference of notebooks/semana6_inferencia_alvos.ipynb, and written as r0_<set>.npz (chaves, tamanhos, margem) for
the sets val, teste and bolivia. This script computes tau on the validation set and, on the source test and on
Bolivia, AUROC_e, the unflagged error rate, the error rate and the mIoU, with the functions of the frozen nucleo.py.
It reports them next to analysis/r0_referencia.json (week 1) and, for Bolivia, next to the range of the five R3
seeds used by precondition 2. The precondition result of the confirmatory run does not change.

    python analysis/exploratoria/e5_r0.py --margens-r0 DIR --margens DIR --confirmatoria DIR --saida OUT
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

CONJUNTOS = ("val", "teste", "bolivia")
METRICAS = ("auroc_e", "erro_nao_sinalizado", "taxa_erro", "miou")
log = AC.log


def normalizar_chaves(chaves_arquivo, chaves_modelo):
    """Which checkpoint key fills each model key. The checkpoint may name the tensors as the model does or with
    the prefix 'model.' of the Lightning task. Returns (map model key -> checkpoint key, unused checkpoint keys).
    Stops when a model key has no tensor in the checkpoint."""
    arquivo = list(chaves_arquivo)
    mapa = {}
    for prefixo in ("", "model."):
        candidato = {k[len(prefixo):]: k for k in arquivo if k.startswith(prefixo)}
        if set(chaves_modelo) <= set(candidato):
            mapa = {k: candidato[k] for k in chaves_modelo}
            break
    if not mapa:
        faltam = sorted(set(chaves_modelo) - {k.removeprefix("model.") for k in arquivo})
        raise ValueError(f"{len(faltam)} model tensors have no tensor in the checkpoint, e.g. {faltam[:3]}")
    sobras = sorted(set(arquivo) - set(mapa.values()))
    return mapa, sobras


def ler_margens_r0(pasta, conj, rot):
    z = np.load(os.path.join(pasta, f"r0_{conj}.npz"), allow_pickle=False)
    if [str(c) for c in z["chaves"]] != rot["chaves"] or not np.array_equal(z["tamanhos"], rot["tamanhos"]):
        raise ValueError(f"r0_{conj}.npz: units or sizes differ from labels_{conj}.npz")
    m = z["margem"]
    if m.dtype != np.float32 or m.size != rot["rotulo"].size or not np.isfinite(m).all():
        raise ValueError(f"r0_{conj}.npz: margins must be finite float32, one per scored pixel")
    return m


def metricas(m, y, tau):
    y = np.asarray(y).astype(bool)
    a = np.abs(m)
    err = (m > 0) != y
    c = N.unit_counts(np.zeros(m.size, dtype=np.int64), m, y, tau, 1)[0]
    cm = N.count_metrics(c)
    return {"auroc_e": float(N.auroc_exact(a[err], a[~err])), "erro_nao_sinalizado": float(cm["erro_nao_sinalizado"]),
            "taxa_erro": float(cm["taxa_erro"]), "miou": float(cm["miou"])}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--margens-r0", required=True, help="folder with r0_val.npz, r0_teste.npz, r0_bolivia.npz")
    ap.add_argument("--margens", required=True, help="folder with labels_<set>.npz of the target inference")
    ap.add_argument("--confirmatoria", required=True, help="output folder of the confirmatory run")
    ap.add_argument("--saida", required=True)
    args = ap.parse_args(argv)

    log("E5 start")
    shas = X.conferir_congelados()
    conf, saida = os.path.abspath(args.confirmatoria), os.path.abspath(args.saida)
    if saida == conf or saida.startswith(conf + os.sep):
        raise ValueError("--saida must be outside the folder of the confirmatory run")
    if os.path.isdir(saida) and os.listdir(saida):
        raise ValueError(f"{saida} is not empty; choose a new folder")
    with open(os.path.join(conf, "resultados.json"), encoding="utf-8") as f:
        res_c = json.load(f)
    rot = {c: AC.ler_rotulos(args.margens, c) for c in CONJUNTOS}
    m = {c: ler_margens_r0(args.margens_r0, c, rot[c]) for c in CONJUNTOS}
    q_hat, tau = N.conformal_threshold(m["val"], rot["val"]["rotulo"])
    taxa_val = float(((m["val"] > 0) != rot["val"]["rotulo"].astype(bool)).mean())
    if taxa_val >= 0.5:
        raise ValueError(f"Validation error rate {taxa_val:.3f}: the margins look inverted or broken")
    log(f"R0, week-6 pipeline: q_hat {q_hat:.6f}, tau {tau:.6f}, validation error rate {taxa_val:.4f}")

    ref = AC.carregar_r0()
    faixa = res_c["precondicoes"]["p2_r0_na_bolivia"]
    out = {"q_hat": {"semana1": ref["q_hat"], "semana6": q_hat}, "tau": {"semana1": ref["tau"], "semana6": tau},
           "val_taxa_erro_semana6": taxa_val}
    linhas = [["val", "q_hat", AC.fmt(ref["q_hat"]), AC.fmt(q_hat), AC.fmt(q_hat - ref["q_hat"]), "", "", "", ""],
              ["val", "tau", AC.fmt(ref["tau"]), AC.fmt(tau), AC.fmt(tau - ref["tau"]), "", "", "", ""]]
    for conj in ("teste", "bolivia"):
        v = metricas(m[conj], rot[conj]["rotulo"], tau)
        out[conj] = {}
        for k in METRICAS:
            d = {"semana1": ref[conj][k], "semana6": v[k], "diferenca": v[k] - ref[conj][k]}
            extra = ["", "", "", ""]
            if conj == "bolivia":
                lo, hi = faixa[k]["minimo_r3"], faixa[k]["maximo_r3"]
                d.update({"r3_minimo": lo, "r3_maximo": hi, "dentro_semana1": faixa[k]["dentro"],
                          "dentro_semana6": bool(lo <= v[k] <= hi)})
                extra = [AC.fmt(lo), AC.fmt(hi), "sim" if faixa[k]["dentro"] else "nao",
                         "sim" if d["dentro_semana6"] else "nao"]
            out[conj][k] = d
            linhas.append([conj, k, AC.fmt(ref[conj][k]), AC.fmt(v[k]), AC.fmt(v[k] - ref[conj][k])] + extra)
            log(f"  {conj} {k}: week 1 {ref[conj][k]:.6f}, week-6 pipeline {v[k]:.6f}")

    os.makedirs(saida, exist_ok=True)
    AC.escrever_csv(os.path.join(saida, "e5_r0.csv"),
                    ["conjunto", "metrica", "semana1", "pipeline_semana6", "diferenca", "r3_minimo", "r3_maximo",
                     "dentro_semana1", "dentro_pipeline_semana6"], linhas)
    meta = {"plano": "prereg/plano_exploratorio.md, E5", "congelados_sha256": shas,
            "e5_sha256": AC.sha256_arquivo(os.path.abspath(__file__)), "referencia": ref.get("origem"),
            "versoes": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
            "observacao": "a pre-condicao 2 registrada pela rodada confirmatoria nao muda"}
    with open(os.path.join(saida, "e5.json"), "w", encoding="utf-8") as f:
        json.dump(AC.para_json({"meta": meta, **out}), f, indent=1, ensure_ascii=False)
    log("done")
    return out


if __name__ == "__main__":
    main()
