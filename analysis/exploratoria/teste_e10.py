#!/usr/bin/env python3
"""Tests of E10 on the synthetic margins of analysis/gerador_sintetico.py.

1. The frozen analysis runs first (small B).
2. IsoMax+ files equal to the R1 margins: every IsoMax+ minus R1 contrast must be exactly 0, with a
   zero-width interval (same draws for both arms), and the minimum distance score equal to |m| must give the
   same AUROC_e as the MSP.
3. IsoMax+ files with other margins and a distance score of their own: the script runs, the tables exist and
   the histogram AUROC_e of the distance score is within the tolerance of the exact value.
4. A missing IsoMax+ file and a non-empty output folder must stop the script.

Usage: python analysis/exploratoria/teste_e10.py
"""
import json
import os
import shutil
import sys
import tempfile

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(AQUI))
sys.path.insert(0, AQUI)
import analise_congelada as AC  # noqa: E402
import e10_isomax as E10  # noqa: E402
import gerador_sintetico as G  # noqa: E402
import nucleo as N  # noqa: E402

falhas = []


def checar(cond, msg):
    if not cond:
        falhas.append(msg)
        print("FALHA:", msg)


def escrever_iso(margens, cod, pasta, transformar):
    os.makedirs(pasta, exist_ok=True)
    corridas = AC.listar_corridas(margens)
    for s in E10.SEMENTES:
        for c in E10.CONJ_ISO:
            z = np.load(corridas[cod][s][c], allow_pickle=False)
            m, d = transformar(z["margem"].astype(np.float32), s, c)
            np.savez_compressed(os.path.join(pasta, f"iso_s{s}_{c}.npz"), chaves=z["chaves"], tamanhos=z["tamanhos"],
                                margem=m.astype(np.float32), mds=d.astype(np.float32))


def main():
    raiz = tempfile.mkdtemp(prefix="e10_")
    try:
        dados = G.gerar(os.path.join(raiz, "dados"), semente=17, escala=1.0)
        conf = os.path.join(raiz, "conf")
        AC.main(["--repo", dados["repo"], "--margens", dados["margens"], "--chave", dados["chave"],
                 "--saida", conf, "--B", "200"])
        with open(os.path.join(conf, "resultados.json"), encoding="utf-8") as f:
            reg2cod = json.load(f)["meta"]["chave"]
        base = ["--repo", dados["repo"], "--margens", dados["margens"], "--chave", dados["chave"],
                "--confirmatoria", conf, "--B", "300"]

        # 2. identical margins
        iso1 = os.path.join(raiz, "iso_igual")
        escrever_iso(dados["margens"], reg2cod["R1"], iso1, lambda m, s, c: (m, np.abs(m)))
        out = E10.main(base + ["--isomax", iso1, "--saida", os.path.join(raiz, "s1")])
        c = out["contrastes"]
        for nome in ("iso_menos_r1_auroc_alvo", "iso_menos_r1_auroc_teste", "iso_menos_r1_nao_sinalizado_alvo_pp"):
            checar(abs(c[nome]["estimativa"]) < 1e-12, f"{nome} should be 0, got {c[nome]['estimativa']}")
            checar(max(abs(v) for v in c[nome]["ic90"]) < 1e-12, f"{nome} interval should be 0, got {c[nome]['ic90']}")
        for nome in ("iso_distancia_menos_msp_alvo", "iso_distancia_menos_msp_teste"):
            checar(abs(c[nome]["estimativa"]) < 1e-12, f"{nome} should be 0 when the score is |m|")
        checar(abs(c["delta_desl_iso"]["estimativa"] - c["delta_desl_r1_sementes_1a3"]["estimativa"]) < 1e-12,
               "delta_desl of the two arms should be equal")

        # 3. other margins and a distance score of their own
        def outra(m, s, c):
            rng = np.random.default_rng(100 * s + E10.CONJ_ISO.index(c))
            m2 = (0.6 * m + rng.normal(0.0, 0.4, m.size)).astype(np.float32)
            d = (-np.abs(rng.normal(1.0, 0.3, m.size)) + 0.2 * np.abs(m2)).astype(np.float32)
            return m2, d
        iso2 = os.path.join(raiz, "iso_outra")
        escrever_iso(dados["margens"], reg2cod["R1"], iso2, outra)
        out2 = E10.main(base + ["--isomax", iso2, "--saida", os.path.join(raiz, "s2")])
        for nome in ("e10_execucoes", "e10_metricas", "e10_contrastes"):
            checar(os.path.exists(os.path.join(raiz, "s2", "tabelas", nome + ".csv")), f"table {nome} missing")
        for s in E10.SEMENTES:
            for conj in ("val", "teste", "alvo"):
                r = out2["est"]["ISO"][s][conj]
                checar(abs(r["auroc_d"] - r["auroc_d_hist"]) <= N.HIST_TOL, f"distance histogram AUROC s{s} {conj}")
                checar(abs(r["auroc"] - r["auroc_hist"]) <= N.HIST_TOL, f"MSP histogram AUROC s{s} {conj}")
        c2 = out2["contrastes"]
        checar(all(c2[n]["ic90"][0] <= c2[n]["ic90"][1] for n in c2), "interval bounds out of order")
        checar(c2["iso_menos_r1_auroc_alvo"]["estimativa"] != 0.0, "contrast should not be 0 with other margins")

        # 4. refusals
        os.remove(os.path.join(iso2, "iso_s3_t2.npz"))
        try:
            E10.main(base + ["--isomax", iso2, "--saida", os.path.join(raiz, "s3")])
            checar(False, "a missing IsoMax+ file should stop the script")
        except FileNotFoundError:
            pass
        try:
            E10.main(base + ["--isomax", iso1, "--saida", os.path.join(raiz, "s1")])
            checar(False, "a non-empty output folder should stop the script")
        except ValueError:
            pass
    finally:
        shutil.rmtree(raiz, ignore_errors=True)
    print(f"{len(falhas)} falha(s)")
    sys.exit(1 if falhas else 0)


if __name__ == "__main__":
    main()
