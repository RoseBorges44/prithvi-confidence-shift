#!/usr/bin/env python3
"""End-to-end test of E7, E8 and E9 on the synthetic margins of analysis/gerador_sintetico.py.

The frozen analysis runs first (small B). Then: E7 with CKA files written here, E8 with distance files written
here (1 <= d <= 2 exactly on the boundary flags), and E9 with masks that place the scored pixels of each unit at
the start of the grid. Every check against the confirmatory run must pass and the tables must exist.

Usage: python analysis/exploratoria/teste_e7_e8_e9_ponta.py [--sem-e8]

E8 takes about an hour here (two variational passes per mixed model); --sem-e8 skips it.
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
import e7_cka as E7  # noqa: E402
import e8_misto as E8  # noqa: E402
import e9_fragmentacao as E9  # noqa: E402
import gerador_sintetico as G  # noqa: E402


def escrever_distancias(margens, pasta, rng):
    os.makedirs(pasta, exist_ok=True)
    for conj in ("teste", "bolivia", "t2"):
        r = AC.ler_rotulos(margens, conj)
        d = rng.integers(3, 40, size=r["rotulo"].size).astype(np.int16)
        d[rng.random(d.size) < 0.05] = -1
        d[r["borda"]] = rng.integers(1, 3, size=int(r["borda"].sum()))
        np.savez_compressed(os.path.join(pasta, f"dist_{conj}.npz"), chaves=np.array(r["chaves"]),
                            tamanhos=r["tamanhos"], distancia=d)


def escrever_mascaras(margens, pasta):
    os.makedirs(pasta, exist_ok=True)
    for conj in E9.CONJ_MASK:
        r = AC.ler_rotulos(margens, conj)
        masc = []
        for t in r["tamanhos"]:
            g = np.zeros(E9.TAM * E9.TAM, dtype=bool)
            g[:t] = True
            masc.append(np.packbits(g))
        np.savez_compressed(os.path.join(pasta, f"mascara_{conj}.npz"), chaves=np.array(r["chaves"]),
                            tamanhos=r["tamanhos"], mascara=np.stack(masc))


def escrever_cka(pasta, reg2cod, rng):
    os.makedirs(pasta, exist_ok=True)
    corridas = [("R1", 1)] + [(r, s) for r in ("R2", "R3") for s in range(1, 6)]
    for reg, sem in corridas:
        base = 1.0 if reg == "R1" else (0.95 if reg == "R2" else 0.7)
        cka = {g: {str(c): (1.0 if reg == "R1" else float(base - 0.01 * k - rng.random() * 0.02))
                   for k, c in enumerate(E7.CAMADAS)} for g in E7.GRUPOS + ("alvo",)}
        rec = {"regime": reg, "semente": sem, "codigo": reg2cod[reg], "camadas": list(E7.CAMADAS), "cka": cka,
               "tokens": {g: 1024 for g in E7.GRUPOS + ("alvo",)}}
        with open(os.path.join(pasta, f"cka_{reg}_s{sem}.json"), "w", encoding="utf-8") as f:
            json.dump(rec, f)


def main():
    raiz = tempfile.mkdtemp(prefix="e789_")
    try:
        rng = np.random.default_rng(7)
        dados = G.gerar(os.path.join(raiz, "dados"), semente=91, escala=2.5)
        conf = os.path.join(raiz, "conf")
        AC.main(["--repo", dados["repo"], "--margens", dados["margens"], "--chave", dados["chave"],
                 "--saida", conf, "--B", "200"])
        with open(os.path.join(conf, "resultados.json"), encoding="utf-8") as f:
            reg2cod = json.load(f)["meta"]["chave"]
        base = ["--repo", dados["repo"], "--margens", dados["margens"], "--chave", dados["chave"],
                "--confirmatoria", conf]

        escrever_cka(os.path.join(raiz, "cka"), reg2cod, rng)
        out7 = E7.main(base + ["--cka", os.path.join(raiz, "cka"), "--saida", os.path.join(raiz, "e7")])
        assert -1.0 <= out7["spearman"] <= 1.0
        for nome in ("e7_cka", "e7_cka_resumo", "e7_auroc"):
            assert os.path.exists(os.path.join(raiz, "e7", "tabelas", nome + ".csv")), nome

        if "--sem-e8" not in sys.argv:
            escrever_distancias(dados["margens"], os.path.join(raiz, "dist"), rng)
            out8 = E8.main(base + ["--distancias", os.path.join(raiz, "dist"), "--saida", os.path.join(raiz, "e8")])
            assert set(out8) == {"principal", "a_gee", "b_pesos", "c_so_alvo"}
            assert any(e["termo"] == "dominio[alvo]" for e in out8["principal"]["fixos"])
            for nome in ("principal", "c_so_alvo"):
                assert len(out8[nome]["convergencia"]) == 2, nome
                assert np.isfinite(out8[nome]["mudanca_max_fixos_entre_passadas"]), nome

        escrever_mascaras(dados["margens"], os.path.join(raiz, "mask"))
        out9 = E9.main(["analise"] + base + ["--mascaras", os.path.join(raiz, "mask"),
                                             "--saida", os.path.join(raiz, "e9"), "--B", "200"])
        for (reg, conj), r in out9.items():
            for viz in (8, 4):
                d = r[viz]
                assert abs(d["diferenca"] - (d["auroc_indice"] - d["auroc_msp"])) < 1e-12
        for nome in ("e9_auroc", "e9_unidades", "e9_eventos"):
            assert os.path.exists(os.path.join(raiz, "e9", "tabelas", nome + ".csv")), nome
        print("all tests passed")
    finally:
        shutil.rmtree(raiz, ignore_errors=True)


if __name__ == "__main__":
    main()
