#!/usr/bin/env python3
"""Tests of exploratorias.py on synthetic data (analysis/gerador_sintetico.py).

1. The frozen analysis runs on the synthetic margins (small B), then exploratorias.py runs on its output; every
   check against the confirmatory run must pass and the tables must be internally consistent.
2. A changed tau in the confirmatory output must stop the script.
3. An output folder inside the confirmatory folder must be refused.

Usage: python analysis/exploratoria/teste_exploratorias.py
"""
import csv
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
import exploratorias as X  # noqa: E402
import gerador_sintetico as G  # noqa: E402


def ler(caminho):
    with open(caminho, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def deve_falhar(funcao, trecho):
    try:
        funcao()
    except ValueError as e:
        assert trecho in str(e), f"unexpected message: {e}"
        return
    raise AssertionError(f"expected an error containing {trecho!r}")


def main():
    raiz = tempfile.mkdtemp(prefix="exploratorias_")
    try:
        dados = G.gerar(os.path.join(raiz, "dados"), semente=91, escala=2.5)
        conf = os.path.join(raiz, "conf")
        AC.main(["--repo", dados["repo"], "--margens", dados["margens"], "--chave", dados["chave"],
                 "--saida", conf, "--B", "200"])
        base = ["--repo", dados["repo"], "--margens", dados["margens"], "--chave", dados["chave"],
                "--confirmatoria", conf]
        saida = os.path.join(raiz, "expl")
        out = X.main(base + ["--saida", saida, "--B", "300"])

        tab = os.path.join(saida, "tabelas")
        for nome in ("e1_eventos", "e2_erros_por_classe", "e4_conjunto", "e4_conjunto_r3_menos_r1", "e6_unidades",
                     "e6_resumo"):
            assert os.path.exists(os.path.join(tab, nome + ".csv")), nome
        assert os.path.exists(os.path.join(saida, "figuras", "e1_floresta.png"))

        # E2: the two classes split the errors and the unflagged errors
        e2 = out["e2"]
        for reg, conjs in e2.items():
            for conj, d in conjs.items():
                for k in ("parcela_erros", "parcela_nao_sinalizados"):
                    s = d["falsa_agua"][k]["media"] + d["falsa_terra"][k]["media"]
                    assert abs(s - 1.0) < 1e-9, (reg, conj, k, s)

        # E4: the bootstrap point estimate is the difference of the ensemble values
        e4 = out["e4"]["por_regime"]
        for chave, v in out["e4"]["conjunto_r3_menos_r1"].items():
            d = e4["R3"][chave]["conjunto"] - e4["R1"][chave]["conjunto"]
            assert (np.isnan(d) and np.isnan(v["ponto"])) or abs(d - v["ponto"]) < 1e-12, chave
            if not np.isnan(v["ponto"]):
                assert v["ic90"][0] <= v["ponto"] <= v["ic90"][1] or v["replicas_nan"] > 0, chave

        # E6: the synthetic units are large enough for at least one event to be summarised
        assert any(r["spearman"] is not None for r in out["e6"].values()), out["e6"]
        assert os.path.exists(os.path.join(saida, "figuras", "e6_dispersao.png"))
        linhas = ler(os.path.join(tab, "e6_unidades.csv"))
        for l in linhas:
            if l["elegivel"] == "True":
                assert abs(float(l["auroc_e_R3"]) - float(l["auroc_e_R1"]) - float(l["delta_reg"])) < 2e-6

        # a changed tau in the confirmatory output stops the script
        conf2 = os.path.join(raiz, "conf2")
        shutil.copytree(conf, conf2)
        caminho = os.path.join(conf2, "resultados.json")
        with open(caminho, encoding="utf-8") as f:
            r = json.load(f)
        r["execucoes"][0]["tau"] += 1e-6
        with open(caminho, "w", encoding="utf-8") as f:
            json.dump(r, f)
        deve_falhar(lambda: X.main(base[:-1] + [conf2, "--saida", os.path.join(raiz, "x2"), "--B", "50"]), "tau of")

        # the output folder may not be inside the confirmatory folder
        deve_falhar(lambda: X.main(base + ["--saida", os.path.join(conf, "expl"), "--B", "50"]), "outside the folder")
        print("all tests passed")
    finally:
        shutil.rmtree(raiz, ignore_errors=True)


if __name__ == "__main__":
    main()
