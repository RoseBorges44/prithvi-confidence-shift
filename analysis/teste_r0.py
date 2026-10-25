#!/usr/bin/env python3
"""Reproduction of the R0 values by the analysis code (prereg/preregistro.md, section 10).

The released checkpoint (R0) was evaluated in week 1. Its per-chip histograms are in two pickle files
(fonte_registros.pkl: validation and test chips; bolivia_registros.pkl: Bolivia chips). Each record has the
chip name, a histogram h of 20000 bins and the chip's pixel count. The first 10000 bins hold the errors in
decreasing confidence and the last 10000 bins hold the correct pixels in increasing confidence.

This script
1. recomputes the pooled AUROC_e and the unflagged error rate on the source test and on Bolivia with the
   functions of the analysis, and compares them with analysis/r0_referencia.json to the third decimal;
2. finds the bin of the conformal threshold from the validation histograms;
3. computes the stratified standard error of Delta_shift for R0 (source test chips resampled within their
   source event, Bolivia chips resampled as one stratum), B = 10,000, seed 0.

Usage: python analysis/teste_r0.py REPO fonte_registros.pkl bolivia_registros.pkl
The pickles are read with a restricted unpickler that only builds dictionaries, lists and arrays.
"""
import json
import math
import os
import pickle
import sys

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import analise_congelada as A  # noqa: E402
import nucleo as N  # noqa: E402
import reamostragem as R  # noqa: E402

BINS_METADE = 10000


class Seguro(pickle.Unpickler):
    PERMITIDOS = {("builtins", "dict"), ("builtins", "list"), ("builtins", "set"), ("builtins", "tuple"),
                  ("collections", "OrderedDict"), ("numpy", "ndarray"), ("numpy", "dtype"),
                  ("numpy._core.multiarray", "_reconstruct"), ("numpy.core.multiarray", "_reconstruct"),
                  ("numpy._core.multiarray", "scalar"), ("numpy.core.multiarray", "scalar")}

    def find_class(self, modulo, nome):
        if (modulo, nome) in self.PERMITIDOS:
            return super().find_class(modulo, nome)
        raise pickle.UnpicklingError(f"class not allowed: {modulo}.{nome}")


def ler_registros(caminho):
    with open(caminho, "rb") as f:
        return Seguro(f).load()


def histogramas(registros):
    """(errors, correct) histograms per record, both in increasing confidence."""
    h = np.array([r["h"] for r in registros], dtype=np.int64)
    return h[:, :BINS_METADE][:, ::-1], h[:, BINS_METADE:]


def bin_do_limiar(he_val, hc_val, alfa=0.10):
    """Largest bin k such that the correct pixels in bins >= k still cover at least ceil((n + 1)(1 - alfa))
    of the n validation pixels. Errors are never covered when tau > 0.5."""
    n = int(he_val.sum() + hc_val.sum())
    precisa = math.ceil((n + 1) * (1 - alfa))
    cobertos = np.cumsum(hc_val[::-1])[::-1]
    return int(np.flatnonzero(cobertos >= precisa).max())


def main(repo, caminho_fonte, caminho_bolivia, B=N.BOOT_B):
    ref = json.load(open(os.path.join(AQUI, "r0_referencia.json")))
    fonte = ler_registros(caminho_fonte)
    bolivia = ler_registros(caminho_bolivia)
    teste = set(A.ler_lista(repo, "sen1floods11_test.txt"))
    val = set(A.ler_lista(repo, "sen1floods11_val.txt"))
    r_val = [r for r in fonte if A.chave_de(r["nome"]) in val]
    r_teste = [r for r in fonte if A.chave_de(r["nome"]) in teste]
    assert len(r_val) == 86 and len(r_teste) == 89 and len(bolivia) == 15, (len(r_val), len(r_teste), len(bolivia))
    assert len(r_val) + len(r_teste) == len(fonte)

    ev, cv = histogramas(r_val)
    et, ct = histogramas(r_teste)
    eb, cb = histogramas(bolivia)
    k = bin_do_limiar(ev.sum(0), cv.sum(0))
    print(f"threshold bin from the validation histograms: {k}")

    def metricas(e, c):
        auroc = float(N.auroc_from_hist(e.sum(0), c.sum(0)))
        nao_sinalizado = float(e.sum(0)[k:].sum() / e.sum())
        taxa_erro = float(e.sum() / (e.sum() + c.sum()))
        return auroc, nao_sinalizado, taxa_erro

    ok = True
    for nome, (e, c) in (("teste", (et, ct)), ("bolivia", (eb, cb))):
        auroc, ns, erro = metricas(e, c)
        for chave, valor in (("auroc_e", auroc), ("erro_nao_sinalizado", ns), ("taxa_erro", erro)):
            r = ref[nome][chave]
            bate = round(valor, 3) == round(r, 3)
            ok &= bate
            print(f"{nome:8s} {chave:20s} script {valor:.6f}  week 1 {r:.6f}  third decimal: {'ok' if bate else 'DIFFERENT'}")

    # stratified standard error of Delta_shift for R0
    eventos = sorted({A.chave_de(r["nome"]).split("_")[0] for r in r_teste})
    est_s = [np.flatnonzero([A.chave_de(r["nome"]).split("_")[0] == e for r in r_teste]) for e in eventos]
    est_t = [np.arange(len(bolivia))]
    rng = R.make_rng()
    cnt_s = R.stratified_counts(rng, est_s, len(r_teste), B)
    cnt_t = R.stratified_counts(rng, est_t, len(bolivia), B)
    a_s, _ = R.hist_metrics(cnt_s, et, ct)
    a_t, _ = R.hist_metrics(cnt_t, eb, cb)
    d = a_t - a_s
    print(f"R0, Bolivia against source test: Delta_shift {metricas(eb, cb)[0] - metricas(et, ct)[0]:+.4f}, "
          f"stratified SE {np.std(d, ddof=1):.4f} (source {np.std(a_s, ddof=1):.4f}, Bolivia {np.std(a_t, ddof=1):.4f}), "
          f"90% interval [{np.quantile(d, 0.05):+.4f}, {np.quantile(d, 0.95):+.4f}], B = {B}")
    print("R0 reproduction: " + ("passed" if ok else "FAILED"))
    return ok


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(2)
    sys.exit(0 if main(*sys.argv[1:4]) else 1)
