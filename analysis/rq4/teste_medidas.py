#!/usr/bin/env python3
"""Tests of medidas.py. Usage: python analysis/rq4/teste_medidas.py"""
import os
import sys

import numpy as np
from scipy.special import expit

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import inferencia as I  # noqa: E402
import mascaras as M  # noqa: E402
import medidas as D  # noqa: E402
import nucleo as N  # noqa: E402

H, W = 1100, 1030


def cenario(seed=0):
    """Small tile: codes, references, strata, windows, labels file, margins on the grid and the window table."""
    rng = np.random.default_rng(seed)
    cods, nrefs = {}, {}
    for k, d in enumerate(M.DISTANCIAS_M):
        c = np.full((H, W), M.FORA, np.uint8)
        c[:, :300 + 100 * k] = M.TERRA
        c[:, 100:200] = M.AGUA
        c[rng.random((H, W)) < 0.05] = M.BORDA
        cods[d] = c
    est = rng.integers(1, 4, (H, W)).astype(np.uint8)
    nr = rng.integers(1, 3, (H, W)).astype(np.uint8)
    nrefs = {d: nr for d in M.DISTANCIAS_M}
    linhas = []
    for d in M.DISTANCIAS_M:
        for j in M.janelas(cods[d], "T", min_pontuados=1):
            linhas.append({"tile": "T", "distancia_m": str(d), "window_id": j["window_id"], "row_off": j["row_off"],
                           "col_off": j["col_off"], "kept": "True"})
    jan = I.janelas_da_inferencia(linhas, "T")
    novos = I.atribuidos(H, W, jan)
    rot = I.rotulos_tile(cods, nrefs, est, jan, novos)
    m = rng.normal(0, 3, (H, W)).astype(np.float32)
    return cods, nr, est, rot, m, linhas, jan, novos


def margens_do_arquivo(m, jan, novos, rot):
    todas = np.concatenate([I.recortar(m, l, c, n) for (_, l, c), n in zip(jan, novos)])
    return todas, todas[rot["pontuado"]]


def teste_contagens_batem_com_a_conta_na_grade():
    cods, nr, est, rot, m, linhas, jan, novos = cenario()
    todas, margem = margens_do_arquivo(m, jan, novos, rot)
    tau = 0.93
    bins = N.Bins()
    res = D.calcular(lambda r, s, t: margem, {("R1", 1): tau}, ["T"], {"T": rot}, linhas, ["R1"], [1], bins,
                     aviso=lambda *_: None)
    assert res["dif_hist"] < 1e-4, res["dif_hist"]
    # only assigned pixels of windows that enter; brute force on the grid
    atrib = np.zeros((H, W), bool)
    for (_, l, c), n in zip(jan, novos):
        atrib[l:l + 512, c:c + 512] |= n
    for d in M.DISTANCIAS_M:
        cont, he, hc = res["contagens"][("R1", 1, "T", d)]
        pont = (cods[d] <= M.AGUA) & atrib
        y = cods[d] == M.AGUA
        disc = (m > 0) != y
        flag = expit(np.abs(m)) < tau
        assert int(cont[:, 0].sum()) == int(pont.sum())
        for tipo, nome, cel in D.GRUPOS:
            mk = np.zeros((H, W), bool)
            for c in cel:
                mk |= pont & (((est.astype(int) - 1) * 2 + (nr.astype(int) - 1)) == c)
            r = D.medidas_do_grupo(cont, he, hc, cel)
            assert r["pixels"] == int(mk.sum()), (d, nome)
            assert r["discordancias"] == int((mk & disc).sum())
            assert r["sinalizados"] == int((mk & flag).sum())
            assert r["discordancias_nao_sinalizadas"] == int((mk & disc & ~flag).sum())
            esperado = N.auroc_exact(np.abs(m)[mk & disc], np.abs(m)[mk & ~disc])
            assert abs(r["auroc_d"] - esperado) < 1e-4, (d, nome, r["auroc_d"], esperado)
            assert r["modelo_agua_referencia_terra"] == int((mk & (m > 0) & ~y).sum())
            assert r["modelo_terra_referencia_agua"] == int((mk & (m <= 0) & y).sum())
            assert r["modelo_agua_referencia_terra"] + r["modelo_terra_referencia_agua"] == r["discordancias"]
            assert r["referencia_agua"] == int((mk & y).sum())
            assert abs(r["fracao_agua_referencia"] - (mk & y).sum() / mk.sum()) < 1e-12
            assert abs(r["discordancia_modelo_agua"] - (mk & (m > 0) & ~y).sum() / mk.sum()) < 1e-12
            assert abs(r["discordancia_nao_sinalizada"] - (mk & disc & ~flag).sum() / (mk & disc).sum()) < 1e-12
            assert abs(r["conjunto_vazio"] - (mk & flag).sum() / mk.sum()) < 1e-12
            assert abs(r["discordancia_entre_sinalizados"] - (mk & flag & disc).sum() / (mk & flag).sum()) < 1e-12
            assert abs(r["discordancia_entre_nao_sinalizados"] - (mk & ~flag & disc).sum() / (mk & ~flag).sum()) < 1e-12


def teste_janela_que_nao_entra_fica_fora():
    cods, nr, est, rot, m, linhas, jan, novos = cenario()
    _, margem = margens_do_arquivo(m, jan, novos, rot)
    sai = next(r for r in linhas if r["distancia_m"] == "500")
    sai["kept"] = "False"
    s0 = D.selecionar(rot, 1, D.janelas_que_entram(linhas, "T", 500))
    s1 = D.selecionar(rot, 1, D.janelas_que_entram([r for r in linhas if r is not sai], "T", 500))
    assert s0["idx"].size == s1["idx"].size
    todas = D.selecionar(rot, 1, {str(w) for w in rot["janelas"]})
    assert todas["idx"].size > s0["idx"].size


def teste_soma_dos_tiles_e_tabelas():
    cods, nr, est, rot, m, linhas, jan, novos = cenario()
    _, margem = margens_do_arquivo(m, jan, novos, rot)
    tau = {(r, s): 0.9 + 0.01 * s for r in ("R1", "R2") for s in (1, 2)}
    rot2 = {"T": rot, "U": rot}
    linhas2 = linhas + [dict(r, tile="U") for r in linhas]
    res = D.calcular(lambda r, s, t: margem if r == "R1" else -margem, tau, ["T", "U"], rot2, linhas2,
                     ["R1", "R2"], [1, 2], N.Bins(), aviso=lambda *_: None)
    ex = D.tabela_execucoes(res["contagens"], ["T", "U"], ["R1", "R2"], [1, 2], [1000, 500, 2000])
    assert len(ex) == 2 * 2 * 3 * 3 * len(D.GRUPOS)
    um = next(r for r in ex if (r["regime"], r["semente"], r["escopo"], r["distancia_m"], r["grupo"])
              == ("R1", 1, "T", 1000, "todos"))
    ag = next(r for r in ex if (r["regime"], r["semente"], r["escopo"], r["distancia_m"], r["grupo"])
              == ("R1", 1, D.AGREGADO, 1000, "todos"))
    assert ag["pixels"] == 2 * um["pixels"] and ag["discordancias"] == 2 * um["discordancias"]
    assert abs(ag["taxa_discordancia"] - um["taxa_discordancia"]) < 1e-12
    # R2 has the opposite sign: disagreement rate is 1 - rate (no pixel at exactly 0)
    dois = next(r for r in ex if (r["regime"], r["semente"], r["escopo"], r["distancia_m"], r["grupo"])
                == ("R2", 1, "T", 1000, "todos"))
    assert abs(um["taxa_discordancia"] + dois["taxa_discordancia"] - 1) < 1e-12
    # groups: strata and references each add up to all
    for escopo in ("T", D.AGREGADO):
        for tipo in ("estrato", "referencias"):
            s = sum(r["pixels"] for r in ex if (r["regime"], r["semente"], r["escopo"], r["distancia_m"],
                                                  r["tipo_grupo"]) == ("R1", 1, escopo, 1000, tipo))
            assert s == next(r for r in ex if (r["regime"], r["semente"], r["escopo"], r["distancia_m"],
                                               r["grupo"]) == ("R1", 1, escopo, 1000, "todos"))["pixels"]
    alvo = D.ler_alvo([
        {"regime": "R1", "conjunto": "alvo", "metrica": "taxa_erro", "media_sementes": "0.03"},
        {"regime": "R1", "conjunto": "alvo|A", "metrica": "taxa_erro", "media_sementes": "0.01"},
        {"regime": "R1", "conjunto": "alvo|B", "metrica": "taxa_erro", "media_sementes": "0.05"},
        {"regime": "R1", "conjunto": "teste", "metrica": "taxa_erro", "media_sementes": "0.9"},
        {"regime": "R1", "conjunto": "alvo", "metrica": "miou", "media_sementes": "0.8"}])
    assert alvo == {("R1", "taxa_discordancia"): {"agregado": 0.03, "eventos": {"A": 0.01, "B": 0.05}}}
    reg = D.tabela_regimes(ex, alvo)
    r = next(x for x in reg if (x["regime"], x["escopo"], x["distancia_m"], x["grupo"], x["medida"])
             == ("R1", "T", 1000, "todos", "taxa_discordancia"))
    assert r["sementes"] == 2 and r["alvo_agregado"] == 0.03 and r["posicao_nos_eventos"] == "acima da faixa"
    r2 = next(x for x in reg if (x["regime"], x["escopo"], x["distancia_m"], x["grupo"], x["medida"])
              == ("R1", "T", 1000, "construído", "taxa_discordancia"))
    assert r2["alvo_agregado"] == ""
    assert D.posicao(0.03, {"A": 0.01, "B": 0.05}) == "dentro da faixa"
    assert D.posicao(0.0, {"A": 0.01, "B": 0.05}) == "abaixo da faixa"
    cel = D.tabela_celulas(res["contagens"], ["T", "U"], ["R1", "R2"], [1, 2], [1000, 500, 2000])
    assert len(cel) == 2 * 2 * 2 * 3 * 6


def teste_histograma_fora_da_tolerancia_e_detectado():
    cods, nr, est, rot, m, linhas, jan, novos = cenario()
    _, margem = margens_do_arquivo(m, jan, novos, rot)
    res = D.calcular(lambda r, s, t: margem, {("R1", 1): 0.9}, ["T"], {"T": rot}, linhas, ["R1"], [1],
                     N.Bins(width=2.0, vmax=40.0), aviso=lambda *_: None)
    assert res["dif_hist"] > 1e-4
    assert res["max_abs_m"] == float(np.abs(margem).max())


def teste_dados_inconsistentes_param():
    cods, nr, est, rot, m, linhas, jan, novos = cenario()
    ruim = dict(rot)
    ruim["nrefs"] = rot["nrefs"].copy()
    ruim["nrefs"][0][ruim["codigo"][0] <= M.AGUA] = 3
    try:
        D.selecionar(ruim, 0, {str(w) for w in rot["janelas"]})
    except ValueError:
        return
    raise AssertionError("3 references were accepted")


def teste_mapas():
    cods, nr, est, rot, m, linhas, jan, novos = cenario()
    todas, margem = margens_do_arquivo(m, jan, novos, rot)
    b = D.blocos_do_tile(I.quantizar(todas), margem, rot, 0.93, altura=H, largura=W)
    assert b["msp"].shape == (H // 10, W // 10)
    atrib = np.zeros((H, W), bool)
    for (_, l, c), n in zip(jan, novos):
        atrib[l:l + 512, c:c + 512] |= n
    # first block of the grid: mean MSP of its assigned pixels (exact margins on scored pixels, quantized elsewhere)
    esperado = np.full((H, W), np.nan, np.float32)
    pont = (cods[1000] <= M.AGUA) | (cods[500] <= M.AGUA) | (cods[2000] <= M.AGUA)
    q = I.quantizar(m).astype(np.float32) / I.ESCALA_Q
    esperado[atrib] = np.where(pont, m, q)[atrib]
    blk = esperado[:10, :10]
    ok = ~np.isnan(blk)
    assert ok.any()
    assert abs(b["msp"][0, 0] - expit(np.abs(blk[ok])).mean()) < 1e-5
    assert abs(b["sinalizado"][0, 0] - (expit(np.abs(blk[ok])) < 0.93).mean()) < 1e-5
    y = cods[1000] == M.AGUA
    sc = (cods[1000] <= M.AGUA) & atrib
    disc = sc & ((m > 0) != y)
    i, j = np.unravel_index(np.argmax(sc[:10 * (H // 10), :10 * (W // 10)].reshape(H // 10, 10, W // 10, 10).sum((1, 3))), (H // 10, W // 10))
    s = sc[i * 10:i * 10 + 10, j * 10:j * 10 + 10]
    assert abs(b["discordancia"][i, j] - disc[i * 10:i * 10 + 10, j * 10:j * 10 + 10].sum() / s.sum()) < 1e-6
    assert np.isnan(b["discordancia"][sc.reshape(H // 10, 10, W // 10, 10).sum((1, 3)) == 0]).all()


def teste_csv(tmp="/tmp/teste_medidas_csv"):
    os.makedirs(tmp, exist_ok=True)
    p = f"{tmp}/x.csv"
    D.escrever_csv(p, [{"a": 1, "b": float("nan"), "c": 0.123456789012, "d": "x"}])
    assert open(p).read().splitlines() == ["a,b,c,d", "1,,0.12345679,x"]


if __name__ == "__main__":
    for nome, f in sorted(globals().items()):
        if nome.startswith("teste_") and callable(f):
            f()
            print("ok", nome)
