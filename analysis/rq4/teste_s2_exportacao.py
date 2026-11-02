#!/usr/bin/env python3
"""Tests of s2_exportacao.py. Usage: python analysis/rq4/teste_s2_exportacao.py"""
import os
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import s2_exportacao as S  # noqa: E402


def ms(texto):
    import datetime as dt
    return int(dt.datetime.fromisoformat(texto).replace(tzinfo=dt.timezone.utc).timestamp() * 1000)


def teste_escolha():
    cand = [
        {"id": "b", "tile": "22JDM", "time_start_ms": ms("2024-05-06T13:32:29"), "nuvem": 36.0},
        {"id": "a", "tile": "22JDM", "time_start_ms": ms("2024-05-06T13:32:31"), "nuvem": 36.0},
        {"id": "c", "tile": "22JDM", "time_start_ms": ms("2024-05-06T23:59:59"), "nuvem": 5.0},
        {"id": "d", "tile": "22JDM", "time_start_ms": ms("2024-05-07T00:00:00"), "nuvem": 0.0},     # next UTC day
        {"id": "e", "tile": "22JDM", "time_start_ms": ms("2024-05-05T23:59:59"), "nuvem": 0.0},     # previous UTC day
        {"id": "f", "tile": "22JCN", "time_start_ms": ms("2024-05-06T13:32:29"), "nuvem": 1.0},
    ]
    assert S.escolher_granulo(cand, "22JDM")["id"] == "c"          # lowest cloud on the day, in UTC
    assert S.escolher_granulo(cand[:2], "22JDM")["id"] == "a"       # tie: smaller id
    assert S.escolher_granulo(cand, "22JCN")["id"] == "f"
    for tile, lista in (("22JDM", cand[3:5]), ("22JCN", cand[:5])):
        try:
            S.escolher_granulo(lista, tile)
        except ValueError as e:
            assert "No granule" in str(e)
        else:
            raise AssertionError("an empty choice was not detected")
    print("choice: ok")


def teste_linha_base():
    S.conferir_linha_base("05.10")
    S.conferir_linha_base("04.00")
    S.conferir_linha_base("10.01")
    try:
        S.conferir_linha_base("03.01")
    except ValueError:
        pass
    else:
        raise AssertionError("an old baseline was accepted")
    print("baseline: ok")


def teste_grade():
    perfil = {"height": 10980, "width": 10980, "count": 6, "dtype": "uint16", "crs": "EPSG:32722",
              "transform": (10.0, 0.0, 300000.0, 0.0, -10.0, 6700020.0)}
    S.conferir_grade("22JDM", perfil, (300000.0, 6700020.0))
    for campo, valor, trecho in (("height", 10240, "size"), ("count", 5, "bands"), ("dtype", "int16", "type"),
                                 ("crs", "EPSG:32722 ", "CRS"), ("transform", (20.0, 0.0, 300000.0, 0.0, -20.0, 6700020.0), "pixel"),
                                 ("transform", (10.0, 0.0, 300010.0, 0.0, -10.0, 6700020.0), "origin")):
        ruim = dict(perfil, **{campo: valor})
        try:
            S.conferir_grade("22JDM", ruim, (300000.0, 6700020.0))
        except ValueError as e:
            assert trecho in str(e), (campo, e)
        else:
            raise AssertionError(f"{campo} wrong was accepted")
    print("grid: ok")


def teste_deslocamento():
    import numpy as np
    orig = (399960.0, 6700000.0)
    base = {"count": 6, "dtype": "uint16", "crs": "EPSG:32722", "height": 10980, "width": 9861,
            "transform": (10.0, 0.0, 399950.0, 0.0, -10.0, 6700010.0)}
    assert S.deslocamento_na_grade("s2", base, orig, "22JDM") == (-1, -1)           # the real 22JDM file
    assert S.deslocamento_na_grade("s2", dict(base, transform=(10.0, 0.0, 399960.0, 0.0, -10.0, 6700000.0)), orig,
                                   "22JDM") == (0, 0)
    assert S.deslocamento_na_grade("scl", dict(base, count=1, dtype="uint8", transform=(10.0, 0.0, 399970.0, 0.0, -10.0, 6699980.0)),
                                   orig, "22JDM") == (1, 2)
    assert S.deslocamento_na_grade("gsw", dict(base, count=1, dtype="uint8"), orig, "22JDM") == (-1, -1)    # auxiliary layers (GSW, WorldCover)
    assert S.deslocamento_na_grade("worldcover", dict(base, count=1, dtype="uint8"), orig, "22JDM") == (-1, -1)
    for ruim, trecho in ((dict(base, count=5), "bands"), (dict(base, dtype="int16"), "type"), (dict(base, crs="EPSG:4326"), "CRS"),
                         (dict(base, transform=(20.0, 0.0, 399950.0, 0.0, -20.0, 6700010.0)), "pixel"),
                         (dict(base, transform=(10.0, 0.0, 399955.0, 0.0, -10.0, 6700010.0)), "whole number"),
                         (dict(base, transform=(10.0, 0.0, 600000.0, 0.0, -10.0, 6700000.0)), "overlap")):
        try:
            S.deslocamento_na_grade("s2", ruim, orig, "22JDM")
        except ValueError as e:
            assert trecho in str(e), (trecho, e)
        else:
            raise AssertionError(f"{trecho} was accepted")
    # placing: smaller file, larger file, shifted file; values never change
    S_LADO = S.LADO
    S.LADO = 8
    try:
        d = np.arange(2 * 5 * 6, dtype=np.uint16).reshape(2, 5, 6) + 1
        g = S.colocar_na_grade(d, 1, 2, 0)
        assert g.shape == (2, 8, 8) and (g[:, 2:7, 1:7] == d).all() and g[:, :2].sum() == 0 and g[:, :, 0].sum() == 0
        g = S.colocar_na_grade(d, -1, -2, 255)                                        # starts before the grid
        assert (g[:, :3, :5] == d[:, 2:, 1:]).all() and (g[:, 3:] == 255).all() and (g[:, :, 5:] == 255).all()
        g = S.colocar_na_grade(np.ones((1, 9, 10), np.uint8), -1, -1, 0)              # larger than the grid
        assert g.shape == (1, 8, 8) and (g == 1).all()
        n_validos = int((d != 0).sum())
        assert int((S.colocar_na_grade(d, 3, 3, 0) != 0).sum()) < n_validos            # cropped on the far side
    finally:
        S.LADO = S_LADO
    print("placement: ok")


def teste_registro():
    cand = {"id": "x", "tile": "22JDM", "time_start_ms": ms("2024-05-06T13:32:29"), "nuvem": 12.5, "n_candidatos": 1}
    r = S.registro_entrada("22JDM", cand, {"PROCESSING_BASELINE": "05.10"}, {"l1c": "COPERNICUS/S2_HARMONIZED"},
                           {"s2": {"arquivo": "a.tif", "sha256": "0" * 64}}, {"height": 10980}, 0.001, {"situacao": "ok"})
    assert r["sensing_utc"] == "2024-05-06T13:32:29Z" and r["nuvem_percentual"] == 12.5 and r["n_candidatos_no_dia"] == 1
    print("record: ok")


if __name__ == "__main__":
    teste_escolha(); teste_linha_base(); teste_grade(); teste_deslocamento(); teste_registro()
    print("all tests passed")
