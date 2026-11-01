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


def teste_registro():
    cand = {"id": "x", "tile": "22JDM", "time_start_ms": ms("2024-05-06T13:32:29"), "nuvem": 12.5, "n_candidatos": 1}
    r = S.registro_entrada("22JDM", cand, {"PROCESSING_BASELINE": "05.10"}, {"l1c": "COPERNICUS/S2_HARMONIZED"},
                           {"s2": {"arquivo": "a.tif", "sha256": "0" * 64}}, {"height": 10980}, 0.001, {"situacao": "ok"})
    assert r["sensing_utc"] == "2024-05-06T13:32:29Z" and r["nuvem_percentual"] == 12.5 and r["n_candidatos_no_dia"] == 1
    print("record: ok")


if __name__ == "__main__":
    teste_escolha(); teste_linha_base(); teste_grade(); teste_registro()
    print("all tests passed")
