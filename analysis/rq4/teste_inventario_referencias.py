#!/usr/bin/env python3
"""Tests of inventario_referencias.py on small synthetic layers. Usage: python analysis/rq4/teste_inventario_referencias.py"""
import json
import os
import shutil
import sys
import tempfile

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import inventario_referencias as I  # noqa: E402


def main():
    import geopandas as gpd
    from shapely.geometry import box, Point

    raiz = tempfile.mkdtemp(prefix="inv_")
    try:
        caixas = I.caixas_tiles(os.path.join(AQUI, "entradas", "rq4_entradas.json"))
        assert caixas["22JDM"] == (399960.0, 6590200.0, 509760.0, 6700000.0), caixas["22JDM"]
        assert caixas["22JCN"] == (300000.0, 6690220.0, 409800.0, 6800020.0), caixas["22JCN"]
        assert caixas["22JDN"] == (399960.0, 6690220.0, 509760.0, 6800020.0), caixas["22JDN"]
        assert I.datas_no_nome("rhguaiba_planet-skysat_inundacao_obs_20240506") == ["2024-05-06"]
        assert I.datas_no_nome("x_20240506_20240508_v12345678") == ["2024-05-06", "2024-05-08"]

        # a 2 x 3 km water polygon fully inside 22JDM, a 4 x 1 km mud polygon half in 22JDM and half outside, in UTM
        agua = box(450000, 6650000, 452000, 6653000)
        lama = box(507760, 6600000, 511760, 6601000)
        g = gpd.GeoDataFrame({"classe": ["agua", "lama"]}, geometry=[agua, lama], crs="EPSG:32722")
        caminho = os.path.join(raiz, "ref.gpkg")
        g.to_file(caminho, layer="mancha_obs_20240506", driver="GPKG")
        # the same water polygon in lat/lon, in another layer of the same file, and an empty layer
        g.iloc[:1].to_crs("EPSG:4326").to_file(caminho, layer="agua_wgs84_20240508", driver="GPKG")
        gpd.GeoDataFrame({"classe": []}, geometry=[], crs="EPSG:32722").to_file(caminho, layer="vazia", driver="GPKG")
        # a table without geometry, like P00_data_source of a Copernicus EMS geodatabase
        import pandas as pd
        import pyogrio
        pyogrio.write_dataframe(pd.DataFrame({"sensor": ["Sentinel-2", "Pleiades"], "data": ["08/05/2024", "07/05/2024"]}),
                                caminho, layer="P00_data_source", driver="GPKG")
        # a folder with a geojson of points (zero area) in 22JCN
        os.makedirs(os.path.join(raiz, "pasta", "sub"))
        gpd.GeoDataFrame({"tipo": ["a", "b", "a"]}, geometry=[Point(350000, 6750000)] * 3, crs="EPSG:32722").to_file(
            os.path.join(raiz, "pasta", "sub", "pontos_06052024.geojson"), driver="GeoJSON")
        with open(os.path.join(raiz, "pasta", "quebrado.geojson"), "w") as f:
            f.write("{not json")

        linhas, valores = I.inventariar([("ufrgs", caminho), ("emsn", os.path.join(raiz, "pasta"))], caixas, base=raiz)
        por = {(l["fonte"], l["camada"]): l for l in linhas}
        m = por[("ufrgs", "mancha_obs_20240506")]
        assert m["n_feicoes"] == 2 and m["datas_no_nome"] == "2024-05-06"
        assert abs(m["area_km2"] - 10.0) < 1e-6, m["area_km2"]
        assert abs(m["area_km2_22JDM"] - 8.0) < 1e-6, m["area_km2_22JDM"]          # 6 water + 2 of mud
        assert m["area_km2_22JCN"] == 0.0
        a = por[("ufrgs", "agua_wgs84_20240508")]
        assert abs(a["area_km2_22JDM"] - 6.0) < 0.01 and a["crs_original"].endswith("4326"), a
        assert por[("ufrgs", "vazia")]["n_feicoes"] == 0
        t = por[("ufrgs", "P00_data_source")]
        assert t["n_feicoes"] == 2 and t["tipos_geometria"] == "tabela sem geometria" and not t.get("erro"), t
        vt = [v for k, v in valores.items() if k.endswith("P00_data_source")][0]
        assert vt["_linhas"][0] == {"sensor": "Sentinel-2", "data": "08/05/2024"}, vt
        p = [l for l in linhas if l["fonte"] == "emsn" and l.get("camada")]
        assert len(p) == 1 and p[0]["n_feicoes"] == 3 and p[0]["area_km2"] == 0.0
        assert p[0]["datas_no_arquivo"] == "" and p[0]["datas_no_nome"] == ""      # 06052024 is not YYYYMMDD
        assert I.datas_no_nome("sub/EMSN194_AOI01_DEL_observedEvent_20240508.geojson") == ["2024-05-08"]
        assert any(l["fonte"] == "emsn" and "erro" in l and l["erro"] for l in linhas), "a broken file was not recorded"
        chave = [k for k in valores if k.endswith("mancha_obs_20240506")][0]
        assert valores[chave]["classe"] == {"agua": 1, "lama": 1}, valores[chave]

        I.escrever(linhas, valores, caixas, os.path.join(raiz, "saida"))
        with open(os.path.join(raiz, "saida", "inventario_referencias.csv"), encoding="utf-8") as f:
            texto = f.read()
        assert "area_km2_22JDM" in texto and "mancha_obs_20240506" in texto
        json.load(open(os.path.join(raiz, "saida", "valores_referencias.json"), encoding="utf-8"))
        print("all tests passed")
    finally:
        shutil.rmtree(raiz, ignore_errors=True)


if __name__ == "__main__":
    main()
