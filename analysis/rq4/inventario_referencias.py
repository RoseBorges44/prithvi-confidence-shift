#!/usr/bin/env python3
"""Inventory of the candidate references of RQ4 (prereg/plano_rq4.md, section 2), before any of them is used.

For every vector layer of a reference file it records the number of features, geometry type, CRS, the bounds and
the area in the UTM grid of the tiles (EPSG:32722), the area inside each tile, the dates written in the layer name
and in the file name (a GeoJSON layer is named by its "name" member, not by its file),
the columns and, for columns with few distinct values, those values with their counts. Nothing is rasterized and no
model output is read: the inventory only shows which layers exist, of which date, with which classes and where.

    python analysis/rq4/inventario_referencias.py --entradas analysis/rq4/entradas/rq4_entradas.json \
        --saida OUT --arquivo ufrgs=PATH.gpkg --arquivo emsn194=DIR_OR_FILE ...
"""
import argparse
import csv
import json
import os
import re

import numpy as np

CRS_TILES = "EPSG:32722"
EXTENSOES = (".gpkg", ".geojson", ".json", ".shp", ".kml", ".gdb")
MAX_VALORES = 30
MAX_LINHAS_TABELA = 500


def caixas_tiles(caminho_entradas):
    """Bounds (xmin, ymin, xmax, ymax) of each granule grid in EPSG:32722, from rq4_entradas.json."""
    with open(caminho_entradas, encoding="utf-8") as f:
        j = json.load(f)
    caixas = {}
    for e in j["entradas"]:
        g = e["grade"]
        if g["crs"] != CRS_TILES:
            raise ValueError(f"{e['tile']}: CRS {g['crs']}, expected {CRS_TILES}")
        a, _, x0, _, ee, y0 = g["transform"]
        caixas[e["tile"]] = (x0, y0 + ee * g["altura"], x0 + a * g["largura"], y0)
    return caixas


def datas_no_nome(nome):
    """Dates written as YYYYMMDD (2015 to 2029) in a layer or file name, as YYYY-MM-DD."""
    achados = re.findall(r"(?<!\d)(20[12]\d)(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(?!\d)", nome)
    return sorted({f"{a}-{m}-{d}" for a, m, d in achados})


def nome_geometria(df):
    """Name of the active geometry column, or None for a table without geometry."""
    try:
        return df.geometry.name
    except AttributeError:
        return None


def resumo_valores(gdf):
    """Columns with at most MAX_VALORES distinct values (geometry excluded): value -> count."""
    out = {}
    geom = nome_geometria(gdf)
    for col in gdf.columns:
        if col == geom:
            continue
        try:
            cont = gdf[col].astype("string").fillna("<vazio>").value_counts()
        except Exception:
            continue
        if 0 < len(cont) <= MAX_VALORES:
            out[col] = {str(k): int(v) for k, v in cont.items()}
    return out


def inventariar_tabela(df, fonte, arquivo, camada):
    """One row of the inventory for a table without geometry (e.g. the data-source table of a Copernicus EMS
    geodatabase). Tables of up to MAX_LINHAS_TABELA rows are kept whole in the values file."""
    linha = {"fonte": fonte, "arquivo": arquivo, "camada": camada, "datas_no_nome": ";".join(datas_no_nome(camada)),
             "datas_no_arquivo": ";".join(datas_no_nome(arquivo)), "n_feicoes": int(len(df)),
             "tipos_geometria": "tabela sem geometria", "colunas": ";".join(map(str, df.columns))}
    val = resumo_valores(df)
    if len(df) <= MAX_LINHAS_TABELA:
        val["_linhas"] = df.astype("string").fillna("").to_dict("records")
    return linha, val


def inventariar_camada(gdf, fonte, arquivo, camada, caixas):
    """One row of the inventory for a GeoDataFrame. A failure of the geometric part is recorded in `erro` and does
    not lose the counts, columns and values."""
    import shapely
    from shapely.geometry import box

    geom_nome = nome_geometria(gdf)
    linha = {"fonte": fonte, "arquivo": arquivo, "camada": camada, "datas_no_nome": ";".join(datas_no_nome(camada)),
             "datas_no_arquivo": ";".join(datas_no_nome(arquivo)),
             "n_feicoes": int(len(gdf)), "crs_original": str(gdf.crs) if gdf.crs is not None else "",
             "tipos_geometria": ";".join(sorted({str(t) for t in gdf.geom_type.dropna().unique()})),
             "colunas": ";".join(c for c in gdf.columns if c != geom_nome)}
    vazio = {"xmin": "", "ymin": "", "xmax": "", "ymax": "", "area_km2": ""}
    vazio.update({f"area_km2_{t}": "" for t in caixas})
    linha.update(vazio)
    if len(gdf) == 0 or gdf.crs is None:
        return linha, resumo_valores(gdf)
    try:
        g = gdf.to_crs(CRS_TILES)
        geom = np.asarray(g.geometry.values)
        try:
            geom = shapely.make_valid(geom)
        except Exception:
            pass
        xmin, ymin, xmax, ymax = shapely.total_bounds(geom)
        area = float(shapely.area(geom).sum()) / 1e6
        linha.update({"xmin": round(float(xmin), 1), "ymin": round(float(ymin), 1), "xmax": round(float(xmax), 1),
                      "ymax": round(float(ymax), 1), "area_km2": round(area, 4)})
        for t, c in caixas.items():
            linha[f"area_km2_{t}"] = round(float(shapely.area(shapely.intersection(geom, box(*c))).sum()) / 1e6, 4)
    except Exception as e:
        linha["erro"] = f"{type(e).__name__}: {e}"
    return linha, resumo_valores(gdf)


def camadas_do_caminho(caminho):
    """(file, layer) pairs for a file or a folder (walked), for the vector formats read by pyogrio."""
    import pyogrio

    arquivos = []
    if os.path.isdir(caminho) and not caminho.lower().endswith(".gdb"):
        for raiz, dirs, nomes in os.walk(caminho):
            for d in list(dirs):
                if d.lower().endswith(".gdb"):
                    arquivos.append(os.path.join(raiz, d))
                    dirs.remove(d)
            arquivos += [os.path.join(raiz, n) for n in nomes if n.lower().endswith(EXTENSOES[:-1])]
    else:
        arquivos = [caminho]
    pares = []
    for a in sorted(arquivos):
        try:
            camadas = [str(c[0]) for c in pyogrio.list_layers(a)]
        except Exception as e:  # unreadable file: recorded, not skipped silently
            pares.append((a, None, f"{type(e).__name__}: {e}"))
            continue
        pares += [(a, c, "") for c in camadas]
    return pares


def inventariar(fontes, caixas, base=None):
    """fontes: list of (name, path). Returns (rows, values) for every layer."""
    import geopandas as gpd

    linhas, valores = [], {}
    for fonte, caminho in fontes:
        for arq, camada, erro in camadas_do_caminho(caminho):
            rel = os.path.relpath(arq, base) if base else os.path.basename(arq)
            if camada is None:
                linhas.append({"fonte": fonte, "arquivo": rel, "camada": "", "erro": erro})
                continue
            try:
                gdf = gpd.read_file(arq, layer=camada)
                if nome_geometria(gdf) is None:
                    linha, val = inventariar_tabela(gdf, fonte, rel, camada)
                else:
                    linha, val = inventariar_camada(gdf, fonte, rel, camada, caixas)
            except Exception as e:
                linha, val = {"fonte": fonte, "arquivo": rel, "camada": camada, "erro": f"{type(e).__name__}: {e}"}, {}
            linhas.append(linha)
            valores[f"{fonte} | {rel} | {camada}"] = val
    return linhas, valores


def escrever(linhas, valores, caixas, saida):
    os.makedirs(saida, exist_ok=True)
    campos = (["fonte", "arquivo", "camada", "datas_no_nome", "datas_no_arquivo", "n_feicoes", "tipos_geometria", "crs_original",
               "xmin", "ymin", "xmax", "ymax", "area_km2"] + [f"area_km2_{t}" for t in caixas] + ["colunas", "erro"])
    with open(os.path.join(saida, "inventario_referencias.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos, extrasaction="ignore")
        w.writeheader()
        for l in linhas:
            w.writerow({k: l.get(k, "") for k in campos})
    with open(os.path.join(saida, "valores_referencias.json"), "w", encoding="utf-8") as f:
        json.dump(valores, f, indent=1, ensure_ascii=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entradas", required=True, help="rq4_entradas.json")
    ap.add_argument("--saida", required=True)
    ap.add_argument("--arquivo", action="append", default=[], help="name=path (file or folder); repeat")
    args = ap.parse_args(argv)
    caixas = caixas_tiles(args.entradas)
    fontes = [tuple(a.split("=", 1)) for a in args.arquivo]
    linhas, valores = inventariar(fontes, caixas)
    escrever(linhas, valores, caixas, args.saida)
    return linhas, valores


if __name__ == "__main__":
    main()
