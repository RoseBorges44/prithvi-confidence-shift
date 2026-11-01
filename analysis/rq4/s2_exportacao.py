#!/usr/bin/env python3
"""Rules of the Sentinel-2 export of RQ4 (prereg/plano_rq4.md, section 1), without any call to Earth Engine.

The notebook notebooks/semana9_rq4_exportacao.ipynb asks Earth Engine for the granules and calls these functions
to pick the granule, to check the exported grid and to write the record of the inputs. Keeping the rules here lets
the tests run without an Earth Engine account.
"""
import datetime as dt

DATA = "2024-05-06"                      # sensing date, UTC
TILES = ("22JDM", "22JCN")
BANDAS = ("B2", "B3", "B4", "B8A", "B11", "B12")
LADO = 10980                             # pixels of a 10 m granule
PIXEL = 10.0
LINHA_BASE_MINIMA = "04.00"              # from this baseline on, L1C values carry an offset of 1000
CRS_ESPERADO = {"22JDM": "EPSG:32722", "22JCN": "EPSG:32722"}   # UTM zone 22 south


def dia_utc(ms):
    """UTC date (YYYY-MM-DD) of an Earth Engine time stamp in milliseconds."""
    return dt.datetime.fromtimestamp(ms / 1000.0, tz=dt.timezone.utc).strftime("%Y-%m-%d")


def escolher_granulo(candidatos, tile, data=DATA):
    """The granule of `tile` sensed on `data` (UTC) with the lowest cloud cover; the smaller id wins a tie.
    Each candidate is a dict with id, tile, time_start_ms and nuvem (CLOUDY_PIXEL_PERCENTAGE). Stops when none fits."""
    boas = [c for c in candidatos if c["tile"] == tile and dia_utc(c["time_start_ms"]) == data]
    if not boas:
        raise ValueError(f"No granule of tile {tile} sensed on {data} (UTC) among {len(candidatos)} candidates.")
    return sorted(boas, key=lambda c: (float(c["nuvem"]), c["id"]))[0]


def conferir_linha_base(linha_base):
    """Stops when the processing baseline is older than the one that added the offset: the harmonized collection
    would then have removed nothing and the check of the offset would not apply."""
    partes = tuple(int(x) for x in str(linha_base).split("."))
    minimas = tuple(int(x) for x in LINHA_BASE_MINIMA.split("."))
    if partes < minimas:
        raise ValueError(f"Processing baseline {linha_base} is older than {LINHA_BASE_MINIMA}.")


def conferir_grade(tile, perfil, origem_esperada):
    """The exported GeoTIFF must be the 10 m grid of the granule: 6 bands of 10980 x 10980 uint16, pixel 10 m,
    the CRS of the tile and the origin that Earth Engine reports for band B2. `perfil` has height, width, count,
    dtype, crs and transform (a, b, c, d, e, f). `origem_esperada` is (x0, y0)."""
    erros = []
    if (perfil["height"], perfil["width"]) != (LADO, LADO):
        erros.append(f"size {perfil['height']} x {perfil['width']}, expected {LADO} x {LADO}")
    if perfil["count"] != len(BANDAS):
        erros.append(f"{perfil['count']} bands, expected {len(BANDAS)}")
    if str(perfil["dtype"]) != "uint16":
        erros.append(f"type {perfil['dtype']}, expected uint16")
    if str(perfil["crs"]) != CRS_ESPERADO[tile]:
        erros.append(f"CRS {perfil['crs']}, expected {CRS_ESPERADO[tile]}")
    a, b, c, d, e, f = perfil["transform"]
    if (a, b, d, e) != (PIXEL, 0.0, 0.0, -PIXEL):
        erros.append(f"pixel {(a, b, d, e)}, expected {(PIXEL, 0.0, 0.0, -PIXEL)}")
    if (c, f) != tuple(origem_esperada):
        erros.append(f"origin {(c, f)}, expected {tuple(origem_esperada)}")
    if erros:
        raise ValueError(f"{tile}: " + "; ".join(erros))


def registro_entrada(tile, candidato, propriedades, colecoes, arquivos, grade, zeros, deslocamento):
    """The record of one input, as written to rq4_entradas.json. Everything that identifies the scene and the files."""
    return {
        "tile": tile,
        "id": candidato["id"],
        "sensing_utc": dt.datetime.fromtimestamp(candidato["time_start_ms"] / 1000.0, tz=dt.timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "nuvem_percentual": float(candidato["nuvem"]),
        "n_candidatos_no_dia": int(candidato["n_candidatos"]),
        "propriedades": propriedades,
        "colecoes": colecoes,
        "arquivos": arquivos,
        "grade": grade,
        "fracao_pixels_com_6_bandas_zero": float(zeros),
        "verificacao_do_deslocamento": deslocamento,
    }
