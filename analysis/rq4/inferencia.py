#!/usr/bin/env python3
"""Windows and pixel order of the inference of RQ4 (prereg/plano_rq4.md, section 1, "Windows" and "Inference").

The model is run on every window that enters at any of the three distances of the UFRGS mapped area (the 2 km set,
since every window that enters at 500 m or 1 km also enters at 2 km). Inside a window only the pixels assigned to it
are kept (each pixel of the tile belongs to the first window, in row-major order, that covers it; see
mascaras.percorrer_janelas). The pixel order of every array written by notebooks/semana9_rq4_inferencia.ipynb is:
windows by (row, column), then the assigned pixels of each window in row-major order. The labels file of a tile holds
the code, the number of covering references and the stratum of those pixels; the run files hold the margins.
"""
import numpy as np

import mascaras as M

ESCALA_Q = 4.0                           # quantized margin for maps: round(4 m), clipped to the int8 range


def janelas_da_inferencia(linhas, tile):
    """Windows of `tile` that enter at any distance, from the rows of rq4_janelas.csv, sorted by (row, column).
    Returns a list of (window_id, row_off, col_off)."""
    entram = {}
    for r in linhas:
        if r["tile"] == tile and str(r["kept"]) == "True":
            entram[r["window_id"]] = (int(r["row_off"]), int(r["col_off"]))
    return sorted(((w, l, c) for w, (l, c) in entram.items()), key=lambda x: (x[1], x[2]))


def atribuidos(altura, largura, janelas, tam=M.TAM):
    """Assigned-pixel mask (tam x tam) of each window of `janelas`, in the order of `janelas`."""
    alvo = {(l, c): i for i, (_, l, c) in enumerate(janelas)}
    if len(alvo) != len(janelas):
        raise ValueError("repeated window")
    out = [None] * len(janelas)
    for lin, col, novo in M.percorrer_janelas(altura, largura, tam):
        i = alvo.get((lin, col))
        if i is not None:
            out[i] = novo
    falta = [janelas[i][0] for i, v in enumerate(out) if v is None]
    if falta:
        raise ValueError(f"windows outside the grid: {falta[:3]}")
    return out


def recortar(grade, lin, col, novo, tam=M.TAM):
    """Assigned pixels of a window, in row-major order, from a 2-D grid (or from the last two axes of a 3-D one)."""
    return grade[..., lin:lin + tam, col:col + tam][..., novo]


def rotulos_tile(codigos, nrefs, estrato, janelas, novos, distancias=M.DISTANCIAS_M):
    """Labels of the assigned pixels of the windows of a tile.

    codigos, nrefs: {distance: 2-D grid}; estrato: 2-D grid. Returns a dict of arrays: janelas, linha, coluna,
    tamanhos (assigned pixels per window), distancias, codigo (len(distancias), N), nrefs (same), estrato (N),
    pontuado (N, scored at some distance) and tamanhos_pontuados (scored pixels per window)."""
    cod = [np.concatenate([recortar(codigos[d], l, c, n) for (_, l, c), n in zip(janelas, novos)]) for d in distancias]
    nr = [np.concatenate([recortar(nrefs[d], l, c, n) for (_, l, c), n in zip(janelas, novos)]) for d in distancias]
    est = np.concatenate([recortar(estrato, l, c, n) for (_, l, c), n in zip(janelas, novos)])
    cod, nr = np.stack(cod).astype(np.uint8), np.stack(nr).astype(np.uint8)
    pont = (cod <= M.AGUA).any(axis=0)
    tam = np.array([int(n.sum()) for n in novos], np.int64)
    limites = np.concatenate([[0], np.cumsum(tam)])
    tam_p = np.array([int(pont[limites[i]:limites[i + 1]].sum()) for i in range(len(janelas))], np.int64)
    return {"janelas": np.array([w for w, _, _ in janelas]), "linha": np.array([l for _, l, _ in janelas], np.int32),
            "coluna": np.array([c for _, _, c in janelas], np.int32), "tamanhos": tam,
            "distancias": np.array(distancias, np.int32), "codigo": cod, "nrefs": nr, "estrato": est.astype(np.uint8),
            "pontuado": pont, "tamanhos_pontuados": tam_p}


def conferir_com_tabela(rot, linhas, tile):
    """The scored, water and land pixels of each window at each distance must equal those of rq4_janelas.csv.
    Returns the number of (window, distance) pairs checked."""
    tab = {(r["window_id"], int(r["distancia_m"])): r for r in linhas if r["tile"] == tile}
    limites = np.concatenate([[0], np.cumsum(rot["tamanhos"])])
    n = 0
    for i, w in enumerate(rot["janelas"]):
        for k, d in enumerate(rot["distancias"]):
            c = rot["codigo"][k, limites[i]:limites[i + 1]]
            r = tab[(str(w), int(d))]
            obtido = (int((c <= M.AGUA).sum()), int((c == M.AGUA).sum()), int((c == M.TERRA).sum()))
            esperado = (int(r["scored_pixels"]), int(r["water_pixels"]), int(r["land_pixels"]))
            if obtido != esperado:
                raise ValueError(f"{w} at {d} m: scored, water, land {obtido}, the table says {esperado}")
            n += 1
    return n


def quantizar(m):
    """Margin for maps: round(4 m) as int8, clipped to -127..127 (|m| above 31.75 is stored as 31.75)."""
    return np.clip(np.rint(np.asarray(m, np.float32) * ESCALA_Q), -127, 127).astype(np.int8)
