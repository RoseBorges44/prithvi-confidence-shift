#!/usr/bin/env python3
"""Reference masks of RQ4 (prereg/plano_rq4.md, sections 1 to 3 and the dated changes at its end).

For every pixel of the 10 m grid of a tile, a code says whether the pixel is scored (water or land in the consensus
of the references) or why it is not; a second layer gives the number of references covering the pixel and a third
the stratum of section 3. The tile is then cut into the 512 x 512 windows of prereg/t2_janelas.md. No model output
is read here. The notebook notebooks/semana9_rq4_mascaras.ipynb reads the files and calls these functions; the tests
(teste_mascaras.py) run them on small synthetic grids.
"""
import hashlib
import re

import numpy as np

TAM = 512
MAX_SEM_DADO = 0.01
MIN_PONTUADOS = 26215                    # 10% of 512 x 512, rounded up (prereg/t2_janelas.md)
RAIO_BORDA = 2                           # the `borda` rule of notebooks/semana6_inferencia_alvos.ipynb
GSW_PERMANENTE = 90                      # JRC GSW 1.4 occurrence, percent
NUVEM_PROB = 40                          # s2cloudless probability, percent; 255 is no data
SCL_EXCLUIDAS = (3, 8, 9, 10)            # cloud shadow, cloud medium and high probability, thin cirrus
WC_CONSTRUIDO, WC_AGUA = 50, 80          # ESA WorldCover 2021: built-up, permanent water bodies
WC_CLASSES = (10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100)
PIXEL = 10.0
DISTANCIAS_M = (1000, 500, 2000)         # mapped area of the UFRGS layer: primary, then the two sensitivities
MARGEM_PX = 200                          # the grid is padded by the largest distance before measuring it

TERRA, AGUA = 0, 1
SEM_DADO, FORA, CLASSE_NAO_PONTUADA, DISCORDAM, BORDA, NUVEM, SCL, SEM_MASCARA = 2, 3, 4, 5, 6, 7, 8, 9
CODIGOS = {
    TERRA: "terra (pontuado)",
    AGUA: "água (pontuado)",
    SEM_DADO: "sem dado de entrada (seis bandas iguais a 0)",
    FORA: "fora de toda referência",
    CLASSE_NAO_PONTUADA: "classe não pontuada (rastro de inundação ou deslizamento)",
    DISCORDAM: "referências discordam",
    BORDA: "a até 2 pixels de uma fronteira água-terra do consenso",
    NUVEM: "nuvem (s2cloudless >= 40%)",
    SCL: "SCL 3, 8, 9 ou 10 (sombra, nuvem, cirro)",
    SEM_MASCARA: "sem probabilidade de nuvem ou sem SCL",
}
# When several reasons hold, the code is the first one in this order; the counts of each condition, overlapping,
# are reported apart (contagens).
ORDEM = (SEM_DADO, FORA, CLASSE_NAO_PONTUADA, DISCORDAM, BORDA, NUVEM, SCL, SEM_MASCARA)
ESTRATO_CONSTRUIDO, ESTRATO_AGUA, ESTRATO_OUTRO = 1, 2, 3
ESTRATOS = {ESTRATO_CONSTRUIDO: "construído", ESTRATO_AGUA: "água permanente", ESTRATO_OUTRO: "outro"}


def borda_de(y):
    """True at water or land pixels with a pixel of the other class within 2 pixels (5 x 5 window).
    Pixels labelled -1 do not form a boundary. Same rule as notebooks/semana6_inferencia_alvos.ipynb."""
    from scipy.ndimage import maximum_filter

    agua = (y == 1).astype(np.uint8)
    terra = (y == 0).astype(np.uint8)
    lado = 2 * RAIO_BORDA + 1
    perto_agua = maximum_filter(agua, size=lado, mode="constant", cval=0).astype(bool)
    perto_terra = maximum_filter(terra, size=lado, mode="constant", cval=0).astype(bool)
    return ((agua == 1) & perto_terra) | ((terra == 1) & perto_agua)


def transform_com_margem(transform, margem=MARGEM_PX):
    """(a, b, c, d, e, f) of the grid padded by `margem` pixels on every side."""
    a, b, c, d, e, f = transform
    return (a, b, c - margem * a, d, e, f - margem * e)


def rasterizar(geometrias, transform, forma):
    """Boolean mask of the pixels whose centre falls inside any of the geometries (all_touched=False)."""
    from affine import Affine
    from rasterio.features import rasterize

    geometrias = [g for g in geometrias if g is not None and not g.is_empty]
    if not geometrias:
        return np.zeros(forma, bool)
    m = rasterize(((g, 1) for g in geometrias), out_shape=forma, transform=Affine(*transform[:6]), fill=0,
                  all_touched=False, dtype="uint8")
    return m.astype(bool)


def distancia_px(inundacao):
    """Distance, in pixels between centres, from each pixel to the nearest pixel of `inundacao` (float32).
    Infinite everywhere if `inundacao` is empty."""
    from scipy.ndimage import distance_transform_edt

    if not inundacao.any():
        return np.full(inundacao.shape, np.inf, np.float32)
    return distance_transform_edt(~inundacao).astype(np.float32)


def area_mapeada(dist, distancia_m, margem=MARGEM_PX):
    """Pixels of the tile (the padded grid without its margin) whose centre is at most `distancia_m` from the
    flood extent."""
    raio = distancia_m / PIXEL
    if margem > 0:
        dist = dist[margem:-margem, margem:-margem]
    return dist <= raio


def camada_ufrgs(inundacao, mapeada, permanente):
    """The UFRGS Planet/SkySat layer as a reference: it covers its mapped area, says water on the flood extent and
    on permanent water inside it, land on the rest of it, and has no unscored class."""
    return {"fonte": "ufrgs", "coberto": mapeada, "agua": mapeada & (inundacao | permanente),
            "nao_pontuado": np.zeros_like(mapeada)}


def camada_emsr720(aoi, pegada, nao_analisada, agua_obs, nao_pontuada_obs, permanente, nome="emsr720"):
    """One EMSR720 product as a reference: it covers its area of interest inside the footprint of the image the
    damage was mapped on, minus the parts marked not analysed. Flooded area and permanent water are water,
    flood trace and landslide are unscored, the rest of the covered area is land."""
    coberto = aoi & pegada & ~nao_analisada
    nao_pontuado = coberto & nao_pontuada_obs
    return {"fonte": nome, "coberto": coberto, "agua": coberto & (agua_obs | permanente) & ~nao_pontuado,
            "nao_pontuado": nao_pontuado}


def consenso(camadas, forma):
    """Consensus of the references (section 2, "Consensus", and the change of 2026-11-02 on unscored classes).

    `camadas`: dicts with fonte, coberto, agua, nao_pontuado (boolean grids). Several layers may share a `fonte`
    (the five EMSR720 products are one reference). Returns (rotulo, motivo, nrefs): rotulo 1 water, 0 land, -1 none;
    motivo 0 where the covering references agree, otherwise FORA, CLASSE_NAO_PONTUADA or DISCORDAM; nrefs the number
    of distinct references covering the pixel."""
    vazio = np.zeros(forma, bool)
    diz_agua, diz_terra, nao_pont = vazio.copy(), vazio.copy(), vazio.copy()
    por_fonte = {}
    for c in camadas:
        cob, np_c = c["coberto"], c["nao_pontuado"]
        assert cob.shape == forma and not (np_c & ~cob).any() and not (c["agua"] & ~cob).any(), c["fonte"]
        nao_pont |= cob & np_c
        diz_agua |= cob & c["agua"] & ~np_c
        diz_terra |= cob & ~c["agua"] & ~np_c
        por_fonte[c["fonte"]] = por_fonte.get(c["fonte"], vazio) | cob
    nrefs = np.zeros(forma, np.uint8)
    for cob in por_fonte.values():
        nrefs += cob.astype(np.uint8)
    rotulo = np.full(forma, -1, np.int8)
    rotulo[diz_agua & ~diz_terra & ~nao_pont] = AGUA
    rotulo[diz_terra & ~diz_agua & ~nao_pont] = TERRA
    motivo = np.zeros(forma, np.uint8)
    motivo[nrefs == 0] = FORA
    motivo[nao_pont] = CLASSE_NAO_PONTUADA
    motivo[diz_agua & diz_terra & ~nao_pont] = DISCORDAM
    assert ((rotulo >= 0) == (motivo == 0)).all()
    return rotulo, motivo, nrefs


def conferir_nuvem(prob, scl):
    """Stops on values that the cloud probability (0 to 100, 255 no data) or the SCL (0 to 11) cannot have."""
    ruins = np.unique(prob[(prob > 100) & (prob != 255)])
    if ruins.size:
        raise ValueError(f"cloud probability with values {ruins[:10].tolist()}")
    ruins = np.unique(scl[scl > 11])
    if ruins.size:
        raise ValueError(f"SCL with values {ruins[:10].tolist()}")


def condicoes(sem_dado, motivo, borda, prob, scl):
    """Boolean grid of each reason a pixel is not scored, before the order of ORDEM is applied."""
    return {
        SEM_DADO: sem_dado,
        FORA: motivo == FORA,
        CLASSE_NAO_PONTUADA: motivo == CLASSE_NAO_PONTUADA,
        DISCORDAM: motivo == DISCORDAM,
        BORDA: borda,
        NUVEM: (prob >= NUVEM_PROB) & (prob != 255),
        SCL: np.isin(scl, SCL_EXCLUIDAS),
        SEM_MASCARA: (prob == 255) | (scl == 0),
    }


def codigos(sem_dado, rotulo, motivo, borda, prob, scl):
    """Code of every pixel (CODIGOS): the first reason in ORDEM that holds, otherwise the consensus label."""
    conferir_nuvem(prob, scl)
    cond = condicoes(sem_dado, motivo, borda, prob, scl)
    cod = np.full(rotulo.shape, 255, np.uint8)
    for k in ORDEM:
        cod[(cod == 255) & cond[k]] = k
    livre = cod == 255
    assert (rotulo[livre] >= 0).all()
    cod[livre] = rotulo[livre].astype(np.uint8)
    return cod, cond


def estrato(worldcover):
    """Stratum of section 3 from ESA WorldCover 2021: built-up (50), permanent water bodies (80), other."""
    ruins = np.setdiff1d(np.unique(worldcover), WC_CLASSES)
    if ruins.size:
        raise ValueError(f"WorldCover with values {ruins.tolist()}")
    e = np.full(worldcover.shape, ESTRATO_OUTRO, np.uint8)
    e[worldcover == WC_CONSTRUIDO] = ESTRATO_CONSTRUIDO
    e[worldcover == WC_AGUA] = ESTRATO_AGUA
    return e


def permanente(gsw):
    """Permanent water: JRC GSW 1.4 occurrence of at least 90%."""
    ruins = np.unique(gsw[gsw > 100])
    if ruins.size:
        raise ValueError(f"GSW occurrence with values {ruins.tolist()}")
    return gsw >= GSW_PERMANENTE


def inicios(n, tam=TAM):
    """Window starts along a side of n pixels: steps of `tam` from 0, the last one against the edge."""
    s = list(range(0, n - tam + 1, tam))
    if s[-1] + tam < n:
        s.append(n - tam)
    return s


def percorrer_janelas(altura, largura, tam=TAM):
    """(row, column, novo) of every window in row-major order; `novo` marks the pixels of the window not covered by
    an earlier one, so that each pixel belongs to exactly one window."""
    ja = np.zeros((altura, largura), bool)
    for lin in inicios(altura, tam):
        for col in inicios(largura, tam):
            novo = ~ja[lin:lin + tam, col:col + tam]
            ja[lin:lin + tam, col:col + tam] = True
            yield lin, col, novo
    assert ja.all()


def janelas(cod, tile, tam=TAM, max_sem_dado=MAX_SEM_DADO, min_pontuados=MIN_PONTUADOS):
    """One row per window (prereg/t2_janelas.md applied to the tile): offsets, fraction of the window without input
    data, scored, water and land pixels assigned to it, and whether it enters."""
    linhas = []
    for lin, col, novo in percorrer_janelas(*cod.shape, tam=tam):
        c = cod[lin:lin + tam, col:col + tam]
        frac = float((c == SEM_DADO).mean())
        agua = int(((c == AGUA) & novo).sum())
        terra = int(((c == TERRA) & novo).sum())
        n = agua + terra
        linhas.append({"tile": tile, "window_id": f"{tile}_r{lin:05d}_c{col:05d}", "row_off": lin, "col_off": col,
                       "frac_missing_input": frac, "scored_pixels": n, "water_pixels": agua, "land_pixels": terra,
                       "kept": frac <= max_sem_dado and n >= min_pontuados})
    return linhas


def contagens(cod, cond, nrefs, estr):
    """Pixel counts of a tile: by code (exclusive), by condition (overlapping), scored pixels by number of references
    and by stratum."""
    pont = cod <= AGUA
    out = {"por_codigo": {str(k): int((cod == k).sum()) for k in CODIGOS},
           "por_condicao": {str(k): int(v.sum()) for k, v in cond.items()},
           "pontuados_por_nrefs": {}, "pontuados_por_estrato": {}}
    for n in np.unique(nrefs[pont]).tolist():
        sel = pont & (nrefs == n)
        out["pontuados_por_nrefs"][str(n)] = {"agua": int((sel & (cod == AGUA)).sum()),
                                              "terra": int((sel & (cod == TERRA)).sum())}
    for k, nome in ESTRATOS.items():
        sel = pont & (estr == k)
        out["pontuados_por_estrato"][nome] = {"agua": int((sel & (cod == AGUA)).sum()),
                                              "terra": int((sel & (cod == TERRA)).sum())}
    return out


def resumo_camada(c):
    """Pixels covered by a reference layer, and its water, land and unscored pixels."""
    cob = c["coberto"]
    return {"coberto": int(cob.sum()), "agua": int(c["agua"].sum()), "nao_pontuado": int(c["nao_pontuado"].sum()),
            "terra": int((cob & ~c["agua"] & ~c["nao_pontuado"]).sum())}


def sha_conteudo(a):
    """SHA-256 of the values of an array (type, shape and bytes), independent of the file format that holds it."""
    a = np.ascontiguousarray(a)
    return hashlib.sha256(f"{a.dtype.str}|{a.shape}|".encode() + a.tobytes()).hexdigest()


# EMSR720 products -------------------------------------------------------------------------------------------------

def _id(x):
    s = str(x).strip()
    return str(int(float(s))) if re.fullmatch(r"-?\d+(\.0+)?", s) else s


def classe_emsr720(notation, event_type):
    """Water or unscored for a feature of observedEventA (prereg/plano_rq4.md, change on the EMSR720 classes)."""
    n, e = str(notation).strip(), str(event_type).strip()
    if n == "Flooded area" and e.startswith("5"):
        return "agua"
    if n == "Flood trace" and e.startswith("5"):
        return "nao_pontuada"
    if n == "Not Applicable" and e.startswith("6"):
        return "nao_pontuada"
    raise ValueError(f"observedEventA: class not foreseen in the plan (notation {n!r}, event_type {e!r})")


def separar_emsr720(eventos, pegadas):
    """Splits one product. `eventos`: (properties, geometry) of observedEventA; `pegadas`: (properties, geometry) of
    imageFootprintA. Returns the geometries of water, of unscored classes, of the footprint of the image the damage
    was mapped on (`or_src_id` equal to the single `dmg_src_id` of the events) and of the parts not analysed, with
    that source id."""
    fontes = {_id(p["dmg_src_id"]) for p, _ in eventos}
    if len(fontes) != 1:
        raise ValueError(f"observedEventA with {len(fontes)} damage sources: {sorted(fontes)}")
    fonte = fontes.pop()
    agua, nao_pont = [], []
    for p, g in eventos:
        (agua if classe_emsr720(p["notation"], p["event_type"]) == "agua" else nao_pont).append(g)
    pegada, nao_analisada = [], []
    for p, g in pegadas:
        t = str(p["obj_type"]).strip()
        if t.startswith("Not Analysed"):
            nao_analisada.append(g)
        elif t == "Image Footprint":
            if _id(p["or_src_id"]) == fonte:
                pegada.append(g)
        else:
            raise ValueError(f"imageFootprintA: obj_type not foreseen ({t!r})")
    if not pegada:
        raise ValueError(f"no image footprint with or_src_id {fonte}, the damage source of observedEventA")
    return {"fonte_dano": fonte, "agua": agua, "nao_pontuada": nao_pont, "pegada": pegada,
            "nao_analisada": nao_analisada}


DATAS_DIA = ("2024-05-06", "20240506", "2024/05/06", "06/05/2024", "06-05-2024", "06.05.2024", "05/06/2024")


def fonte_do_dia(linhas, fonte):
    """True if a row of the source table of the product has `fonte` as an id and the date 2024-05-06 in any of its
    fields. `linhas`: list of dicts (all values as text)."""
    for r in linhas:
        ids = [_id(v) for k, v in r.items() if "id" in str(k).lower()]
        if fonte in ids:
            texto = " ".join(str(v) for v in r.values())
            return any(d in texto for d in DATAS_DIA)
    return False


# Check of the alignment of a layer with the image ----------------------------------------------------------------

def alinhamento(img_agua, camada, valido, desloc=3, raio=5, max_pixels=5_000_000):
    """Agreement between a water mask read from the image (not the model) and a reference layer shifted by (dy, dx)
    pixels, on the valid pixels within `raio` of the layer's boundary. A layer on the right grid has its best
    agreement at or next to (0, 0). Returns {(dy, dx): agreement} and the number of pixels used."""
    from scipy.ndimage import maximum_filter

    lado = 2 * raio + 1
    perto_dentro = maximum_filter(camada.astype(np.uint8), size=lado, mode="constant", cval=0).astype(bool)
    perto_fora = maximum_filter((~camada).astype(np.uint8), size=lado, mode="constant", cval=0).astype(bool)
    zona = perto_dentro & perto_fora & valido
    h, w = camada.shape
    zona[:desloc], zona[h - desloc:], zona[:, :desloc], zona[:, w - desloc:] = False, False, False, False
    r, c = np.nonzero(zona)
    if r.size > max_pixels:
        passo = int(np.ceil(r.size / max_pixels))
        r, c = r[::passo], c[::passo]
    if r.size == 0:
        return {}, 0
    img = img_agua[r, c]
    out = {}
    for dy in range(-desloc, desloc + 1):
        for dx in range(-desloc, desloc + 1):
            out[(dy, dx)] = float((img == camada[r + dy, c + dx]).mean())
    return out, int(r.size)
