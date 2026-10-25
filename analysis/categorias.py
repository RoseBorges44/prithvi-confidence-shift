"""Decision categories of H1 against H2 and of H3 (prereg/preregistro.md, section 1).

Both functions take the point estimate and the limits of the 90% interval and apply the rules in the order
stated there; the first that applies decides. delta is the margin, 0.02."""
import nucleo as N
from reamostragem import ceil_two_thirds


def category_h1_h2(ponto, lo, hi, delta=N.DELTA):
    """Categories for Delta_shift = AUROC_e(target) - AUROC_e(source test), regime R3."""
    if lo > -delta and hi < delta:
        return "H2, preservation"
    if hi < 0 and ponto <= -delta:
        return "H1, relevant degradation"
    if hi < 0:
        return "Degradation smaller than the margin"
    if lo > 0:
        return "Improvement"
    return "Inconclusive"


def category_h3(ponto, lo, hi, s1_ponto, n_negativos, n_positivos, n_eventos, delta=N.DELTA):
    """Categories for Delta_reg = AUROC_e(R3) - AUROC_e(R1) on the target.

    s1_ponto: point estimate of S1 (Delta_reg on the target minus Delta_reg on the source test).
    n_negativos, n_positivos: number of target events whose per-event Delta_reg is negative, positive.
    An event with an undefined AUROC_e in any run counts in neither number."""
    exigido = ceil_two_thirds(n_eventos)
    if lo > -delta and hi < delta:
        return "Null"
    if hi < 0:
        forte = ponto <= -delta and s1_ponto < 0 and n_negativos >= exigido
        return "Distortion, strong" if forte else "Distortion, weak"
    if lo > 0:
        forte = ponto >= delta and s1_ponto > 0 and n_positivos >= exigido
        return "Rigidity, strong" if forte else "Rigidity, weak"
    return "Inconclusive"
