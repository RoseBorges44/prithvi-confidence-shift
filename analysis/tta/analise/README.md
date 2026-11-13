# Análise do TTA

Registro de `notebooks/semana10_tta_analise.ipynb` (commit `b96b8de`), rodado em CPU padrão (13,6 GB de RAM, pico de 10,9 GB) em 2026-11-12. Análise secundária da seção 9 de `prereg/preregistro.md`, rodada depois da abertura da chave (`prereg/desvios.md`, entradas de 2026-11-05, 2026-11-07 e 2026-11-12). Descritiva, fora da correção de Holm.

Roda `analysis/analise_congelada.py` sem `--sem-tta`, sobre as margens da semana 6 e as margens do TTA (`analysis/tta/margens/`). Os scripts foram conferidos por SHA-256 contra `analysis/README.md` antes da rodada.

- `rodada.json`: scripts e versões, comparação com a primeira rodada, conclusão.
- `tta_resumo.csv`: AUROC_e, AUGRC e e-AURC por regime, alvo único contra TTA, e a diferença com IC90 e p.
- `resultados.json` e `tabelas/`: saída completa da rodada.
- `manifesto_entrada_sha256.csv`: SHA-256 de cada arquivo de entrada.
- `analise.log`: registro da execução.

Resultado: as sete tabelas ficaram idênticas, byte a byte, às da primeira rodada. No `resultados.json`, diferem apenas `meta.sem_tta`, `meta.versoes.python` (3.13.18 contra 3.13.16) e `variantes.<regime>.tta`, que só existe nesta rodada. numpy 2.1.3 e scipy 1.16.3 são os mesmos da primeira.

Diferença de AUROC_e no alvo, TTA menos alvo único (média das 5 sementes, IC90): R1 +0,0009 (−0,0002 a +0,0019), R2 +0,0014 (+0,0006 a +0,0023), R3 +0,0032 (+0,0021 a +0,0046). Valores p nominais.
