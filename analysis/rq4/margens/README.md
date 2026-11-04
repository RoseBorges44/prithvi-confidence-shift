# Margens do modelo no Rio Grande do Sul

Registro da inferência de `notebooks/semana9_rq4_inferencia.ipynb` (commit `50f8f5c`), rodada em 2026-11-04 em GPU Tesla T4, 32 bits.

- `info_inferencia_rq4.json`: módulos e pesos usados, versões, janelas e pixels por tile, conferência de cada execução na Bolívia.
- `manifesto_sha256.csv`: 48 arquivos (3 de rótulos e 45 de margens, 15 execuções por tile), com bytes e SHA-256. Os arquivos ficam no Drive, em `resultados/semana9/rq4_margens`, e não estão neste repositório (5,5 GB).

Conferência na Bolívia: as 15 execuções reproduzem as margens da semana 6 com diferença máxima 0 e nenhum pixel com sinal trocado (tolerância prevista: 0,05).

Janelas: 152 (22JDM), 216 (22JCN) e 207 (22JDN). Nenhuma medida de discordância foi calculada nesta etapa.
