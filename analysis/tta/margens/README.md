# Margens do TTA

Registro de `notebooks/semana10_tta_margens.ipynb` (commit `8f71a1f`), rodado em GPU Tesla T4, 32 bits, terminado em 2026-11-07 (cerca de 7,3 horas). Análise secundária da seção 9 de `prereg/preregistro.md`, rodada depois da abertura da chave (`prereg/desvios.md`, entradas de 2026-10-25 e 2026-11-05).

- `info_tta.json`: definição do TTA, pesos, versões, unidades e pixels por conjunto, conferência de cada execução e conjunto.
- `manifesto_sha256.csv`: 45 arquivos (15 execuções, conjuntos teste, bolivia e t2), com bytes e SHA-256. Os arquivos ficam no Drive, em `resultados/semana10/tta_margens`, e não estão neste repositório (7,0 GB). Os arquivos da semana 6 não foram alterados.

Conferência da margem sem transformação contra a gravada na semana 6 (limite 1e-3): diferença máxima 2,3e-5. Diferença zero nas execuções do R1 e em quatro do R2. Diferenças entre 6,7e-6 e 2,3e-5 no R2 semente 5 (só no T2) e nas cinco do R3. Três pixels com sinal trocado em 137.160.388 por execução: um no T2 em R2 s5, R3 s1 e R3 s2. A versão do torch no Colab passou de 2.11.0 (inferência do RQ4) para 2.14.1, o que é uma explicação provável para as diferenças na quinta casa decimal; isso não foi testado. A análise usa a margem gravada na semana 6, não a recalculada.

Unidades e pixels: teste 90 chips (20.517.367 pixels), Bolívia 15 (2.867.815), T2 561 janelas (113.775.206). Nenhuma métrica foi calculada nesta etapa.
