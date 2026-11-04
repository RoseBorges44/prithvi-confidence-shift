# Medidas de discordância no Rio Grande do Sul

Registro da rodada de `notebooks/semana9_rq4_medidas.ipynb` (commit `816289f`), 2026-11-04, 16,4 min, CPU. Entradas: as margens de `analysis/rq4/margens/` (SHA-256 conferidos com o manifesto), o τ de cada execução e os valores do alvo de `resultados.json` e `tabelas/regimes.csv` da análise confirmatória (SHA-256 em `info_medidas_rq4.json`). Os 7 arquivos foram conferidos de novo depois de remontar o Drive; os SHA-256 de `medidas_sha256.csv` batem com os arquivos aqui.

- `medidas_por_regime.csv`: média, desvio e faixa das 5 sementes, por regime, escopo (tile ou 3 tiles), distância, grupo e medida, com o alvo agregado e a faixa dos cinco eventos para todos os pixels.
- `medidas_por_execucao.csv`: as mesmas medidas, por execução.
- `contagens_celulas.csv`: contagens de cada célula (estrato por número de referências) de cada execução.
- `mapas_<tile>.png`: MSP médio, fração sinalizada e fração em discordância, semente 1 de cada regime, blocos de 100 m.

O AUROC_d do histograma ficou a 9,6e-7 do valor exato (limite 1e-4), com os intervalos de 0,005 até 40 (não precisou refinar). O maior |m| foi 37,4.

## Valores principais (média das 5 sementes, 1 km, 3 tiles, todos os pixels; 66.826.366 pixels)

| Medida | R1 | R2 | R3 | Alvo agregado (R1, R2, R3) | Faixa dos cinco eventos |
|---|---|---|---|---|---|
| Taxa de discordância | 0,182 | 0,184 | 0,181 | 0,029, 0,030, 0,030 | 0,009 a 0,069 |
| AUROC_d | 0,727 | 0,742 | 0,771 | 0,935, 0,943, 0,939 | 0,889 a 0,981 |
| Discordância não sinalizada | 0,664 | 0,634 | 0,645 | 0,326, 0,274, 0,290 | 0,119 a 0,343 |
| Conjunto vazio | 0,131 | 0,146 | 0,135 | 0,060, 0,068, 0,065 | 0,016 a 0,162 |

Por tile, a taxa de discordância (R1) é 0,130 (22JDM), 0,253 (22JCN) e 0,172 (22JDN). Pela distância da camada da UFRGS (R3, 3 tiles) ela é 0,232 a 500 m, 0,181 a 1 km e 0,132 a 2 km. Por estrato (R3, 1 km): construído 0,253, outro 0,191 e água permanente 0,022. Pixels com duas referências (1.043.250): 0,008, com 8.727 discordâncias em média.

## O que os mapas mostram

A discordância se concentra em trechos contínuos de vale (Taquari e Jacuí em 22JCN e 22JDN, o delta do Jacuí em 22JDM), com fração perto de 1 dentro do trecho e perto de 0 fora, e é parecida nos três regimes. Nesses trechos a fração sinalizada é baixa. Em 22JCN, na semente 1 do R3, uma janela isolada no canto noroeste tem MSP médio perto de 0,7 e quase todos os pixels sinalizados; não aparece em R1 e R2.

Esta rodada não separa o sentido da discordância (modelo água e referência terra, ou o contrário), porque `contagens_celulas.csv` guarda só as contagens de discordância e de sinalização. Qualquer leitura da causa fica para depois dessa separação e da leitura do texto do plano (seção 4).
