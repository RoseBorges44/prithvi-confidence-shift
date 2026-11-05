# Medidas de discordância no Rio Grande do Sul, segunda rodada (sentido da discordância)

Registro da segunda rodada de `notebooks/semana9_rq4_medidas.ipynb` (commit `c0273b3`), 2026-11-05, 24,2 min, CPU. Ela repete a primeira (`analysis/rq4/medidas/`) e acrescenta as três medidas do sentido da discordância, datadas no plano em 2026-11-04. As 1296 médias das seis medidas originais saíram iguais às da primeira rodada (maior diferença 5e-9, só arredondamento do CSV). Os SHA-256 de `medidas_sha256.csv` batem com os arquivos aqui, e o notebook os conferiu depois de remontar o Drive.

Arquivos: os mesmos da primeira rodada. `medidas_por_regime.csv` e `medidas_por_execucao.csv` ganham `discordancia_modelo_agua`, `discordancia_modelo_terra` e `fracao_agua_referencia`, com as contagens; `contagens_celulas.csv` ganha as contagens de cada sentido e de pixels de água da referência. Os mapas são os mesmos.

## Sentido (média das 5 sementes, 1 km, 3 tiles, todos os pixels)

| Medida | R1 | R2 | R3 |
|---|---|---|---|
| Taxa de discordância | 0,182 | 0,184 | 0,181 |
| Modelo água, referência terra | 0,006 | 0,006 | 0,005 |
| Modelo terra, referência água | 0,176 | 0,179 | 0,176 |
| Água na referência | 0,487 | 0,487 | 0,487 |

Quase toda a discordância (97%) é o modelo dizendo terra onde a referência diz água. Em relação à água da referência, isso é cerca de 36% dos pixels de água dela. Por tile, o modelo terra e referência água é 0,247 (22JCN), 0,122 (22JDM) e 0,167 (22JDN), para frações de água da referência de 0,413, 0,640 e 0,409.

Por grupo (R3, 1 km): construído 0,252 (modelo terra) contra 0,001 (modelo água); outro 0,187 contra 0,005; água permanente 0,005 contra 0,017, a única em que o sentido inverso pesa mais. Pela distância da UFRGS, o modelo terra e referência água cai de 0,229 (500 m) para 0,176 (1 km) e 0,125 (2 km), enquanto a fração de água da referência cai de 0,635 para 0,487 e 0,346.

## O que isto não diz

O sentido mostra o que a discordância é, não por quê. Água da referência que o modelo chama de terra é compatível com a água ter baixado entre o pico e a imagem (ressalva do plano, seção 2, item 4), com a camada da UFRGS ser larga, e com omissão real do modelo (água turva, vegetação alagada, ruas). Esta rodada não separa essas explicações.
