# Sweep v2: busca de parâmetros em estágios, multi-dataset

Plano: `docs/superpowers/plans/2026-11-21-sweep-v2.md`. Relatórios por estágio:
`docs/relatorios/sweep_v2/<estágio>.md|csv`. O sweep antigo (`configs/sweep/`) fica
como registro histórico.

## Arquivos

| arquivo | papel |
|---|---|
| `base.yaml` | config congelada de onde todos os runs partem (`report_step: 0`) |
| `datasets.yaml` | `input`, `data_folder`, `pixel_size` calibrado e papel (dev/teste) de cada dataset |
| `stages.py` | desenho de cada estágio (`STAGES`) e materialização dos YAMLs em `generated/` (gitignored) |
| `run.py` | executa um estágio em paralelo; pula o que já tem `metrics.json`; `--prune` libera disco |
| `summarize.py` | tabela, ranking (regra de empate), agregado multi-dataset e `--promote` |
| `centers/<ds>/<estágio>.yaml` | vencedor (e top-N) promovido de cada estágio = centro do seguinte |
| `candidates.yaml` | candidatos da validação V (montado à mão a partir dos centros) |

Saídas: `data/results/sweep_v2/<estágio>/<ds>/<config>/<timestamp>_<data_folder>/`, com
`spec.json` na pasta da config e `manifest.jsonl` na raiz.

## Fluxo de cada estágio

```bash
python configs/sweep_v2/run.py --stage A1 --datasets d1,d2 -j 3 --prune     # roda (retomável)
python configs/sweep_v2/summarize.py --stage A1 --datasets d1,d2            # revisar o .md
python configs/sweep_v2/summarize.py --stage A1 --datasets d1,d2 --promote --top 3
```

`--pick NOME` promove uma config específica em vez da escolhida pela regra. O
`centers/*.yaml` registra `how: manual (--pick)`.

| estágio | centro | modo | o que varia |
|---|---|---|---|
| S0 | base | full | base (= legacy13) e legacy15 |
| S0conv | S0 (`--pick base`) | integration | max_iter × conv_tol |
| A1 | base | full | método/banda/kernel × smooth × median × zero_border (89) |
| A2 | top-3 de A1 | full | r_max, interpolação, square, median_ksize, smooth_sigma (OFAT) |
| B | A2 | photometric | shadow_threshold × outlier_threshold_multiplier |
| C1 | B | integration | hints_weight × initial_method |
| C2 | C1 | integration | pixel_size ×{0,8; 0,9; 1,1; 1,25} (sensibilidade; não promover) |
| C3 | top-3 de A1 + B + C1 | photometric | interação multifocus × hints |
| V | `candidates.yaml` | full | candidatos nos 4 datasets (`--aggregate`) |
| Vhints | V | integration | `hints_weight` 0–5 (estendido; rodado em d1–d4) |
| Vfinal | V | integration | candidatos V sem B com `hints_weight` 1,0 (reconfirmação) |

Resultado e recomendação: `docs/relatorios/2026-11-22-sweep-v2.md`.

Métrica primária: `integration_height_object.pearson_r`. Empate: |Δr| < 0,005 e
Δrmse < 2 %; no empate vence quem tem menos overrides. Ao comparar solvers fotométricos, conferir
`photometric_normals.valid_fraction` (≥ 0,99): om < 2 gera normais NaN que inflam o ranking.
