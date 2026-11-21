# Investigação forense — `00_check_results copy.ipynb`, célula 17 (melhores resultados, 22/04/2025)

**Data da investigação:** 2026-07-12
**Notebook:** `notebooks/hybrid_method/00_check_results copy.ipynb` (não versionado — `notebooks/` está no `.gitignore`)
**Alvo:** imagens da célula 17 (`execution_count=19`), que lê `src/hybrid_method/integrate_recursive/teste_10/teste-0{0..6}-end-Z.fni`

---

## 0. Conclusão em uma frase

O resultado da célula 17 é a **integração recursiva multiescala do gradiente derivado do normal map do photometric stereo, SEM o mapa de hints (`-hints`)** — e a memória sobre o "filtro de frequência" está **confirmada**, mas ele fica a montante: é o **passa-alta gaussiano elíptico no domínio de Fourier** (descritor focal `fourier`, raio 0.1) usado no multifocus stereo que gera os mosaicos all-in-focus dos quais o photometric stereo extrai as normais.

Grau de certeza: a receita foi **provada por fingerprint byte a byte** dos sistemas lineares (`teste-XX-beg-S.txt`, preservados no git) — detalhes na §5.

---

## 1. RECEITA DE REPRODUÇÃO

### Etapa A — Pipeline híbrido (multifocus fourier + photometric stereo)

> Opcional: os artefatos originais desta etapa **ainda existem em disco** com mtime de 22/04/2025 e podem ser usados diretamente (pule para a Etapa B).
> Artefato-chave: `data/results/hybrid_stereo/20250422_1747_2025-03-08-stQ-melon24-amb0.00-glo0.50/photometric_stereo/normal_map_with_residuals.fni`

```bash
# ambiente: conda env hybrid_stereo_method_py310 (Python 3.10.15) — ver §6
cd /home/lelis/Documents/Projetos/hybrid-stereo-method
PYTHONPATH=src python src/hybrid_method/main.py --param_file tests/hb_experiment.yaml
```

com `tests/hb_experiment.yaml` exatamente como está hoje no `main` (estado igual ao do run 17:47, confirmado pelo log do run):

```yaml
data_foldername: '2025-03-08-stQ-melon24-amb0.00-glo0.50'
interpolation_type: 'linear_interpolation'
focal_step: 1
focal_descriptor_paramiters:
  focal_descriptor: 'fourier'     # <-- o "filtro de frequência"
  fourier_radius: 0.1             # <-- NÃO 0.05
  square: False
  smooth: False
  zero_border: False
mask: False
photometric_stereo:
  data_scale: 1.0
  image_type: "png"
  method_name: 'wodham_implementation_argmax'
```

O pipeline: (1) média das imagens sVal por plano focal; (2) multifocus stereo com descritor `fourier` (FFT 2D → máscara passa-alta gaussiana elíptica com raio 0.1, `focus_indicator_fourier.py`) na média e em cada luz `L000..L011`; (3) photometric stereo (`wodham_implementation_argmax` = `estimate_normals_argmax_lstsq_robust` em `wps.py`) sobre os mosaicos `sMos.png` all-in-focus de cada luz, com `images_ps/lights.npy` → `normal_map_with_residuals.fni` (4 canais: X, Y, Z, confiança).

⚠️ **Uma divergência do código atual afeta só quem for usar hints** (não é o caso da célula 17): desde o commit `7299e31` (26/04/2025), `zMos_with_confidence.fni` é salvo **normalizado 0–1**. O run 17:47 usou uma versão pré-commit que salvava em **profundidade absoluta 15–125**. Ver §4.

### Etapa B — Gradiente (células 4, 6, 7, 8 do notebook)

```python
import numpy as np, sys
sys.path.append('src')
from common.io import read_fni_to_image_array, convert_image_array_to_fni

normal_map = read_fni_to_image_array(
    'data/results/hybrid_stereo/20250422_1747_2025-03-08-stQ-melon24-amb0.00-glo0.50/'
    'photometric_stereo/normal_map_with_residuals.fni')          # célula 4

z = normal_map[:, :, 2]                                          # célula 6
z[z == 0] = 1e-10
gradient_map = np.stack((normal_map[:,:,0]/z, normal_map[:,:,1]/z), axis=2)

gradient_map = np.clip(gradient_map,                             # célula 7
                       np.percentile(gradient_map, 1),
                       np.percentile(gradient_map, 99))
gradient_map[:, :, 0] = -gradient_map[:, :, 0]                   # inverte sinal do canal dz/dx

conf = np.expand_dims(normal_map[:, :, 3], axis=-1)              # célula 8
convert_image_array_to_fni(np.concatenate((gradient_map, conf), axis=-1),
                           'gradiente_recursive.fni')
```

### Etapa C — Integração recursiva (o "método híbrido/recursivo")

```bash
cd src/hybrid_method/integrate_recursive
mkdir -p out
./gus_integrate_recursive \
    -initial zero 10 \
    -outPrefix out/teste \
    -slopes /caminho/para/gradiente_recursive.fni
# SEM -hints — este é o ponto decisivo (provado, ver §5)
```

Defaults efetivos do binário: `-maxIter 100000`, `-convTol 5e-7`, 9 níveis (00–08).

### Etapa D — Visualização (célula 17)

```python
for i in range(0, 7):
    image = read_fni_to_image_array(f'out/teste-0{i}-end-Z.fni')
    plt.imshow(image[:, :, 0], cmap='gray'); plt.colorbar(); plt.show()
# sem nenhum clip
```

**Verificação executada nesta investigação:** essa receita reproduz a forma da célula 17 com fidelidade (pirâmide + esfera brilhante + esfera inferior, fundo ~0); range obtido −5.9..71 vs ~−5..90 no notebook (diferença de amplitude discutida na §5.3 — sistema sem hints é singular e a solução depende do ruído inicial aleatório e do critério de parada).

---

## 2. Tabela de parâmetros e origem

| Parâmetro | Valor | Origem / evidência |
|---|---|---|
| Dataset | `2025-03-08-stQ-melon24-amb0.00-glo0.50` | log `hybrid_stereo_20250422_1747.log`; pasta e `.tgz` intactos em `data/raw/hybrid_stereo/` |
| Planos focais `zFoc` | `[15, 25, …, 125]` (12 frames, passo 10) | hardcoded `src/multifocus_stereo/main.py:103`; dirs `organized_by_focus/zfXXX-df020` |
| `focal_descriptor` | `fourier` | log 1747 (linha "Parameter - …focal_descriptor: fourier") |
| `fourier_radius` | `0.1` | log 1747; yaml committado em `fe4d56e` |
| `square` / `smooth` / `zero_border` | `False` / `False` / `False` | log 1747 |
| `interpolation_type` | `linear_interpolation` | log 1747 |
| `focal_step` | 1 | log 1747 |
| Filtro de frequência | passa-alta gaussiano elíptico em FFT 2D, máscara `1 − exp(−(d²/2))` com `radius_y = 0.1·H`, `radius_x = 0.1·W` | `src/multifocus_stereo/focus_indicator_fourier.py` (idêntico desde `fe4d56e`) |
| Método PS | `wodham_implementation_argmax` (`estimate_normals_argmax_lstsq_robust`) | log 1747; `src/photometric_stereo/wps.py` |
| Luzes | `images_ps/lights.npy` | log 1801 (caminho impresso) |
| Normal map usado no gradiente | `20250422_1747/.../normal_map_with_residuals.fni` | célula 4 do notebook (caminho explícito); provado pelo fingerprint §5 |
| Clip do gradiente | percentis 1 e 99 (globais, sobre os 2 canais) | célula 7 |
| Inversão de sinal | apenas canal 0 (dz/dx) após o clip | célula 7 |
| Canal de confiança do slopes | canal 3 do normal map (resíduos do PS) | célula 8 |
| Integrador | `gus_integrate_recursive` (binário committado, ELF x86-64) | `src/hybrid_method/integrate_recursive/` |
| `-initial` | `zero 10` (ruído aleatório ±2115 no chute inicial) | célula 27 e markdown célula 9 (padrão de todos os runs); stderr do binário |
| `-hints` | **AUSENTE** | **provado**: nenhuma equação de hint nos `teste_10/teste-XX-beg-S.txt` (§5) |
| `-maxIter` / `-convTol` | defaults `100000` / `5e-7` | `gus_integrate_recursive.c:87-90` |
| Visualização | `imshow(image[:,:,0], cmap='gray')`, sem clip | célula 17 |

---

## 3. Linha do tempo de 22/04/2025 (horários +0200, como nos nomes das pastas)

| Hora | Evento | Evidência |
|---|---|---|
| 15:17–15:22 | Commits `1e689d7`, `e032663`, `cbfc33d`, `be99155`, `952eea8`, `cd49dbb`, `8d16992` — pipeline híbrido committado; yaml ainda `laplacian`/`0.05` | `git log`; `git show 8d16992:tests/hb_experiment.yaml` |
| 15:43, 15:58, 15:59 | Runs 1543/1558/1559 com **laplacian** 0.05 | logs dos runs |
| ~16h–17h | Edição não committada: descritor → `fourier`, raio → `0.1` | log 1716 já mostra fourier 0.1 |
| 17:16 | Run 1716 (fourier 0.1; ainda **sem** `zMos_with_confidence`) | log 1716; ausência do arquivo no dir |
| 17:27 | Integração manual no dir do 1716 (com hints, peso 1) | mtime; assinatura +1.0 na diagonal do S |
| ~17:30–17:47 | Edição não committada: `zMos_with_confidence.fni` passa a ser salvo — **em profundidade absoluta 15–125 (sem normalize)** | 1747 tem o arquivo com range 15–125 |
| 17:47–17:51 | **Run 1747** — o run "bom": multifocus fourier 0.1 + PS; `normal_map_with_residuals.fni` salvo 17:50:52 | log 1747; mtimes |
| ~17:52–18:05 | Notebook: gradiente das células 4–8; runs manuais no terminal → `teste_9`, **`teste_10`** (célula 17) | exec_counts; janela de mtimes |
| 17:55–17:57 | Integração com hints (1747avg, peso 1) escrita em `1747/integration` — **reproduzida byte a byte nesta investigação** | md5 idênticos aos da reprodução |
| 18:01–18:04 | Run 1801 — código já editado com `normalize(zMos)` → hints 0–1 | log 1801; range do fni |
| 18:06–18:07 | Célula 27 (`exec 37`): integração com hints normalizados → `1801/integration` (resultado pior, célula 28) | mtimes; range −10..2.6 |
| **26/04 19:23–19:25** | Commits `3750369`, `823d5ce`, `7299e31` (**adiciona `normalize()` ao zMos**), `fe4d56e` (yaml → fourier/0.1) — capturam o estado final do working tree, **não** o estado usado às 17:47 | `git show` de cada um |

Depois da época:

| Data | Evento |
|---|---|
| 2026-01-04 | `39b3d2f` adiciona `integrate_recursive/` ao git, incluindo os `teste_*/teste-XX-beg-S.txt` (os `.fni` ficam de fora — `*.fni` no `.gitignore`) |
| 2026-03-11 | Notebook re-executado parcialmente/salvo (mtime; célula 2 importa `src.hybrid_stereo_method`, pacote que só existe desde 2026) |
| 2026-03-25 | `notebooks/hybrid_method/gradiente_recursive.fni` **sobrescrito** por um teste 5×5 (o original de 422×512 foi perdido — mas é regenerável pela Etapa B) |
| ≥2026-03-11 | Os `teste_10/*-end-Z.fni` foram apagados do disco (hoje só restam os `beg-S.txt` versionados) |

Branches: todos os commits da época estão na linhagem do `main` (contido em todas as branches atuais, que são de 2026). **Não há branches abandonadas de abril/2025.**

---

## 4. Divergências código da época × código atual (main @ `39b3d2f`)

Comparação `git diff fe4d56e HEAD` nos módulos usados:

| Arquivo | Estado | Impacto |
|---|---|---|
| `src/common/io.py` | **idêntico** | nenhum |
| `src/multifocus_stereo/main.py`, `mosaic.py`, `focus_indicator_fourier.py`, `argmax_fuzzy.py` | **idênticos** | nenhum |
| `src/photometric_stereo/mainwps.py` | **idêntico** | nenhum |
| `src/photometric_stereo/wps.py` | só whitespace | nenhum |
| `src/hybrid_method/main.py` | cosmético (renomeia `mean_zf*.png` → `average_zf*.png`, docstrings) | nenhum no resultado |
| `src/multifocus_stereo/image_alignment.py` | refatorado | **não é usado** neste pipeline (sem referência em `main.py`) |
| `tests/hb_experiment.yaml` | idêntico a `fe4d56e` (fourier/0.1) | nenhum |

**Única divergência funcional relevante** (para experimentos com `-hints`, não para reproduzir a célula 17): o código atual (e desde `7299e31`) salva `zMos_with_confidence.fni` como `np.stack((normalize(zMos), normalize(wSel)))` → 0–1. A versão em vigor às 17:47 de 22/04 salvava `zMos` **sem normalizar** (15–125). Correção, se precisar: remover o `normalize()` do `zMos` em `src/multifocus_stereo/main.py` (~linha 127) ou usar o arquivo preservado `20250422_1747/.../multifocus_stereo/average/zMos_with_confidence.fni`.

Observação: o notebook cita hoje imports de `src.hybrid_stereo_method.infrastructure.io.image_io` (pacote novo, 2026). Para reproduzir, use `src/common/io.py` (legado, idêntico à época) ou o equivalente do pacote novo.

---

## 5. Evidência: como a receita foi provada

### 5.1 Fingerprint dos sistemas lineares

Os `teste-XX-beg-S.txt` (dump do sistema `pst_imgsys_t` por nível) dependem **somente** do arquivo de slopes e das opções `-hints/-slopes` — não do ruído inicial. Os de `teste_9` e `teste_10` estão preservados no git (commit `39b3d2f`).

- **`1747/integration` reproduzido byte a byte**: gradiente regenerado pelas células 4–8 + `-hints 1747avg…zMos_with_confidence.fni 1` → md5 dos `beg-S.txt` **idênticos** aos de `data/results/.../20250422_1747/integration/` (níveis 00, 02, 05 conferidos). Isso valida ambiente, binário, IO e a construção do gradiente.
- **`teste_10` não tem hints**: no nível 00 (216 999 equações), as equações de `teste_10` são puramente de gradiente (diagonal = soma dos off-diagonais); a reprodução COM hints difere de `teste_10` exatamente pelos termos de hint em ~205 k equações, e os termos de gradiente (off-diagonais) são **idênticos**.
- **`teste_9` = `teste_10` + hints normalizados peso 1**: diagonal de `teste_9` excede a de `teste_10` em exatamente `+1.0` em todas as equações (assinatura de `-hints <zMos 0–1> 1`) — é por isso que a célula 16 (teste_9) sai achatada em 0–1.25 e ruim: no nível fino os pesos de gradiente caem para ~0.03 e o hint (peso 1) domina, forçando o resultado para o zMos normalizado.
- **Por que a célula 28 (tentativa de re-execução com hints) ficou pior que a 17**: usou os hints normalizados 0–1 do run 1801 — mesma patologia do teste_9.

### 5.2 Verificação visual

Rodando a receita (Etapas B–D) hoje: mesmas formas, mesmas posições, mesmo caráter suave da célula 17 em todos os níveis. Nível 01: range −3.1..36 vs ~45 do notebook; nível 00: −5.9..71 vs ~90.

### 5.3 Diferenças residuais (não impedem a reprodução)

1. **±1 ULP no 7º dígito** em ~92% das equações entre o gradiente regenerado e o de `teste_10` (ex.: `-0.065167` vs `-0.065168`) — o slopes de `teste_10` era uma variante ~1e-6 do gerado pelas células 4–8 (possivelmente outra sessão/numpy). Visualmente irrelevante.
2. **Amplitude do pico (71 vs ~90)**: sem hints o sistema é singular (offset livre) e mal condicionado; a solução final depende do ruído inicial **aleatório** (`-initial zero 10` → perturbação ±2115, sem seed exposto) e do critério de parada (`convTol 5e-7` atingido após ~45 k iterações no nível 00). Reprodução bit a bit do `end-Z` é impossível por causa do RNG; a forma é estável.

---

## 6. Versões de software da época

Do `environment.yml` @ `fe4d56e` (conda env `hybrid_stereo_method_py310`) e metadata do notebook (kernel `hybrid_stereo_method_py310`, Python 3.10.15):

- Python 3.10.15
- numpy 2.0.2 (pip) / numpy-base 2.0.1 (conda)
- opencv-python 4.10.0.84
- scipy 1.14.1
- matplotlib 3.9.3
- Binário C `gus_integrate_recursive`: committado no repo (ELF x86-64, com debug info, não stripped); fonte `gus_integrate_recursive.c` (changelog até 2025-02-18; formato `pst_imgsys_t` de 2025-03-04; libs `libpst`/JSLIBS de J. Stolfi). **Use o binário committado** — a reprodução byte-exata do 1747/integration foi feita com ele.

---

## 7. Lacunas e como verificá-las

| # | Lacuna | Hipótese | Como verificar |
|---|---|---|---|
| 1 | Origem da variação ~1e-6 no gradiente de `teste_10` | `teste_10` foi (re)gerado em outra sessão com numpy/dtype ligeiramente diferente (ex.: float64 vs float32 na escrita `%.7e`) | Regenerar o gradiente forçando `float64` e comparar `beg-S.txt`; testar com numpy 2.0.2 no env py310 |
| 2 | Amplitude exata (~90) da célula 17 | Ruído inicial aleatório + parada do solver no sistema singular | Rodar N vezes com `-initial zero 10` e observar a variância do pico; testar `-initial zero 0` e `-convTol 1e-8` |
| 3 | Cronologia exata `teste_9` vs `teste_10` | Numeração não segue a ordem de execução; `teste_9` usou hints normalizados, que só existiram a partir das 18:01 → `teste_9` provavelmente foi rodado DEPOIS de `teste_10` | mtimes originais perdidos (checkout de 2026); irrecuperável, mas irrelevante para a receita |
| 4 | Qual arquivo exato de hints o `teste_9` usou | `1801/multifocus_stereo/average/zMos_with_confidence.fni` (normalizado) | Rodar com esse hints e comparar `beg-S.txt` com `teste_9` (mesmo método da §5) |
| 5 | Estado misto do notebook (exec_counts de sessões diferentes; célula 2 importa pacote de 2026) | Notebook re-aberto/re-executado parcialmente em 2026-03-11 para visualização; os artefatos de 2025 em disco são a fonte de verdade | Conferir mtimes citados na §3 |
| 6 | Comando exato de `teste_10` além dos slopes | `./gus_integrate_recursive -initial zero 10 -outPrefix teste_10/teste -slopes gradiente_recursive.fni` (padrão dos demais runs; `-initial`/`-outPrefix` não afetam o S) | O fingerprint do S prova slopes+ausência de hints; `-initial zero 10` é inferido do padrão (células 9 e 27) |

---

## 8. Arquivos-chave preservados (não mexer)

- `data/raw/hybrid_stereo/2025-03-08-stQ-melon24-amb0.00-glo0.50/` + `.tgz` — dados brutos
- `data/results/hybrid_stereo/20250422_1747_*/photometric_stereo/normal_map_with_residuals.fni` — insumo direto da célula 17
- `data/results/hybrid_stereo/20250422_1747_*/multifocus_stereo/average/zMos_with_confidence.fni` — único zMos em escala absoluta (15–125) sobrevivente
- `data/results/hybrid_stereo/20250422_17*/hybrid_stereo_*.log` — registro completo dos parâmetros
- `src/hybrid_method/integrate_recursive/teste_10/teste-XX-beg-S.txt` (no git) — fingerprint do run bom
