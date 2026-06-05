# Resultados dos testes sintéticos

Data: 2026-06-04

---

## test_fni_roundtrip (Task 8)

- **Comando:** `pytest tests/test_fni_roundtrip.py -v`
- **Resultado:** `4 passed in 0.36s`

```
tests/test_fni_roundtrip.py::test_roundtrip_2d PASSED                    [ 25%]
tests/test_fni_roundtrip.py::test_roundtrip_3channel PASSED              [ 50%]
tests/test_fni_roundtrip.py::test_roundtrip_negative_and_large_values PASSED [ 75%]
tests/test_fni_roundtrip.py::test_nan_handling_documented PASSED         [100%]
```

- **Interpretação:**
  - Round-trip Python↔Python sem flip/transposição confirmado: o teste 2D com array (5,9) assimétrico e o teste 3-channel com (4,6,3) reconstroem shapes e valores idênticos. Consistente com a entrada "Verificado" de `04-io.md` (indexação `[y,x]` confere em escrita e leitura — agora com evidência executável).
  - NaN sobrevive ao round-trip Python: `test_nan_handling_documented` passa, confirmando que `float("+nan")` no parser Python reconstrói NaN corretamente. Consistente com a nota "Verificado" em `04-io.md` (`float("+nan")` devolve `NaN` em CPython).
  - Precisão efetiva `rtol~6e-8` (max rel diff observado): `read_fni_to_image_array` aloca `float32`, então o round-trip float64→texto `%.7e`→float32 introduz erro de ~6e-8 relativo — dentro de `rtol=1e-6` e consistente com IO-01 (truncamento `%.7e` limita-se ao epsilon de float32 ~1.19e-7). O retorno `float32` do reader não estava explicitamente documentado nos achados; confirma que a fronteira de precisão é no reader, não apenas no writer.
  - Valores negativos, grandes e zero sobrevivem corretamente (incluindo `-0.0` que é armazenado como `+0.0000000e+00` e recuperado como `0.0` — comportamento float normal sem achado).

---

## test_convention_integration (Task 9)

- **Comando:** `pytest tests/test_convention_integration.py -v -s`
- **Build do binário C:** `cd csrc/integrate_recursive && make` **FALHA** (ver `## Bloqueios`), mas o binário `gus_integrate_recursive` **já está versionado e funcional** (executa e responde ao `--help`); `DEFAULT_EXECUTABLE.exists()` é `True`, então o guard `needs_binary` NÃO pulou os testes — eles rodaram contra o binário pré-compilado.
- **Resultado:** `1 failed, 1 passed in 0.26s`
  - `test_constant_slopes_recover_ramp_and_decide_convention` — **FAILED** (crash ao invocar o binário; NÃO foi um assert de convenção)
  - `test_normals_path_recovers_bump_shape` — **PASSED**

### (1) Caminho `-slopes` (rampa): CRASH do binário — a tabela de convenção NÃO chegou a ser impressa

O teste falhou **antes** de calcular/imprimir a tabela de candidatos: o `integrate_slopes_to_height` levantou `RuntimeError` dentro da chamada ao C, antes de retornar `z`. Logo **não há tabela de RMSE por candidato** para registrar — o assert de convenção (`best == "z = +ax*x + ay*y ..."`) nunca foi alcançado.

Saída de erro do binário (verbatim, do `RuntimeError`/stderr capturado):
```
RuntimeError: Integration failed: reading the slope map {G} ...
Reading .../ramp_slopes.fni ...
allocating the height map {Z} ...
zeroing the initial solution ...
writing out the initial guess ...
wrote .../ramp-ini-Z.fni
  writing .../ramp-00-beg-G.fni ...
    writing .../ramp-01-beg-G.fni ...
      writing .../ramp-02-beg-G.fni ...
        writing .../ramp-03-beg-G.fni ...
          writing .../ramp-04-beg-G.fni ...
pst_integrate_iterative.c:47: ** (pst_integrate_iterative) slope map {G} must have 3 channels
```

Causa-raiz (confirmada na fonte C):
- O entry-point de topo aceita 2 **ou** 3 canais: `demand((NC_G == 2) || (NC_G == 3), "gradient map {G} must have 2 or 3 channels")` (`gus_integrate_recursive.c:503`).
- Mas o solver iterativo interno exige **exatamente 3 canais**: `demand(NC_G == 3, "slope map {G} must have 3 channels")` (`lib-src/pst_integrate_iterative.c:47`).
- O `integrate_slopes_to_height` grava um mapa `(H,W,2)` (canais dZ/dX, dZ/dY) — exatamente o que a docstring documenta como aceitável (`integrate.py:261-263`: "shape (H, W, 2) or (H, W, 3)"). O topo aceita; a recursão **estoura** dentro de `pst_integrate_recursive`→`pst_integrate_iterative`, no nível 0 após escrever os `-NN-beg-G.fni`.
- **Conclusão:** o caminho de **slopes com 2 canais** está quebrado por construção (topo promete 2/3 canais, solver exige 3, e o topo não promove 2→3 antes de recursar). O caminho `-normals` não sofre disso (normais entram com 3 canais Nx,Ny,Nz). **Achado novo de implementação** (registrar como INT/CONV; ver abaixo).

Consequência para CONV-1/CONV-2: o teste da rampa **não pôde decidir** o sinal/orientação de dZ/dY **pelo caminho `-slopes`** com o wrapper Python atual — ele crasha antes de integrar. A decisão de sinal continua **pendente de execução** por essa via; um teste futuro precisaria (a) passar um mapa de 3 canais (dZ/dX, dZ/dY, peso) ao `-slopes`, ou (b) decidir o sinal pelo caminho `-normals` com uma rampa assimétrica. O teste, como escrito (2 canais), expõe o bug do wrapper em vez de medir a convenção.

### (2) Caminho `-normals` (bump): PASSED — integração end-to-end funciona

```
bump: affine-fit rmse=0.0732, a=0.9989, b=1.1207, std(gt)=1.0384
```
- `rmse=0.0732 < 0.15·std(gt)=0.1558` ⇒ forma recuperada a menos de afim. `a≈0.999` (ganho ~unitário) e `b≈1.12` (offset, esperado: constante de integração arbitrária). `np.isfinite(est).all()` ok.
- Confirma: o caminho `-normals` integra um bump simétrico e produz `bump-00-end-Z.fni` corretamente (callback `reportHeights` final ⇒ end-Z existe em sucesso). Como o bump é simétrico, este caminho **não** decide sinais de X/Y (conforme docstring do próprio teste).

### Interpretação / o que isto decide

- A convenção real do C **NÃO foi decidida pelo teste da rampa** porque o caminho `-slopes`/2-canais crasha antes de integrar. CONV-1 e CONV-2 permanecem indecidos-por-execução nesta via; a leitura estática (sinais `-nx/nz`, `-ny/nz` corretos internamente; y-up das luzes vs y-down do numpy não reconciliado) segue de pé como **suspeita não refutada**.
- A integração `-normals` é confirmada funcional end-to-end (INT verificações de "end-Z garantido em sucesso").
- O crash confirma empiricamente a análise de **INT-02**: um `demand` de canais falho aborta o binário com retorno != 0, `subprocess.run(check=True)` levanta `CalledProcessError` **antes** do fallback `-ini-Z.fni` — o fallback **não** é alcançado num crash (apesar de o `ramp-ini-Z.fni` ter sido escrito em disco, ele nunca é lido).
- **Achado novo de implementação:** `integrate_slopes_to_height` com 2 canais é inutilizável (topo aceita 2/3, solver iterativo exige 3) — documentação do wrapper (`integrate.py:261-263`) incorreta para o caso de 2 canais.

---

## Bloqueios

### Build do C (`make`) falha — NÃO bloqueante para Task 9 (binário pré-compilado versionado)

`cd csrc/integrate_recursive && make` falha na recompilação do fonte. **Porém o binário `gus_integrate_recursive` está versionado (rastreado pelo git) e é funcional**, então os testes da Task 9 puderam rodar contra ele. Registro o erro de build aqui para visibilidade (NÃO consertado — sem mudanças em fonte de produção):

```
gus_integrate_recursive.c:9:1: error: "/*" within comment [-Werror=comment]
    9 | /*./gus_integrate_recursive -initial zero 0 -outPrefix teste_blabla -normals .../normal_map_with_residuals.fni */
gus_integrate_recursive.c:312:10: fatal error: bool.h: No such file or directory
  312 | #include <bool.h>
      |          ^~~~~~~~
cc1: all warnings being treated as errors
make: *** [Makefile:126: gus_integrate_recursive.o] Error 1
```

- 1º erro: `-Werror=comment` — um `/*` aninhado dentro de um comentário de bloco (`gus_integrate_recursive.c:9`).
- 2º erro: `#include <bool.h>` (`:312`) não encontrado no caminho de include de sistema (o `bool.h` do projeto vive em `include/`, alcançável só via `"bool.h"` ou se o `-Iinclude` cobrir — mas o build para no `-Werror` antes).
- **Impacto:** nenhum para Task 9 (o binário existente roda). Recompilar do zero exigiria editar o fonte C de produção, o que está fora do escopo desta task (sem mudanças de produção).
