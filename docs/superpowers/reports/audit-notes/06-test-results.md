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
