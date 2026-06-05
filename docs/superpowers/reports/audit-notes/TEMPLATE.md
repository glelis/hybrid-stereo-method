# Template de achado de auditoria

Copie o bloco abaixo para registrar cada achado. IDs por estágio: MF-xx (multifocus),
PS-xx (fotométrico), INT-xx (integração), IO-xx (infra/IO), CONV-xx (convenções).
Numere sequencialmente dentro de cada prefixo.

---

## <ID>: <título curto>

- **Localização:** `caminho/arquivo.py:linha`
- **Tipo:** conceitual | implementação
- **Severidade:** crítico | alto | médio | baixo
- **Status:** suspeita | confirmado (ref. ao teste) | refutado (ref. ao teste)

**Descrição:** o que está errado e por quê (1 parágrafo; cite a teoria quando o tipo
for conceitual).

**Evidência:** trecho de código, raciocínio matemático, ou nome do teste sintético.

**Sugestão de correção:** o que mudar (NÃO aplicar).

---

Critérios de severidade (do spec):
- crítico: corrompe o resultado científico
- alto: erro mensurável no resultado
- médio: degrada robustez/precisão em casos comuns
- baixo: caso de borda

Critério de inclusão: só corretude, precisão ou robustez. Estilo/performance ficam fora.
