# AGENTS.md — Regras do Projeto Data Pilot

## Regra Central
- **O Python calcula, a IA só explica.** Nunca delegue contas, somas, médias ou taxas ao modelo de linguagem.
- Toda métrica exibida no painel ou respondida no chat deve vir de funções determinísticas em `src/data_pilot/metrics.py`.

## Dados e Formatação
- Use `pandas` para limpeza de dados: conversão de datas, padronização de status, remoção de duplicados e cálculo da coluna de receita.
- O arquivo limpo deve ser salvo e lido usando `;` como separador (`sep=";"`) e `,` como decimal (`decimal=","`).
- Cálculos de métricas de vendas devem respeitar o recorte de pedidos concluídos quando aplicável.

## Arquitetura e Limites de Camada
- `data_prep.py`: apenas preparação, limpeza e leitura/escrita do CSV tratado.
- `metrics.py`: apenas funções puras de cálculo de métricas. Não importe `streamlit` nem cliente de LLM aqui.
- `tools.py`: define os schemas (`name`, `description`, `parameters`) e o mapeamento de execução das ferramentas.
- `llm_agent.py`: faz a comunicação com o OpenRouter (`openrouter/free`), trata `tool_calls` e lida com falhas/limites do modelo gratuito de forma resiliente.
- `ui/` e `app.py`: camada de apresentação (Streamlit + Plotly). Evite regras de negócio complexas diretamente na UI.

## Segurança e Repositório
- Nunca escreva segredos ou chaves de API no código; use `.env` (ignorado no `.gitignore`) e mantenha `.env.example`.
- Mantenha mudanças focadas, testáveis e compatíveis com `uv run pytest` e `uv run ruff check .`.
