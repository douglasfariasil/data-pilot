# Data Pilot — Análise Inteligente de Dados com Agentes de IA

Aplicação em Python que combina um painel interativo de vendas (**Streamlit** + **Plotly**) com um assistente conversacional em português conectado ao **OpenRouter** via *tool calling*.

O projeto segue o princípio **"o Python calcula, a IA só explica"**: todas as métricas são processadas de forma determinística com `pandas` sobre pedidos concluídos, enquanto o modelo de linguagem apenas decide qual função chamar e traduz o retorno em uma resposta clara.

---

## Funcionalidades

- **Preparação e Padronização de Dados (`data_prep.py`)**:
  - Leitura da base bruta (`data/raw/sales.csv`), conversão segura de datas, padronização de status, remoção de duplicatas e criação da coluna de receita (`revenue`).
  - Persistência e leitura do dataset tratado (`data/processed/sales_clean.csv`) usando `;` como separador e `,` como separador decimal.
- **Painel de Vendas (`dashboard.py`)**:
  - Indicadores principais filtráveis por categoria: **Receita Total**, **Pedidos Concluídos**, **Ticket Médio**, **Clientes Únicos**, **Taxa de Cancelamento** e **Receita Média por Cliente**.
  - Gráfico interativo de **Receita Mensal** construído com **Plotly**.
- **Conversa com os Dados (`llm_agent.py` + `tools.py`)**:
  - Integração com o OpenRouter (`openrouter/free`) utilizando chamada de ferramentas (*function/tool calling*).
  - Garantia de consistência entre os números exibidos no painel e os valores respondidos no chat.

---

## Melhoria Implementada

Além do escopo base, foi adicionada uma nova métrica de negócio integrada de ponta a ponta (cálculo em `metrics.py`, card no `dashboard.py`, ferramenta registrada em `tools.py` e cobertura de testes automatizados):

- **Receita Média por Cliente (`average_revenue_per_customer` / `get_average_revenue_per_customer`)**: calcula a receita média gerada por cada cliente único considerando apenas pedidos concluídos (com suporte a filtro por categoria).

---

## Ferramentas Disponíveis para o Agente

| Ferramenta | Descrição | Parâmetros |
| :--- | :--- | :--- |
| `get_total_revenue` | Retorna a receita total em reais de pedidos concluídos. | `category` *(opcional)* |
| `get_total_orders` | Retorna a quantidade de pedidos concluídos. | `category` *(opcional)* |
| `get_average_ticket` | Retorna o ticket médio (receita total / pedidos concluídos). | `category` *(opcional)* |
| `get_unique_customers` | Retorna o total de clientes únicos em pedidos concluídos. | `category` *(opcional)* |
| `get_cancellation_rate` | Retorna a taxa de cancelamento (%) sobre o total de pedidos. | `category` *(opcional)* |
| `get_monthly_revenue` | Retorna a receita mensal agrupada (`YYYY-MM`) de pedidos concluídos. | `category` *(opcional)* |
| `get_average_revenue_per_customer` | Retorna a receita média por cliente único em pedidos concluídos. | `category` *(opcional)* |

---

## Estrutura do Projeto

```text
data-pilot/
├── .env.example
├── .gitignore
├── pyproject.toml
├── uv.lock
├── README.md
├── AGENTS.md
├── app.py
├── assets/
│   ├── painel.png
│   └── chat.png
├── data/
│   ├── raw/
│   │   └── sales.csv
│   └── processed/
│       └── sales_clean.csv
├── src/
│   └── data_pilot/
│       ├── __init__.py
│       ├── config.py
│       ├── data_prep.py
│       ├── metrics.py
│       ├── tools.py
│       ├── llm_agent.py
│       └── ui/
│           ├── dashboard.py
│           └── chat_view.py
└── tests/
    ├── test_data_prep.py
    ├── test_metrics.py
    └── test_tools.py

```

## Como Instalar, Configurar e Rodar

1. Pré-requisitos
Python 3.13+

uv instalado

Conta gratuita no OpenRouter para gerar sua chave de API

2. Instalação das dependências

git clone [https://github.com/douglasfariasil/data-pilot.git](https://github.com/douglasfariasil/data-pilot.git)

cd data-pilot

uv sync

3. Configuração das variáveis de ambiente

Copie o arquivo de exemplo e preencha sua chave do OpenRouter

cp .env.example .env

Edite o .env:

OPENROUTER_API_KEY=

OPENROUTER_BASE_URL= [https://openrouter.ai/api/v1](https://openrouter.ai/api/v1)

OPENROUTER_MODEL=openrouter/free


4. Executar a aplicação

uv run streamlit run app.py

5. Executar os testes automatizados

uv run pytest

# Evidências de Funcionamento
Substitua os arquivos abaixo pelos prints reais da sua aplicação em execução antes de finalizar o repositório

## Painel de Vendas

Conversa com os Dados

---

### 2. Checklist Final Antes de Enviar o Link[cite: 1]

- **Base real preenchida**: Assim que colocar os dados reais no `data/raw/sales.csv`, rode o app e confira se os nomes das colunas (`order_id`, `customer_id`, `category`, `status`, `order_date`, `quantity`, `unit_price`) batem com o CSV real — ajustando apenas o mapeamento em `data_prep.py` se necessário.
- **Prints reais em `assets/`**: Tire um print nítido do **Painel** (mostrando os cards e o gráfico Plotly) e um print do **Chat** respondendo uma pergunta (por exemplo, sobre a receita média por cliente ou receita total) e salve como `assets/painel.png` e `assets/chat.png`.
- **Coerência de arquivos citados**: Garanta que todas as pastas e arquivos mencionados no `README.md` existam no repositório e não estejam vazios (por exemplo, se não for versionar `data/processed/sales_clean.csv`, remova-o da árvore do README ou adicione um `.gitkeep`).
- **Segurança do `.env`**: Confirme com `git status` que o `.env` está ignorado e que apenas o `.env.example` será enviado.
- **Números sincronizados**: Confira se o valor respondido no chat bate exatamente com o exibido nos cards do painel.
- **Configuração do repositório no GitHub**:
  - Repositório **público** na sua conta (`douglasfariasil`).
  - Nome legível, em **minúsculas** e **sem acento** (ex.: `data-pilot`).
  - Envie o link da raiz do repositório (ex.: `[https://github.com/douglasfariasil/data-pilot](https://github.com/douglasfariasil/data-pilot)`), e não de um arquivo interno.

Assim que você colocar o `sales.csv` real, se quiser colar aqui as primeiras linhas ou os nomes das colunas dele, ajustamos o `data_prep.py` em segundos!

---
# Arquitetura
```
                  DATA PILOT
                         |
              +----------+----------+
              |                     |
          DASHBOARD              AI AGENT
              |                     |
       Streamlit + Plotly      OpenRouter
              |                     |
              |                Tool Calling
              |                     |
              +----------+----------+
                         |
                  BUSINESS LOGIC
                         |
                  METRICS ENGINE
                         |
                  DATA PIPELINE
                         |
                  CSV / DATABASE

```

## Base de dados recomendada: Olist

Repositório com os arquivos CSV da Olist

- link da  arquivos CSV da Olist: [https://github.com/infobarbosa/dataset-olist](https://github.com/infobarbosa/dataset-olist)

Conjunto de dados públicos de comércio eletrônico brasileiro por Olist

- Link da Dataset oficial da Olist no Kaggle: [https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

A licença é CC BY-NC-SA 4.0: permite uso educacional e de pesquisa não comercial, exige atribuição e determina compartilhamento sob a mesma licença.
