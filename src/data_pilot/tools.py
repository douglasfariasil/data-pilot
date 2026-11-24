import json
import logging
from collections.abc import Callable
from typing import Any

import pandas as pd

from data_pilot import metrics

logger = logging.getLogger(__name__)


def get_tool_definitions() -> list[dict[str, Any]]:
    """Define os schemas das ferramentas expostas ao modelo via OpenRouter."""
    return [
        {
            "type": "function",
            "function": {
                "name": "get_total_revenue",
                "description": (
                    "Retorna a receita total em reais apenas de pedidos concluídos. "
                    "Use quando o usuário perguntar sobre faturamento ou receita total."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "description": "Categoria opcional para filtrar (ex.: 'eletronicos').",
                        }
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_total_orders",
                "description": (
                    "Retorna a quantidade total de pedidos concluídos. "
                    "Use quando o usuário perguntar quantos pedidos foram concluídos ou vendidos."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "description": "Categoria opcional para filtrar.",
                        }
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_average_ticket",
                "description": (
                    "Retorna o ticket médio ( "
                    "receita total dividida pelo número de pedidos concluídos)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "description": "Categoria opcional para filtrar.",
                        }
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_unique_customers",
                "description": "Retorna a quantidade de clientes únicos em pedidos concluídos.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "description": "Categoria opcional para filtrar.",
                        }
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_average_revenue_per_customer",
                "description": (
                    "Retorna a receita média por cliente único em reais considerando apenas "
                    "pedidos concluídos (receita total dividida pelo número de clientes únicos). "
                    "Use quando o usuário perguntar quanto cada cliente gera em média de receita."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "description": "Categoria opcional para filtrar ( ex.: 'eletronicos').",
                        }
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_cancellation_rate",
                "description": (
                    "Retorna a taxa de cancelamento em percentual em relação ao total de pedidos."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "description": "Categoria opcional para filtrar.",
                        }
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_monthly_revenue",
                "description": (
                    "Retorna a receita mensal agrupada por mês (YYYY-MM) de pedidos concluídos."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "description": "Categoria opcional para filtrar.",
                        }
                    },
                    "required": [],
                },
            },
        },
    ]


def build_tool_registry(df: pd.DataFrame) -> dict[str, Callable[..., Any]]:
    """Mapeia o nome da ferramenta pedida pelo modelo para a função determinística em Python."""
    return {
        "get_total_revenue": lambda category=None: {
            "category": category,
            "total_revenue": round(metrics.total_revenue(df, category=category), 2),
        },
        "get_total_orders": lambda category=None: {
            "category": category,
            "total_orders": metrics.total_orders(df, category=category),
        },
        "get_average_ticket": lambda category=None: {
            "category": category,
            "average_ticket": round(metrics.average_ticket(df, category=category), 2),
        },
        "get_unique_customers": lambda category=None: {
            "category": category,
            "unique_customers": metrics.unique_customers(df, category=category),
        },
        "get_average_revenue_per_customer": lambda category=None: {
            "category": category,
            "average_revenue_per_customer": round(
                metrics.average_revenue_per_customer(df, category=category), 2
            ),
        },
        "get_cancellation_rate": lambda category=None: {
            "category": category,
            "cancellation_rate_pct": round(metrics.cancellation_rate(df, category=category), 2),
        },
        "get_monthly_revenue": lambda category=None: {
            "category": category,
            "monthly_revenue": metrics.monthly_revenue(df, category=category).to_dict(
                orient="records"
            ),
        },
    }


def execute_tool_call(
    tool_name: str,
    arguments_json: str | None,
    registry: dict[str, Callable[..., Any]],
) -> dict[str, Any]:
    """Executa uma ferramenta com validação de argumentos e tratamento seguro de erros."""

    func = registry.get(tool_name)

    if func is None:
        return {"error": "A ferramenta solicitada não está disponível."}

    # Valida e interpreta os argumentos recebidos.
    if arguments_json and arguments_json.strip():
        try:
            parsed = json.loads(arguments_json)
        except json.JSONDecodeError:
            return {"error": "Argumentos inválidos enviados pelo modelo."}

        if not isinstance(parsed, dict):
            return {"error": "Os argumentos devem ser enviados como um objeto JSON."}

        allowed_arguments = {"category"}
        unexpected_arguments = set(parsed) - allowed_arguments

        if unexpected_arguments:
            return {"error": "A ferramenta recebeu argumentos não permitidos."}

        category = parsed.get("category")

        if category is not None and not isinstance(category, str):
            return {"error": "O argumento category deve ser um texto ou null."}

        args = {"category": category.strip() or None} if category is not None else {}
    else:
        args = {}

    try:
        return func(**args)
    except Exception:
        logger.exception("Falha ao executar a ferramenta '%s'.", tool_name)
        return {"error": "Não foi possível executar a ferramenta solicitada."}
