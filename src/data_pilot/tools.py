import json
import logging
import time
from collections.abc import Callable
from typing import Any

import pandas as pd

from data_pilot import metrics
from data_pilot.logging_config import get_request_id

logger = logging.getLogger(__name__)

MAX_TOOL_ARGUMENT_BYTES = 4 * 1024
MAX_CATEGORY_LENGTH = 100


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

    if arguments_json is not None and not isinstance(arguments_json, str):
        return {"error": "Os argumentos da ferramenta possuem um formato inválido."}

    if func is None:
        return {"error": "A ferramenta solicitada não está disponível."}

    # Impede argumentos excessivamente grandes.
    if arguments_json is not None and len(arguments_json.encode("utf-8")) > MAX_TOOL_ARGUMENT_BYTES:
        return {"error": "Os argumentos da ferramenta ultrapassaram o limite permitido."}

    # Valida e interpreta os argumentos recebidos.
    args = {}

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

        if category is not None:
            category = category.strip()

            if len(category) > MAX_CATEGORY_LENGTH:
                return {"error": "O argumento category ultrapassou o limite permitido."}

            args = {"category": category or None}
        else:
            args = {}

    request_id = get_request_id()
    start_time = time.perf_counter()

    logger.info(
        "Execução de ferramenta iniciada | request_id=%s | tool=%s",
        request_id,
        tool_name,
    )

    try:
        result = func(**args)

        duration_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "Execução de ferramenta concluída | request_id=%s | tool=%s | "
            "duration_ms=%.2f",
            request_id,
            tool_name,
            duration_ms,
        )

        return result

    except Exception as exc:
        duration_ms = (time.perf_counter() - start_time) * 1000

        logger.error(
            "Falha na ferramenta | request_id=%s | tool=%s | "
            "duration_ms=%.2f | error_type=%s",
            request_id,
            tool_name,
            duration_ms,
            type(exc).__name__,
        )

        return {"error": "Não foi possível executar a ferramenta solicitada."}
