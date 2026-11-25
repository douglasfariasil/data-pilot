import json
import logging
import re
import unicodedata

import pandas as pd
from openai import OpenAI

from data_pilot.config import Settings, settings
from data_pilot.tools import build_tool_registry, execute_tool_call, get_tool_definitions

logger = logging.getLogger(__name__)

MAX_TOOL_CALLS = 3

SYSTEM_PROMPT = (
    "Você é o assistente do Data Pilot, um painel de análise de vendas em português. "
    "Regra absoluta: nunca calcule, estime ou invente números de vendas. "
    "Sempre que precisar de métricas ou dados da base, utilize as ferramentas disponíveis "
    "e explique apenas os valores retornados por elas. "
    "Não invente valores, percentuais, datas ou comparações. "
    "Não apresente resultados que não estejam nos dados retornados pelas ferramentas. "
    "Se uma ferramenta informar erro ou se a base estiver vazia ou sem dados suficientes, "
    "informe isso claramente."
)

DATA_KEYWORDS = {
    "receita",
    "faturamento",
    "venda",
    "vendas",
    "pedido",
    "pedidos",
    "ticket",
    "cliente",
    "clientes",
    "cancelamento",
    "cancelamentos",
    "categoria",
    "categorias",
    "mensal",
    "mes",
    "produto",
    "produtos",
    "valor",
    "quantidade",
    "total",
    "media",
    "taxa",
    "comparar",
    "compare",
    "crescimento",
    "quanto",
    "quantos",
    "desempenho",
    "tendencia",
    "evolucao",
    "ranking",
    "maior",
    "menor",
    "melhor",
    "pior",
    "top",
    "mostrar",
    "mostre",
    "comparacao",
    "queda",
    "aumento",
    "reducao",
}


def requires_data_tool(user_message: str) -> bool:
    """Identifica perguntas que exigem consulta aos dados de vendas."""

    normalized = unicodedata.normalize("NFKD", user_message.casefold())
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))

    words = set(re.findall(r"\w+", normalized))

    return bool(words & DATA_KEYWORDS)


def extract_response_message(response: object):
    """Extrai a mensagem da resposta da API, validando sua estrutura."""

    choices = getattr(response, "choices", None)

    if not choices:
        raise ValueError("A API retornou uma resposta sem opções.")

    message = getattr(choices[0], "message", None)

    if message is None:
        raise ValueError("A API retornou uma opção sem mensagem.")

    return message


class DataPilotAgent:
    """Agente que usa tool calling via OpenRouter mantendo o cálculo em Python."""

    def __init__(self, df: pd.DataFrame, cfg: Settings = settings) -> None:
        self.df = df
        self.cfg = cfg
        self.tools = get_tool_definitions()
        self.registry = build_tool_registry(df)

    def _build_client(self) -> OpenAI:
        if not self.cfg.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY não configurada no arquivo .env.")

        return OpenAI(
            api_key=self.cfg.openrouter_api_key,
            base_url=self.cfg.openrouter_base_url,
            timeout=30.0,
            max_retries=0,
        )

    def ask(self, user_message: str) -> str:
        """Processa perguntas, exigindo ferramentas para consultas quantitativas."""

        if not user_message or not user_message.strip():
            return "Digite uma pergunta sobre as vendas para eu consultar as métricas."

        message = user_message.strip()
        must_use_tools = requires_data_tool(message)

        try:
            client = self._build_client()

            messages: list[dict] = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": message},
            ]

            first_response = client.chat.completions.create(
                model=self.cfg.openrouter_model,
                messages=messages,
                tools=self.tools,
                tool_choice="required" if must_use_tools else "auto",
                temperature=0,
            )

            assistant_msg = extract_response_message(first_response)
            tool_calls = getattr(assistant_msg, "tool_calls", None) or []

            if len(tool_calls) > MAX_TOOL_CALLS:
                logger.warning(
                    "O modelo solicitou %s ferramentas; limite permitido: %s.",
                    len(tool_calls),
                    MAX_TOOL_CALLS,
                )
                return (
                    "A consulta solicitou operações demais de uma só vez. "
                    "Tente fazer uma pergunta mais específica."
                )

            seen_calls = set()

            for tc in tool_calls:
                tool_name = tc.function.name
                arguments = tc.function.arguments or "{}"

                if tool_name not in self.registry:
                    logger.warning(
                        "O modelo solicitou uma ferramenta desconhecida: %s.",
                        tool_name,
                    )
                    return (
                        "Não foi possível realizar a consulta solicitada. "
                        "Tente reformular sua pergunta."
                    )

                try:
                    normalized_arguments = json.dumps(
                        json.loads(arguments),
                        sort_keys=True,
                        ensure_ascii=False,
                    )
                except (json.JSONDecodeError, TypeError):
                    normalized_arguments = arguments

                call_signature = (tool_name, normalized_arguments)

                if call_signature in seen_calls:
                    logger.warning(
                        "O modelo solicitou uma chamada duplicada: %s.",
                        tool_name,
                    )
                    return "A consulta gerou operações repetidas. Tente reformular sua pergunta."

                seen_calls.add(call_signature)

            if not tool_calls:
                if must_use_tools:
                    logger.warning(
                        "O modelo não chamou ferramentas para uma pergunta quantitativa."
                    )
                    return (
                        "Não foi possível consultar os dados necessários para responder. "
                        "Tente novamente."
                    )

                return (
                    assistant_msg.content
                    or "Não consegui identificar uma consulta para realizar agora."
                )

            messages.append(
                {
                    "role": "assistant",
                    "content": assistant_msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ],
                }
            )

            for tc in tool_calls:
                result = execute_tool_call(
                    tool_name=tc.function.name,
                    arguments_json=tc.function.arguments,
                    registry=self.registry,
                )

                if "error" in result:
                    logger.warning(
                        "A ferramenta '%s' não conseguiu concluir a consulta.",
                        tc.function.name,
                    )
                    return (
                        "Não foi possível consultar os dados necessários. "
                        "Tente reformular sua pergunta ou tente novamente."
                    )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )

            second_response = client.chat.completions.create(
                model=self.cfg.openrouter_model,
                messages=messages,
                temperature=0,
            )

            second_assistant_msg = extract_response_message(second_response)

            return (
                second_assistant_msg.content
                or "Os dados foram consultados, mas não foi possível gerar uma explicação."
            )

        except Exception:
            logger.exception("Falha ao processar pergunta no Data Pilot.")
            return (
                "Não foi possível concluir sua consulta no momento. "
                "Verifique a disponibilidade do serviço e tente novamente."
            )
