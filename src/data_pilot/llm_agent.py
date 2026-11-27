import json
import logging
import re
import time
import unicodedata
import uuid

import openai
import pandas as pd
from openai import OpenAI

from data_pilot.config import Settings, settings
from data_pilot.logging_config import request_id_context
from data_pilot.tools import (
    MAX_TOOL_ARGUMENT_BYTES,
    build_tool_registry,
    execute_tool_call,
    get_tool_definitions,
)

logger = logging.getLogger(__name__)

MAX_TOOL_CALLS = 3
MAX_MESSAGE_LENGTH = 2000
MAX_TOOL_RESULT_BYTES = 16 * 1024

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


def validate_tool_calls(tool_calls: list) -> str | None:
    """Valida a estrutura das chamadas de ferramentas antes da execução."""

    if len(tool_calls) > MAX_TOOL_CALLS:
        return "A consulta solicitou operações demais de uma só vez."

    seen_calls = set()
    seen_ids = set()

    for tool_call in tool_calls:
        call_id = getattr(tool_call, "id", None)
        function = getattr(tool_call, "function", None)

        if not isinstance(call_id, str) or not call_id.strip():
            return "A chamada de ferramenta possui um identificador inválido."

        if call_id in seen_ids:
            return "A consulta contém identificadores de chamadas repetidos."

        seen_ids.add(call_id)

        if function is None:
            return "A chamada de ferramenta possui uma estrutura inválida."

        tool_name = getattr(function, "name", None)
        arguments = getattr(function, "arguments", None)

        if not isinstance(tool_name, str) or not tool_name.strip():
            return "A chamada de ferramenta possui um nome inválido."

        if not isinstance(arguments, str):
            return "Os argumentos da ferramenta possuem um formato inválido."

        if len(arguments.encode("utf-8")) > MAX_TOOL_ARGUMENT_BYTES:
            return "Os argumentos da ferramenta ultrapassaram o limite permitido."

        try:
            parsed_arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return "Os argumentos da ferramenta não são um JSON válido."

        if not isinstance(parsed_arguments, dict):
            return "Os argumentos da ferramenta devem ser um objeto JSON."

        call_signature = (
            tool_name,
            json.dumps(
                parsed_arguments,
                sort_keys=True,
                ensure_ascii=False,
            ),
        )

        if call_signature in seen_calls:
            return "A consulta gerou operações repetidas."

        seen_calls.add(call_signature)

    return None


def serialize_tool_result(result: dict) -> str:
    """Serializa o resultado e verifica o limite de tamanho permitido."""

    serialized = json.dumps(result, ensure_ascii=False)
    result_size = len(serialized.encode("utf-8"))

    if result_size > MAX_TOOL_RESULT_BYTES:
        raise ValueError("O resultado da ferramenta ultrapassou o limite permitido.")

    return serialized


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
            max_retries=2,
        )

    def ask(self, user_message: str) -> str:
        """Processa uma pergunta com identificação e medição de duração."""

        request_id = uuid.uuid4().hex[:8]
        token = request_id_context.set(request_id)
        start_time = time.perf_counter()

        logger.info("Consulta iniciada | request_id=%s", request_id)

        try:
            answer = self._ask(user_message)

            duration_ms = (time.perf_counter() - start_time) * 1000

            logger.info(
                "Consulta finalizada | request_id=%s | duration_ms=%.2f",
                request_id,
                duration_ms,
            )

            return answer

        except Exception as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000

            logger.error(
                "Falha inesperada | request_id=%s | duration_ms=%.2f | error_type=%s",
                request_id,
                duration_ms,
                type(exc).__name__,
            )
            raise

        finally:
            request_id_context.reset(token)

    def _ask(self, user_message: str) -> str:
        """Processa perguntas, exigindo ferramentas para consultas quantitativas."""

        if not user_message or not user_message.strip():
            return "Digite uma pergunta sobre as vendas para eu consultar as métricas."

        message = user_message.strip()

        if len(message) > MAX_MESSAGE_LENGTH:
            logger.warning(
                "Mensagem rejeitada por exceder o limite de %s caracteres.",
                MAX_MESSAGE_LENGTH,
            )
            return (
                f"Sua pergunta ultrapassou o limite de "
                f"{MAX_MESSAGE_LENGTH} caracteres. "
                "Resuma sua pergunta e tente novamente."
            )

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

            validation_error = validate_tool_calls(tool_calls)

            if validation_error:
                logger.warning("Chamada de ferramenta rejeitada: %s", validation_error)
                return validation_error

            for tc in tool_calls:
                if tc.function.name not in self.registry:
                    logger.warning(
                        "O modelo solicitou uma ferramenta desconhecida: %s.",
                        tc.function.name,
                    )
                    return (
                        "Não foi possível realizar a consulta solicitada. "
                        "Tente reformular sua pergunta."
                    )

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

                try:
                    serialized_result = serialize_tool_result(result)
                except ValueError:
                    logger.warning(
                        "Resultado da ferramenta '%s' excedeu o limite permitido.",
                        tc.function.name,
                    )
                    return (
                        "O resultado da consulta ultrapassou o limite permitido. "
                        "Tente fazer uma consulta mais específica."
                    )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": serialized_result,
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

        except openai.APITimeoutError as exc:
            logger.error(
                "Timeout no serviço de IA | request_id=%s | error_type=%s",
                request_id_context.get(),
                type(exc).__name__,
            )
            return "O serviço de IA demorou para responder. Tente novamente em alguns instantes."

        except openai.APIConnectionError as exc:
            logger.error(
                "Falha de conexão com o serviço de IA | request_id=%s | error_type=%s",
                request_id_context.get(),
                type(exc).__name__,
            )
            return (
                "Não foi possível estabelecer conexão com o serviço de IA. "
                "Verifique sua conexão e tente novamente."
            )

        except openai.RateLimitError as exc:
            logger.error(
                "Limite de solicitações do serviço de IA | request_id=%s | error_type=%s",
                request_id_context.get(),
                type(exc).__name__,
            )
            return (
                "O serviço de IA recebeu muitas solicitações. "
                "Aguarde alguns instantes e tente novamente."
            )

        except openai.AuthenticationError as exc:
            logger.error(
                "Falha de autenticação no serviço de IA | request_id=%s | error_type=%s",
                request_id_context.get(),
                type(exc).__name__,
            )
            return (
                "Não foi possível autenticar o acesso ao serviço de IA. "
                "Verifique a configuração do serviço."
            )

        except openai.PermissionDeniedError as exc:
            logger.error(
                "Acesso negado pelo serviço de IA | request_id=%s | error_type=%s",
                request_id_context.get(),
                type(exc).__name__,
            )
            return (
                "O acesso ao serviço de IA não está autorizado. "
                "Verifique as permissões da configuração."
            )

        except openai.APIError as exc:
            logger.error(
                "Erro na API do serviço de IA | request_id=%s | error_type=%s",
                request_id_context.get(),
                type(exc).__name__,
            )
            return (
                "O serviço de IA apresentou uma falha ao processar a solicitação. "
                "Tente novamente mais tarde."
            )

        except Exception as exc:
            logger.error(
                "Falha inesperada no Data Pilot | request_id=%s | error_type=%s",
                request_id_context.get(),
                type(exc).__name__,
            )
            return (
                "Não foi possível concluir sua consulta no momento. "
                "Verifique a disponibilidade do serviço e tente novamente."
            )
