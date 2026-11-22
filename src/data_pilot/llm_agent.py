import json
import pandas as pd
from openai import OpenAI

from data_pilot.config import Settings, settings
from data_pilot.tools import build_tool_registry, execute_tool_call, get_tool_definitions

SYSTEM_PROMPT = (
    "Você é o assistente do Data Pilot, um painel de análise de vendas em português. "
    "Regra absoluta: nunca calcule, estime ou invente números de vendas. "
    "Sempre que precisar de métricas ou dados da base, chame as ferramentas disponíveis "
    "e explique apenas os valores retornados por elas. Se a base estiver vazia ou sem dados "
    "suficientes, informe isso claramente."
)


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
        )

    def ask(self, user_message: str) -> str:
        """Processa uma pergunta do usuário com suporte a tool calling e fallback resiliente."""
        if not user_message or not user_message.strip():
            return "Digite uma pergunta sobre as vendas para eu consultar as métricas."

        try:
            client = self._build_client()
            messages: list[dict] = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message.strip()},
            ]

            first_response = client.chat.completions.create(
                model=self.cfg.openrouter_model,
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
                temperature=0,
            )

            assistant_msg = first_response.choices[0].message
            tool_calls = getattr(assistant_msg, "tool_calls", None) or []

            if not tool_calls:
                return (
                    assistant_msg.content
                    or "Não consegui identificar uma métrica específica para consultar agora."
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

            return (
                second_response.choices[0].message.content
                or "A métrica foi calculada, mas o modelo não retornou texto."
            )

        except Exception as exc:
            return (
                "Não foi possível concluir a consulta com o modelo no momento "
                f"(o modelo gratuito pode estar ocupado ou indisponível). Detalhe: {exc}"
            )
