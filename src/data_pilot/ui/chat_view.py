import logging

import pandas as pd
import streamlit as st

from data_pilot.config import settings
from data_pilot.llm_agent import DataPilotAgent
from data_pilot.logging_config import get_request_id

logger = logging.getLogger(__name__)


def render_chat_view(df: pd.DataFrame) -> None:
    """Renderiza a interface de chat que consulta métricas calculadas em Python."""
    st.subheader("Conversa com os Dados")
    st.caption(
        "Faça perguntas em português. O Python calcula as métricas e o modelo "
        "apenas explica os resultados retornados pelas ferramentas."
    )

    if not settings.openrouter_api_key:
        st.warning(
            "Configure `OPENROUTER_API_KEY` no seu arquivo `.env` (a partir do `.env.example`) "
            "para habilitar o chat com o OpenRouter."
        )

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": (
                    "Olá! Pergunte sobre receita total, pedidos concluídos, ticket médio, "
                    "clientes únicos, taxa de cancelamento ou receita mensal."
                ),
            }
        ]

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_prompt = st.chat_input("Ex.: Qual foi a receita total de pedidos concluídos?")
    if not user_prompt:
        return

    st.session_state.chat_messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consultando métricas e gerando resposta..."):
            try:
                agent = DataPilotAgent(df=df)
                answer = agent.ask(user_prompt)
            except Exception as exc:
                logger.error(
                    "Falha inesperada na interface do chat | "
                    "request_id=%s | error_type=%s",
                    get_request_id(),
                    type(exc).__name__,
                )
                answer = (
                    "Ocorreu um erro inesperado ao processar sua pergunta. "
                    "Tente novamente em instantes."
                )

            st.markdown(answer)

    st.session_state.chat_messages.append({"role": "assistant", "content": answer})
