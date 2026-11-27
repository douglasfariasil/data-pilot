from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd

from data_pilot.ui import chat_view


class FakeSessionState(dict):
    """Simula o acesso por atributo do st.session_state."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


def configurar_streamlit(monkeypatch):
    """Prepara uma simulação do Streamlit para os testes."""
    streamlit_mock = MagicMock()
    streamlit_mock.session_state = FakeSessionState()
    streamlit_mock.chat_input.return_value = "Qual foi a receita total?"

    monkeypatch.setattr(chat_view, "st", streamlit_mock)
    monkeypatch.setattr(
        chat_view,
        "settings",
        SimpleNamespace(openrouter_api_key="test-key"),
    )

    return streamlit_mock


def test_chat_view_exibe_e_armazena_resposta_do_agente(
    monkeypatch,
):
    streamlit_mock = configurar_streamlit(monkeypatch)
    agent_mock = MagicMock()
    agent_mock.ask.return_value = "A receita total foi R$ 1.000,00."

    monkeypatch.setattr(
        chat_view,
        "DataPilotAgent",
        lambda df: agent_mock,
    )

    chat_view.render_chat_view(pd.DataFrame())

    agent_mock.ask.assert_called_once_with("Qual foi a receita total?")

    assert streamlit_mock.session_state["chat_messages"][-1] == {
        "role": "assistant",
        "content": "A receita total foi R$ 1.000,00.",
    }


def test_chat_view_trata_excecao_inesperada(
    monkeypatch,
):
    streamlit_mock = configurar_streamlit(monkeypatch)
    agent_mock = MagicMock()
    agent_mock.ask.side_effect = RuntimeError("informação interna sensível")

    logger_mock = MagicMock()

    monkeypatch.setattr(
        chat_view,
        "DataPilotAgent",
        lambda df: agent_mock,
    )
    monkeypatch.setattr(chat_view, "logger", logger_mock)

    chat_view.render_chat_view(pd.DataFrame())

    resposta = streamlit_mock.session_state["chat_messages"][-1]["content"]

    assert resposta == (
        "Ocorreu um erro inesperado ao processar sua pergunta. "
        "Tente novamente em instantes."
    )

    logger_mock.error.assert_called_once()

    dados_log = str(logger_mock.error.call_args)

    assert "RuntimeError" in dados_log
    assert "informação interna sensível" not in dados_log
