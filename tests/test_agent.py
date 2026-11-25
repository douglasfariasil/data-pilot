from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from data_pilot.config import Settings
from data_pilot.llm_agent import DataPilotAgent, requires_data_tool


def make_response(message: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeCompletions:
    def __init__(self, responses: list[SimpleNamespace]) -> None:
        self.responses = responses
        self.calls: list[dict] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        return self.responses.pop(0)


class FakeClient:
    def __init__(self, responses: list[SimpleNamespace]) -> None:
        self.chat = SimpleNamespace(completions=FakeCompletions(responses))


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": ["ORD-1"],
            "customer_id": ["C1"],
            "category": ["Livros"],
            "status": ["completed"],
            "order_date": pd.to_datetime(["2026-03-01"]),
            "revenue": [100.0],
        }
    )


@pytest.fixture
def agent(sample_df: pd.DataFrame) -> DataPilotAgent:
    return DataPilotAgent(
        sample_df,
        cfg=Settings(openrouter_api_key="test-key"),
    )


@pytest.mark.parametrize(
    "question",
    [
        "Qual foi o faturamento?",
        "Quanto vendemos?",
        "Quantos pedidos foram concluídos?",
        "Qual foi o ticket médio?",
        "Quantos clientes compraram?",
        "Qual foi a taxa de cancelamento?",
        "Compare as vendas por categoria.",
        "Qual foi a receita mensal?",
        "Qual foi o desempenho das vendas?",
        "Mostre a evolução mensal do faturamento.",
        "Quais categorias tiveram melhor resultado?",
        "Qual foi a tendência das vendas?",
        "Faça um ranking das categorias.",
        "Qual foi o maior faturamento mensal?",
        "Houve queda nas vendas?",
        "Mostre o crescimento do faturamento.",
    ],
)
def test_requires_data_tool_for_sales_questions(question: str) -> None:
    assert requires_data_tool(question) is True


@pytest.mark.parametrize(
    "question",
    [
        "Olá, tudo bem?",
        "Como funciona o Data Pilot?",
        "O que você consegue fazer?",
    ],
)
def test_does_not_require_tool_for_general_questions(question: str) -> None:
    assert requires_data_tool(question) is False


def test_agent_requires_tool_for_quantitative_question(
    agent: DataPilotAgent,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool_call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(
            name="get_total_revenue",
            arguments="{}",
        ),
    )

    first_response = make_response(
        SimpleNamespace(
            content=None,
            tool_calls=[tool_call],
        )
    )

    second_response = make_response(
        SimpleNamespace(
            content="A receita total foi R$ 100,00.",
            tool_calls=None,
        )
    )

    client = FakeClient([first_response, second_response])
    monkeypatch.setattr(agent, "_build_client", lambda: client)

    answer = agent.ask("Qual foi o faturamento?")

    assert answer == "A receita total foi R$ 100,00."

    first_call = client.chat.completions.calls[0]
    assert first_call["tool_choice"] == "required"

    second_call = client.chat.completions.calls[1]
    tool_messages = [message for message in second_call["messages"] if message["role"] == "tool"]

    assert len(tool_messages) == 1
    assert '"total_revenue": 100.0' in tool_messages[0]["content"]


def test_agent_does_not_answer_quantitative_question_without_tool(
    agent: DataPilotAgent,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_response = make_response(
        SimpleNamespace(
            content="O faturamento foi R$ 999.999,00.",
            tool_calls=None,
        )
    )

    client = FakeClient([first_response])
    monkeypatch.setattr(agent, "_build_client", lambda: client)

    answer = agent.ask("Qual foi o faturamento?")

    assert "Não foi possível consultar" in answer
    assert "999.999" not in answer


def test_agent_allows_general_question_without_tool(
    agent: DataPilotAgent,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_response = make_response(
        SimpleNamespace(
            content="O Data Pilot é um painel de análise de vendas.",
            tool_calls=None,
        )
    )

    client = FakeClient([first_response])
    monkeypatch.setattr(agent, "_build_client", lambda: client)

    answer = agent.ask("Como funciona o Data Pilot?")

    assert answer == "O Data Pilot é um painel de análise de vendas."
    assert client.chat.completions.calls[0]["tool_choice"] == "auto"


def test_agent_stops_when_tool_returns_error(
    agent: DataPilotAgent,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool_call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(
            name="get_total_revenue",
            arguments='{"unexpected": "valor"}',
        ),
    )

    first_response = make_response(
        SimpleNamespace(
            content=None,
            tool_calls=[tool_call],
        )
    )

    client = FakeClient([first_response])
    monkeypatch.setattr(agent, "_build_client", lambda: client)

    answer = agent.ask("Qual foi o faturamento?")

    assert "Não foi possível consultar" in answer
    assert len(client.chat.completions.calls) == 1


def test_agent_handles_missing_api_key(
    sample_df: pd.DataFrame,
) -> None:
    agent = DataPilotAgent(
        sample_df,
        cfg=Settings(openrouter_api_key=""),
    )

    answer = agent.ask("Olá, tudo bem?")

    assert "Não foi possível concluir sua consulta" in answer
    assert "OPENROUTER_API_KEY" not in answer


def test_agent_handles_first_api_call_failure(
    agent: DataPilotAgent,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create = Mock(side_effect=RuntimeError("OpenRouter indisponível"))

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    monkeypatch.setattr(agent, "_build_client", lambda: client)

    answer = agent.ask("Como funciona o Data Pilot?")

    assert "Não foi possível concluir sua consulta" in answer
    assert "OpenRouter indisponível" not in answer
    create.assert_called_once()


def test_agent_handles_second_api_call_failure(
    agent: DataPilotAgent,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool_call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(
            name="get_total_revenue",
            arguments="{}",
        ),
    )

    first_response = make_response(
        SimpleNamespace(
            content=None,
            tool_calls=[tool_call],
        )
    )

    create = Mock(
        side_effect=[
            first_response,
            RuntimeError("Falha ao gerar explicação"),
        ]
    )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    monkeypatch.setattr(agent, "_build_client", lambda: client)

    answer = agent.ask("Qual foi o faturamento?")

    assert "Não foi possível concluir sua consulta" in answer
    assert "Falha ao gerar explicação" not in answer
    assert create.call_count == 2


def test_agent_rejects_more_than_three_tool_calls(
    agent: DataPilotAgent,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool_calls = [
        SimpleNamespace(
            id=f"call-{index}",
            function=SimpleNamespace(
                name="get_total_revenue",
                arguments=f'{{"category": "Categoria {index}"}}',
            ),
        )
        for index in range(4)
    ]

    first_response = make_response(
        SimpleNamespace(
            content=None,
            tool_calls=tool_calls,
        )
    )

    client = FakeClient([first_response])
    monkeypatch.setattr(agent, "_build_client", lambda: client)

    answer = agent.ask("Qual foi o faturamento?")

    assert "operações demais" in answer
    assert len(client.chat.completions.calls) == 1


def test_agent_rejects_unknown_tool(
    agent: DataPilotAgent,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool_call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(
            name="tool_inexistente",
            arguments="{}",
        ),
    )

    first_response = make_response(
        SimpleNamespace(
            content=None,
            tool_calls=[tool_call],
        )
    )

    client = FakeClient([first_response])
    monkeypatch.setattr(agent, "_build_client", lambda: client)

    answer = agent.ask("Qual foi o faturamento?")

    assert "Não foi possível realizar a consulta" in answer
    assert len(client.chat.completions.calls) == 1


def test_agent_rejects_duplicate_tool_calls(
    agent: DataPilotAgent,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool_calls = [
        SimpleNamespace(
            id=f"call-{index}",
            function=SimpleNamespace(
                name="get_total_revenue",
                arguments="{}",
            ),
        )
        for index in range(2)
    ]

    first_response = make_response(
        SimpleNamespace(
            content=None,
            tool_calls=tool_calls,
        )
    )

    client = FakeClient([first_response])
    monkeypatch.setattr(agent, "_build_client", lambda: client)

    answer = agent.ask("Qual foi o faturamento?")

    assert "operações repetidas" in answer
    assert len(client.chat.completions.calls) == 1


def test_agent_accepts_three_distinct_tool_calls(
    agent: DataPilotAgent,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool_names = [
        "get_total_revenue",
        "get_total_orders",
        "get_average_ticket",
    ]

    tool_calls = [
        SimpleNamespace(
            id=f"call-{index}",
            function=SimpleNamespace(
                name=tool_name,
                arguments="{}",
            ),
        )
        for index, tool_name in enumerate(tool_names)
    ]

    first_response = make_response(
        SimpleNamespace(
            content=None,
            tool_calls=tool_calls,
        )
    )

    second_response = make_response(
        SimpleNamespace(
            content="Consultei as três métricas.",
            tool_calls=None,
        )
    )

    client = FakeClient([first_response, second_response])
    monkeypatch.setattr(agent, "_build_client", lambda: client)

    answer = agent.ask("Qual foi o faturamento?")

    assert answer == "Consultei as três métricas."
    assert len(client.chat.completions.calls) == 2

    second_call_messages = client.chat.completions.calls[1]["messages"]
    tool_messages = [message for message in second_call_messages if message["role"] == "tool"]

    assert len(tool_messages) == 3


def test_agent_detects_duplicate_calls_with_different_json_format(
    agent: DataPilotAgent,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool_calls = [
        SimpleNamespace(
            id="call-1",
            function=SimpleNamespace(
                name="get_total_revenue",
                arguments='{"category": "Livros"}',
            ),
        ),
        SimpleNamespace(
            id="call-2",
            function=SimpleNamespace(
                name="get_total_revenue",
                arguments='{ "category" : "Livros" }',
            ),
        ),
    ]

    first_response = make_response(
        SimpleNamespace(
            content=None,
            tool_calls=tool_calls,
        )
    )

    client = FakeClient([first_response])
    monkeypatch.setattr(agent, "_build_client", lambda: client)

    answer = agent.ask("Qual foi o faturamento?")

    assert "operações repetidas" in answer
    assert len(client.chat.completions.calls) == 1


def test_agent_builds_client_with_timeout_and_retries_disabled(
    agent: DataPilotAgent,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client_mock = Mock()

    openai_mock = Mock(return_value=client_mock)

    monkeypatch.setattr("data_pilot.llm_agent.OpenAI", openai_mock)

    client = agent._build_client()

    assert client is client_mock

    openai_mock.assert_called_once_with(
        api_key="test-key",
        base_url=agent.cfg.openrouter_base_url,
        timeout=30.0,
        max_retries=0,
    )


def test_agent_handles_first_response_without_choices(
    agent: DataPilotAgent,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient([SimpleNamespace(choices=[])])

    monkeypatch.setattr(agent, "_build_client", lambda: client)

    answer = agent.ask("Como funciona o Data Pilot?")

    assert "Não foi possível concluir sua consulta" in answer
    assert len(client.chat.completions.calls) == 1


def test_agent_handles_second_response_without_choices(
    agent: DataPilotAgent,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool_call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(
            name="get_total_revenue",
            arguments="{}",
        ),
    )

    first_response = make_response(
        SimpleNamespace(
            content=None,
            tool_calls=[tool_call],
        )
    )

    second_response = SimpleNamespace(choices=[])

    client = FakeClient([first_response, second_response])

    monkeypatch.setattr(agent, "_build_client", lambda: client)

    answer = agent.ask("Qual foi o faturamento?")

    assert "Não foi possível concluir sua consulta" in answer
    assert len(client.chat.completions.calls) == 2
