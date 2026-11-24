from types import SimpleNamespace

import pandas as pd
import pytest

from data_pilot.config import Settings
from data_pilot.llm_agent import DataPilotAgent, requires_data_tool


def make_response(message: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)]
    )


class FakeCompletions:
    def __init__(self, responses: list[SimpleNamespace]) -> None:
        self.responses = responses
        self.calls: list[dict] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        return self.responses.pop(0)


class FakeClient:
    def __init__(self, responses: list[SimpleNamespace]) -> None:
        self.chat = SimpleNamespace(
            completions=FakeCompletions(responses)
        )


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
    tool_messages = [
        message
        for message in second_call["messages"]
        if message["role"] == "tool"
    ]

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

    assert "Não foi possível consultar" in answer
    assert len(client.chat.completions.calls) == 1
