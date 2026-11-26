import json

import pandas as pd

from data_pilot.tools import (
    MAX_CATEGORY_LENGTH,
    MAX_TOOL_ARGUMENT_BYTES,
    build_tool_registry,
    execute_tool_call,
    get_tool_definitions,
)


def test_tool_definitions_and_execution() -> None:
    df = pd.DataFrame(
        {
            "order_id": ["ORD-1", "ORD-2"],
            "customer_id": ["C1", "C2"],
            "category": ["Livros", "Livros"],
            "status": ["concluido", "concluido"],
            "order_date": pd.to_datetime(["2026-03-01", "2026-03-10"]),
            "revenue": [120.0, 80.0],
        }
    )

    tools = get_tool_definitions()
    tool_names = {t["function"]["name"] for t in tools}
    assert "get_total_revenue" in tool_names
    assert "get_average_ticket" in tool_names

    registry = build_tool_registry(df)
    output = execute_tool_call(
        tool_name="get_average_ticket",
        arguments_json='{"category": "Livros"}',
        registry=registry,
    )

    assert output == {"category": "Livros", "average_ticket": 100.0}

    output_customer = execute_tool_call(
        tool_name="get_average_revenue_per_customer",
        arguments_json='{"category": "Livros"}',
        registry=registry,
    )
    assert output_customer == {"category": "Livros", "average_revenue_per_customer": 100.0}


def test_execute_tool_rejects_invalid_json() -> None:
    registry = {"probe": lambda category=None: {"category": category}}

    output = execute_tool_call(
        "probe",
        "{invalid",
        registry,
    )

    assert output == {"error": "Argumentos inválidos enviados pelo modelo."}


def test_execute_tool_rejects_non_object_arguments() -> None:
    registry = {"probe": lambda category=None: {"category": category}}

    output = execute_tool_call(
        "probe",
        '["Livros"]',
        registry,
    )

    assert output == {"error": "Os argumentos devem ser enviados como um objeto JSON."}


def test_execute_tool_rejects_unexpected_arguments() -> None:
    registry = {"probe": lambda category=None: {"category": category}}

    output = execute_tool_call(
        "probe",
        '{"category": "Livros", "limit": 10}',
        registry,
    )

    assert output == {"error": "A ferramenta recebeu argumentos não permitidos."}


def test_execute_tool_rejects_invalid_category_type() -> None:
    registry = {"probe": lambda category=None: {"category": category}}

    output = execute_tool_call(
        "probe",
        '{"category": 123}',
        registry,
    )

    assert output == {"error": "O argumento category deve ser um texto ou null."}


def test_execute_tool_normalizes_category() -> None:
    registry = {"probe": lambda category=None: {"category": category}}

    output = execute_tool_call(
        "probe",
        '{"category": "  Livros  "}',
        registry,
    )

    assert output == {"category": "Livros"}


def test_execute_tool_rejects_unknown_tool() -> None:
    output = execute_tool_call(
        "tool_inexistente",
        "{}",
        {},
    )

    assert output == {"error": "A ferramenta solicitada não está disponível."}


def test_execute_tool_does_not_expose_internal_exception() -> None:
    def failing_tool() -> None:
        raise RuntimeError("detalhe interno confidencial")

    output = execute_tool_call(
        "failing_tool",
        None,
        {"failing_tool": failing_tool},
    )

    assert output == {"error": "Não foi possível executar a ferramenta solicitada."}
    assert "detalhe interno confidencial" not in str(output)


def test_execute_tool_rejects_arguments_above_size_limit() -> None:
    registry = {"probe": lambda category=None: {"category": category}}

    arguments = '{"category": "' + (
        "a" * MAX_TOOL_ARGUMENT_BYTES
    ) + '"}'

    output = execute_tool_call("probe", arguments, registry)

    assert output == {
        "error": "Os argumentos da ferramenta ultrapassaram o limite permitido."
    }


def test_execute_tool_rejects_category_above_length_limit() -> None:
    registry = {"probe": lambda category=None: {"category": category}}

    arguments = '{"category": "' + (
        "a" * (MAX_CATEGORY_LENGTH + 1)
    ) + '"}'

    output = execute_tool_call("probe", arguments, registry)

    assert output == {
        "error": "O argumento category ultrapassou o limite permitido."
    }


def test_execute_tool_accepts_category_at_length_limit() -> None:
    registry = {"probe": lambda category=None: {"category": category}}

    category = "a" * MAX_CATEGORY_LENGTH
    arguments = json.dumps({"category": category})

    output = execute_tool_call("probe", arguments, registry)

    assert output == {"category": category}


def test_execute_tool_rejects_non_string_arguments() -> None:
    registry = {"probe": lambda category=None: {"category": category}}

    output = execute_tool_call("probe", 123, registry)

    assert output == {
        "error": "Os argumentos da ferramenta possuem um formato inválido."
    }
