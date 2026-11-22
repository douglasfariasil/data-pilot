import pandas as pd

from data_pilot.tools import build_tool_registry, execute_tool_call, get_tool_definitions


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
