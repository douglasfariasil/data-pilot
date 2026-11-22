import pandas as pd

from data_pilot import metrics


def _sample_clean_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": ["ORD-1", "ORD-2", "ORD-3", "ORD-4"],
            "customer_id": ["C1", "C2", "C1", "C3"],
            "category": ["Eletronicos", "Eletronicos", "Livros", "Eletronicos"],
            "status": ["concluido", "cancelado", "concluido", "concluido"],
            "order_date": pd.to_datetime(
                ["2026-01-10", "2026-01-15", "2026-02-05", "2026-02-20"]
            ),
            "revenue": [200.0, 150.0, 100.0, 300.0],
        }
    )


def test_metrics_on_completed_orders() -> None:
    df = _sample_clean_df()

    # Pedidos concluídos: ORD-1 (200), ORD-3 (100), ORD-4 (300) -> total 600.0 em 3 pedidos
    assert metrics.total_revenue(df) == 600.0
    assert metrics.total_orders(df) == 3
    assert metrics.average_ticket(df) == 200.0
    # Clientes únicos em pedidos concluídos: C1 (ORD-1, ORD-3) e C3 (ORD-4) -> 2 clientes
    assert metrics.unique_customers(df) == 2
    assert metrics.average_revenue_per_customer(df) == 300.0
    # 1 cancelado (ORD-2) em 4 pedidos totais -> 25.0%
    assert metrics.cancellation_rate(df) == 25.0


def test_metrics_filtered_by_category() -> None:
    df = _sample_clean_df()

    # Categoria 'Eletronicos': ORD-1 (200 concluído), ORD-2 (150 cancelado), ORD-4 (300 concluído)
    assert metrics.total_revenue(df, category="Eletronicos") == 500.0
    assert metrics.total_orders(df, category="Eletronicos") == 2
    assert metrics.average_ticket(df, category="Eletronicos") == 250.0
    assert metrics.unique_customers(df, category="Eletronicos") == 2
    assert metrics.average_revenue_per_customer(df, category="Eletronicos") == 250.0
    # 1 cancelado em 3 pedidos da categoria -> 33.33...%
    assert round(metrics.cancellation_rate(df, category="Eletronicos"), 2) == 33.33


def test_monthly_revenue_grouping() -> None:
    df = _sample_clean_df()
    monthly = metrics.monthly_revenue(df)

    assert list(monthly["month"]) == ["2026-01", "2026-02"]
    assert list(monthly["revenue"]) == [200.0, 400.0]


def test_metrics_handle_empty_dataframe() -> None:
    empty_df = pd.DataFrame()

    assert metrics.total_revenue(empty_df) == 0.0
    assert metrics.total_orders(empty_df) == 0
    assert metrics.average_ticket(empty_df) == 0.0
    assert metrics.unique_customers(empty_df) == 0
    assert metrics.cancellation_rate(empty_df) == 0.0
    assert metrics.monthly_revenue(empty_df).empty
