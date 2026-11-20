import pandas as pd


def filter_completed_orders(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna apenas os registros com status de pedido concluído."""
    raise NotImplementedError


def total_revenue(df: pd.DataFrame, category: str | None = None) -> float:
    """Calcula a receita total de pedidos concluídos (opcionalmente filtrada por categoria)."""
    raise NotImplementedError


def total_orders(df: pd.DataFrame, category: str | None = None) -> int:
    """Calcula a quantidade de pedidos concluídos (opcionalmente filtrada por categoria)."""
    raise NotImplementedError


def average_ticket(df: pd.DataFrame, category: str | None = None) -> float:
    """Calcula o ticket médio dos pedidos concluídos (opcionalmente por categoria)."""
    raise NotImplementedError


def unique_customers(df: pd.DataFrame, category: str | None = None) -> int:
    """Calcula o total de clientes únicos em pedidos concluídos."""
    raise NotImplementedError


def cancellation_rate(df: pd.DataFrame, category: str | None = None) -> float:
    """Calcula a taxa de cancelamento em relação ao total de pedidos."""
    raise NotImplementedError


def monthly_revenue(df: pd.DataFrame, category: str | None = None) -> pd.DataFrame:
    """Agrega a receita mensal de pedidos concluídos para o gráfico do painel."""
    raise NotImplementedError
