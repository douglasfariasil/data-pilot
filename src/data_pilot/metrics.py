import pandas as pd

COMPLETED_STATUSES = {"concluido", "concluído", "completed", "entregue", "paid", "pago"}
CANCELLED_STATUSES = {"cancelado", "cancelled", "canceled", "cancelada"}


def _apply_category_filter(
    df: pd.DataFrame,
    category: str | None = None,
    category_col: str = "category",
) -> pd.DataFrame:
    """Filtra por categoria quando informada e existente no DataFrame."""
    if df.empty or not category or category_col not in df.columns:
        return df

    target = category.strip().lower()
    if target in {"todas", "all", ""}:
        return df

    series = df[category_col].astype("string").str.strip().str.lower()
    return df[series == target]


def filter_completed_orders(
    df: pd.DataFrame,
    status_col: str = "status",
) -> pd.DataFrame:
    """Retorna apenas os registros de pedidos concluídos."""
    if df.empty or status_col not in df.columns:
        return df.iloc[0:0].copy()

    status_norm = df[status_col].astype("string").str.strip().str.lower()
    return df[status_norm.isin(COMPLETED_STATUSES)].copy()


def total_revenue(
    df: pd.DataFrame,
    category: str | None = None,
    revenue_col: str = "revenue",
) -> float:
    """Calcula a receita total de pedidos concluídos (opcionalmente por categoria)."""
    filtered = _apply_category_filter(df, category=category)
    completed = filter_completed_orders(filtered)
    if completed.empty or revenue_col not in completed.columns:
        return 0.0

    revenue = pd.to_numeric(completed[revenue_col], errors="coerce").dropna()
    return float(revenue.sum()) if not revenue.empty else 0.0


def total_orders(
    df: pd.DataFrame,
    category: str | None = None,
    order_id_col: str = "order_id",
) -> int:
    """Calcula a quantidade de pedidos concluídos (opcionalmente por categoria)."""
    filtered = _apply_category_filter(df, category=category)
    completed = filter_completed_orders(filtered)
    if completed.empty:
        return 0

    if order_id_col in completed.columns:
        return int(completed[order_id_col].nunique(dropna=True))
    return int(len(completed))


def average_ticket(
    df: pd.DataFrame,
    category: str | None = None,
    revenue_col: str = "revenue",
    order_id_col: str = "order_id",
) -> float:
    """Calcula o ticket médio (receita total / total de pedidos concluídos)."""
    orders_count = total_orders(df, category=category, order_id_col=order_id_col)
    if orders_count == 0:
        return 0.0

    revenue = total_revenue(df, category=category, revenue_col=revenue_col)
    return float(revenue / orders_count)


def unique_customers(
    df: pd.DataFrame,
    category: str | None = None,
    customer_id_col: str = "customer_id",
) -> int:
    """Calcula o total de clientes únicos em pedidos concluídos."""
    filtered = _apply_category_filter(df, category=category)
    completed = filter_completed_orders(filtered)
    if completed.empty or customer_id_col not in completed.columns:
        return 0

    return int(completed[customer_id_col].nunique(dropna=True))


def average_revenue_per_customer(
    df: pd.DataFrame,
    category: str | None = None,
    revenue_col: str = "revenue",
    customer_id_col: str = "customer_id",
) -> float:
    """Calcula a receita média por cliente único em pedidos concluídos."""
    customers_count = unique_customers(
        df,
        category=category,
        customer_id_col=customer_id_col,
    )
    if customers_count == 0:
        return 0.0

    revenue = total_revenue(df, category=category, revenue_col=revenue_col)
    return float(revenue / customers_count)


def cancellation_rate(
    df: pd.DataFrame,
    category: str | None = None,
    status_col: str = "status",
    order_id_col: str = "order_id",
) -> float:
    """Calcula a taxa de cancelamento (%) em relação ao total de pedidos do recorte."""
    filtered = _apply_category_filter(df, category=category)
    if filtered.empty or status_col not in filtered.columns:
        return 0.0

    if order_id_col in filtered.columns:
        unique_orders = filtered.drop_duplicates(subset=[order_id_col])
    else:
        unique_orders = filtered

    total_count = len(unique_orders)
    if total_count == 0:
        return 0.0

    status_norm = unique_orders[status_col].astype("string").str.strip().str.lower()
    cancelled_count = int(status_norm.isin(CANCELLED_STATUSES).sum())
    return float((cancelled_count / total_count) * 100.0)


def monthly_revenue(
    df: pd.DataFrame,
    category: str | None = None,
    date_col: str = "order_date",
    revenue_col: str = "revenue",
) -> pd.DataFrame:
    """Agrega a receita mensal de pedidos concluídos para o gráfico do painel."""
    filtered = _apply_category_filter(df, category=category)
    completed = filter_completed_orders(filtered)

    if completed.empty or date_col not in completed.columns or revenue_col not in completed.columns:
        return pd.DataFrame(columns=["month", "revenue"])

    temp = completed.copy()
    temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
    temp[revenue_col] = pd.to_numeric(temp[revenue_col], errors="coerce")
    temp = temp.dropna(subset=[date_col, revenue_col])

    if temp.empty:
        return pd.DataFrame(columns=["month", "revenue"])

    temp["month"] = temp[date_col].dt.to_period("M").astype(str)
    grouped = (
        temp.groupby("month", as_index=False)[revenue_col]
        .sum()
        .rename(columns={revenue_col: "revenue"})
        .sort_values("month")
    )
    return grouped.reset_index(drop=True)
