import pandas as pd
import plotly.express as px
import streamlit as st

from data_pilot import metrics


def _format_brl(value: float) -> str:
    """Formata valores monetários no padrão brasileiro para exibição visual."""
    formatted = f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {formatted}"


def render_dashboard(df: pd.DataFrame, category_col: str = "category") -> None:
    """Renderiza os cards de métricas, filtro de categoria e gráfico mensal no Plotly."""
    st.subheader("Painel de Vendas")
    st.caption(
        "Métricas calculadas em Python a partir dos pedidos concluídos "
        "(e taxa de cancelamento sobre o total de pedidos do recorte)."
    )

    if df.empty:
        st.warning(
            "A base de dados está vazia no momento. Adicione registros ao `data/raw/sales.csv` "
            "e ajuste os nomes das colunas em `data_prep.py` para visualizar as métricas."
        )
        return

    categories = ["Todas"]
    if category_col in df.columns:
        unique_cats = sorted(
            {
                str(val).strip()
                for val in df[category_col].dropna().unique()
                if str(val).strip()
            }
        )
        categories.extend(unique_cats)

    selected_category = st.selectbox("Filtrar por categoria", options=categories, index=0)
    category_arg = None if selected_category == "Todas" else selected_category

    rev = metrics.total_revenue(df, category=category_arg, category_col=category_col)
    orders = metrics.total_orders(df, category=category_arg)
    ticket = metrics.average_ticket(df, category=category_arg)
    customers = metrics.unique_customers(df, category=category_arg)
    avg_rev_customer = metrics.average_revenue_per_customer(df, category=category_arg)
    cancel_rate = metrics.cancellation_rate(df, category=category_arg)

    col1, col2, col3 = st.columns(3)
    col1.metric("Receita Total", _format_brl(rev))
    col2.metric("Pedidos Concluídos", f"{orders}")
    col3.metric("Ticket Médio", _format_brl(ticket))

    col4, col5, col6 = st.columns(3)
    col4.metric("Clientes Únicos", f"{customers}")
    col5.metric("Taxa de Cancelamento", f"{cancel_rate:.2f}%".replace(".", ","))
    col6.metric("Receita Média / Cliente", _format_brl(avg_rev_customer))

    st.divider()
    st.markdown("### Receita Mensal (Pedidos Concluídos)")

    monthly_df = metrics.monthly_revenue(df, category=category_arg)
    if monthly_df.empty:
        st.info("Não há dados de receita mensal disponíveis para o filtro selecionado.")
        return

    fig = px.bar(
        monthly_df,
        x="month",
        y="revenue",
        labels={"month": "Mês", "revenue": "Receita (R$)"},
        title=f"Receita por Mês — Categoria: {selected_category}",
    )
    fig.update_layout(xaxis_type="category")
    st.plotly_chart(fig, use_container_width=True)
