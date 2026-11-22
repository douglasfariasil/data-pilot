from pathlib import Path
import pandas as pd
import streamlit as st

from data_pilot.config import settings
from data_pilot.data_prep import load_clean_sales, prepare_sales_pipeline
from data_pilot.ui.chat_view import render_chat_view
from data_pilot.ui.dashboard import render_dashboard


def _load_dataframe(raw_path: Path, clean_path: Path) -> pd.DataFrame:
    """Tenta carregar ou preparar os dados limpos; retorna DataFrame vazio se CSV estiver vazio."""
    if not raw_path.exists() or raw_path.stat().st_size == 0:
        if clean_path.exists() and clean_path.stat().st_size > 0:
            return load_clean_sales(clean_path)
        return pd.DataFrame()

    try:
        return prepare_sales_pipeline(raw_path=raw_path, clean_path=clean_path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def main() -> None:
    st.set_page_config(page_title=settings.app_title, layout="wide")
    st.title(settings.app_title)

    menu = st.sidebar.radio(
        "Navegação",
        options=["Painel", "Conversa com os Dados"],
        index=0,
    )

    df = _load_dataframe(settings.raw_sales_path, settings.clean_sales_path)

    if menu == "Painel":
        render_dashboard(df)
    else:
        render_chat_view(df)


if __name__ == "__main__":
    main()
