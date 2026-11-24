from pathlib import Path

import pandas as pd


def inspect_raw_columns(df: pd.DataFrame) -> list[str]:
    """Retorna os nomes de colunas normalizados para apoiar o mapeamento inicial."""
    return [col.strip().lower() for col in df.columns]


def load_raw_sales(raw_path: Path) -> pd.DataFrame:
    """Lê o CSV bruto sem assumir separador/decimal do arquivo limpo."""
    return pd.read_csv(raw_path)


def parse_order_dates(series: pd.Series, dayfirst: bool = True) -> pd.Series:
    """Converte datas ISO e brasileiras de forma segura."""

    values = series.astype("string").str.strip()

    # Identifica datas ISO: YYYY-MM-DD
    iso_mask = values.str.match(
        r"^\d{4}-\d{2}-\d{2}(?:$|[ T])",
        na=False,
    )

    parsed = pd.Series(
        pd.NaT,
        index=series.index,
        dtype="datetime64[ns]",
    )

    # Datas ISO: preserva ano-mês-dia.
    if iso_mask.any():
        parsed.loc[iso_mask] = pd.to_datetime(
            values.loc[iso_mask],
            errors="coerce",
            format="ISO8601",
        )

    # Demais formatos: respeita a configuração dayfirst.
    other_mask = ~iso_mask & values.notna()

    if other_mask.any():
        parsed.loc[other_mask] = pd.to_datetime(
            values.loc[other_mask],
            errors="coerce",
            format="mixed",
            dayfirst=dayfirst,
        )

    return parsed


def normalize_status(series: pd.Series) -> pd.Series:
    """Padroniza textos de status (ex.: minúsculas, sem espaços extras)."""
    return series.astype("string").str.strip().str.lower()


def compute_revenue(df: pd.DataFrame, quantity_col: str, unit_price_col: str) -> pd.Series:
    """Calcula a receita a partir de colunas numéricas convertidas com segurança."""
    qty = pd.to_numeric(df[quantity_col], errors="coerce")
    price = pd.to_numeric(df[unit_price_col], errors="coerce")
    return qty * price


def clean_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica limpeza, datas, status, deduplicação e coluna de receita.

    Ajuste os nomes das colunas-chave assim que inspecionar o sales.csv real.
    """
    cleaned = df.copy()
    cleaned.columns = inspect_raw_columns(cleaned)

    # 1. Remover duplicatas exatas iniciais
    cleaned = cleaned.drop_duplicates()

    # 2. Converter datas (substitua 'order_date' pelo nome real após inspecionar)
    if "order_date" in cleaned.columns:
        cleaned["order_date"] = parse_order_dates(cleaned["order_date"], dayfirst=True)
        cleaned = cleaned.dropna(subset=["order_date"])

    # 3. Padronizar status (substitua 'status' pelo nome real após inspecionar)
    if "status" in cleaned.columns:
        cleaned["status"] = normalize_status(cleaned["status"])

    # 4. Criar coluna de receita (ajuste conforme as colunas numéricas do CSV)
    if {"quantity", "unit_price"}.issubset(cleaned.columns):
        cleaned["revenue"] = compute_revenue(cleaned, "quantity", "unit_price")

    return cleaned.reset_index(drop=True)


def save_clean_sales(df: pd.DataFrame, output_path: Path) -> Path:
    """Salva o dataset limpo usando ';' como separador e ',' como decimal."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, sep=";", decimal=",", index=False)
    return output_path


def load_clean_sales(clean_path: Path, date_col: str = "order_date") -> pd.DataFrame:
    """Lê o dataset limpo com ';' e ',' e restaura a tipagem de data."""
    df = pd.read_csv(clean_path, sep=";", decimal=",")
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    return df


def prepare_sales_pipeline(raw_path: Path, clean_path: Path) -> pd.DataFrame:
    """Executa limpeza completa, salva no padrão exigido e retorna o DataFrame."""
    raw_df = load_raw_sales(raw_path)
    clean_df = clean_sales_data(raw_df)
    save_clean_sales(clean_df, clean_path)
    return load_clean_sales(clean_path)
