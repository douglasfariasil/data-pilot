from pathlib import Path

import pandas as pd

from data_pilot.data_prep import (
    clean_sales_data,
    load_clean_sales,
    parse_order_dates,
    prepare_sales_pipeline,
    save_clean_sales,
)


def _sample_raw_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": ["ORD-1", "ORD-1", "ORD-2", "ORD-3", "ORD-4"],
            "customer_id": ["C1", "C1", "C2", "C3", "C4"],
            "category": ["Eletronicos", "Eletronicos", "Livros", "Eletronicos", "Casa"],
            "status": [" Concluído ", " Concluído ", "CANCELADO", "concluido", "Concluído"],
            "order_date": ["10/01/2026", "10/01/2026", "15/01/2026", "20/02/2026", "data_invalida"],
            "quantity": [2, 2, 1, 3, 1],
            "unit_price": [150.5, 150.5, 80.0, 100.0, 50.0],
        }
    )


def test_clean_sales_data_removes_duplicates_and_computes_revenue() -> None:
    cleaned = clean_sales_data(_sample_raw_df())

    # Linha duplicada removida e data inválida descartada -> restam ORD-1, ORD-2 e ORD-3
    assert len(cleaned) == 3
    assert list(cleaned["order_id"]) == ["ORD-1", "ORD-2", "ORD-3"]
    assert list(cleaned["status"]) == ["concluído", "cancelado", "concluido"]
    assert pd.api.types.is_datetime64_any_dtype(cleaned["order_date"])
    assert "revenue" in cleaned.columns
    assert cleaned.loc[cleaned["order_id"] == "ORD-1", "revenue"].iloc[0] == 301.0


def test_save_and_load_clean_sales_uses_semicolon_and_comma(tmp_path: Path) -> None:
    cleaned = clean_sales_data(_sample_raw_df())
    out_file = tmp_path / "sales_clean.csv"

    save_clean_sales(cleaned, out_file)
    raw_text = out_file.read_text(encoding="utf-8")

    # Verifica separador ';' e decimal ',' no arquivo salvo
    assert ";" in raw_text
    assert "301,0" in raw_text or "301,00" in raw_text

    reloaded = load_clean_sales(out_file)
    assert len(reloaded) == len(cleaned)
    assert pd.api.types.is_datetime64_any_dtype(reloaded["order_date"])
    assert reloaded.loc[reloaded["order_id"] == "ORD-1", "revenue"].iloc[0] == 301.0


def test_prepare_sales_pipeline_end_to_end(tmp_path: Path) -> None:
    raw_file = tmp_path / "sales.csv"
    clean_file = tmp_path / "sales_clean.csv"
    _sample_raw_df().to_csv(raw_file, index=False)

    result = prepare_sales_pipeline(raw_path=raw_file, clean_path=clean_file)

    assert clean_file.exists()
    assert len(result) == 3
    assert "revenue" in result.columns


def test_parse_order_dates_supports_iso_and_brazilian_formats() -> None:
    dates = pd.Series(
        [
            "2016-10-02 12:30:00",
            "10/01/2026",
            "data_invalida",
        ]
    )

    parsed = parse_order_dates(dates, dayfirst=True)

    assert parsed.iloc[0] == pd.Timestamp("2016-10-02 12:30:00")
    assert parsed.iloc[1] == pd.Timestamp("2026-01-10")
    assert pd.isna(parsed.iloc[2])
