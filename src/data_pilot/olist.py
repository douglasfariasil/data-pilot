from pathlib import Path

import pandas as pd


def build_olist_sales(source_dir: Path, output_path: Path) -> pd.DataFrame:
    """Consolida os arquivos públicos da Olist em uma base de vendas."""

    # 1. Carregar os arquivos originais
    orders = pd.read_csv(source_dir / "olist_orders_dataset.csv")
    items = pd.read_csv(source_dir / "olist_order_items_dataset.csv")
    customers = pd.read_csv(source_dir / "olist_customers_dataset.csv")
    products = pd.read_csv(source_dir / "olist_products_dataset.csv")
    translations = pd.read_csv(source_dir / "product_category_name_translation.csv")

    # 2. Validar as colunas necessárias
    required_columns = {
        "orders": {
            "order_id",
            "customer_id",
            "order_status",
            "order_purchase_timestamp",
        },
        "items": {
            "order_id",
            "order_item_id",
            "product_id",
            "price",
            "freight_value",
        },
        "customers": {"customer_id", "customer_unique_id"},
        "products": {"product_id", "product_category_name"},
        "translations": {
            "product_category_name",
            "product_category_name_english",
        },
    }

    datasets = {
        "orders": orders,
        "items": items,
        "customers": customers,
        "products": products,
        "translations": translations,
    }

    for name, required in required_columns.items():
        missing = required - set(datasets[name].columns)
        if missing:
            raise ValueError(f"Colunas ausentes em {name}: {sorted(missing)}")

    # 3. Preparar os dados de pedidos
    orders = orders[
        [
            "order_id",
            "customer_id",
            "order_status",
            "order_purchase_timestamp",
        ]
    ].copy()

    if orders["order_id"].duplicated().any():
        raise ValueError("A tabela de pedidos contém order_id duplicado.")

    # 4. Preparar os itens
    items = items[
        [
            "order_id",
            "order_item_id",
            "product_id",
            "price",
            "freight_value",
        ]
    ].copy()

    # 5. Relacionar pedidos e itens
    sales = orders.merge(
        items,
        on="order_id",
        how="left",
        validate="one_to_many",
    )

    # 6. Relacionar produtos e categorias
    products = products[["product_id", "product_category_name"]].copy()

    translations = translations[
        [
            "product_category_name",
            "product_category_name_english",
        ]
    ].copy()

    sales = sales.merge(
        products,
        on="product_id",
        how="left",
        validate="many_to_one",
    )

    sales = sales.merge(
        translations,
        on="product_category_name",
        how="left",
        validate="many_to_one",
    )

    # 7. Relacionar clientes
    customers = customers[["customer_id", "customer_unique_id"]].copy()

    if customers["customer_id"].duplicated().any():
        raise ValueError("A tabela de clientes contém customer_id duplicado.")

    sales = sales.merge(
        customers,
        on="customer_id",
        how="left",
        validate="many_to_one",
    )

    if sales["customer_unique_id"].isna().any():
        raise ValueError("Existem pedidos sem identificação de cliente único.")

    # 8. Normalizar status
    sales["status"] = (
        sales["order_status"]
        .astype("string")
        .str.strip()
        .str.lower()
        .replace(
            {
                "delivered": "completed",
                "canceled": "cancelled",
            }
        )
    )

    # 9. Normalizar datas
    sales["order_date"] = pd.to_datetime(
        sales["order_purchase_timestamp"],
        errors="coerce",
    )

    if sales["order_date"].isna().any():
        raise ValueError("Existem pedidos com datas de compra inválidas.")

    # 10. Tratar categorias ausentes
    sales["category"] = (
        sales["product_category_name_english"]
        .fillna(sales["product_category_name"])
        .astype("string")
        .str.strip()
        .str.lower()
    )

    no_item = sales["order_item_id"].isna()

    sales.loc[no_item, "category"] = "sem_itens"
    sales.loc[~no_item, "category"] = sales.loc[~no_item, "category"].fillna("sem_categoria")

    # 11. Preparar quantidade e preços
    sales["quantity"] = no_item.map({True: 0, False: 1}).astype("int64")

    sales["unit_price"] = pd.to_numeric(sales["price"], errors="coerce").fillna(0.0)

    sales["freight_value"] = pd.to_numeric(sales["freight_value"], errors="coerce").fillna(0.0)

    # 12. Calcular receita sem incluir frete
    sales["revenue"] = sales["quantity"] * sales["unit_price"]

    # 13. Selecionar as colunas do Data Pilot
    result = sales[
        [
            "order_id",
            "order_item_id",
            "customer_unique_id",
            "customer_id",
            "order_date",
            "status",
            "category",
            "quantity",
            "unit_price",
            "freight_value",
            "revenue",
        ]
    ].rename(
        columns={
            "customer_unique_id": "customer_id",
            "customer_id": "customer_record_id",
        }
    )

    # 14. Salvar o CSV consolidado
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    return result
