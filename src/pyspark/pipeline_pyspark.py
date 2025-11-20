from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    countDistinct,
    explode,
    expr,
    row_number,
    sum as F_sum,
    udf,
)
from pyspark.sql.types import BooleanType, StringType
from pyspark.sql.window import Window


def load_settings(path: Path) -> Dict[str, Any]:
    """Return configuration dictionary loaded from the YAML file at ``path``."""
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_path(base_dir: Path, raw_path: str) -> Path:
    """Resolve ``raw_path`` relative to ``base_dir`` unless it is already absolute."""
    candidate = Path(raw_path)
    return candidate if candidate.is_absolute() else (base_dir / candidate)


def controle_bool(v: Any) -> bool:
    """Normalize various truthy inputs (1, 'yes', etc.) into a real boolean."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if v is None:
        return False
    s = str(v).strip().lower()
    return s in ("1", "true", "yes", "y", "t")


def order_date_str(value: Any) -> str:
    """Convert a timestamp string into an ISO date (YYYY-MM-DD) using known formats."""
    value = str(value or "").strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).date().isoformat()
        except ValueError:
            continue
    raise ValueError(f"Format de date non reconnu: {value!r}")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    cfg_path = repo_root / "settings.yaml"
    cfg = load_settings(cfg_path)

    input_dir = resolve_path(repo_root, cfg.get("input_dir", "./data/march-input"))
    output_dir = resolve_path(repo_root, cfg.get("output_dir", "./data/out"))
    db_path = resolve_path(repo_root, cfg.get("db_path", "./data/sales_db.db"))
    csv_sep = cfg.get("csv_sep", ";")
    csv_encoding = cfg.get("csv_encoding", "utf-8")
    csv_float_format = cfg.get("csv_float_format", "%.2f")

    output_dir.mkdir(parents=True, exist_ok=True)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    orders_files = sorted(input_dir.glob("orders_*.json"))
    if not orders_files:
        raise FileNotFoundError(f"Aucun fichier orders_*.json dans {input_dir}")

    customers_file = input_dir / "customers.csv"
    refunds_file = input_dir / "refunds.csv"
    if not customers_file.exists():
        raise FileNotFoundError(f"Fichier clients introuvable: {customers_file}")
    if not refunds_file.exists():
        raise FileNotFoundError(f"Fichier remboursements introuvable: {refunds_file}")

    spark = (
        SparkSession.builder.appName("PipelinePySpark")
        .master("local[*]")
        .getOrCreate()
    )

    controle_bool_udf = udf(controle_bool, BooleanType())
    order_date_udf = udf(order_date_str, StringType())

    orders_df = spark.read.option("multiLine", "true").json(
        [str(p) for p in orders_files]
    )
    refunds_df = spark.read.csv(str(refunds_file), header=True, inferSchema=True)
    customers_df = spark.read.csv(str(customers_file), header=True, inferSchema=True)

    customers_clean = (
        customers_df.withColumn("is_active", controle_bool_udf(col("is_active")))
        .select("customer_id", "city", "is_active")
        .cache()
    )
    refunds_clean = (
        refunds_df.withColumn("amount", expr("try_cast(amount AS double)"))
        .na.fill({"amount": 0.0})
        .select("order_id", "amount")
    )

    orders_paid = orders_df.filter(col("payment_status") == "paid")
    orders_exploded = orders_paid.withColumn("item", explode("items"))
    orders_flat = orders_exploded.select(
        "order_id",
        "customer_id",
        "channel",
        "created_at",
        col("item.sku").alias("item_sku"),
        col("item.qty").alias("item_qty"),
        col("item.unit_price").alias("item_unit_price"),
    )

    neg_prices = orders_flat.filter(col("item_unit_price") < 0)
    neg_count = neg_prices.count()
    if neg_count:
        rejects_path = output_dir / "rejects_items.csv"
        neg_prices.toPandas().to_csv(
            rejects_path, index=False, encoding=csv_encoding, sep=csv_sep
        )
        print(f"{neg_count} lignes rejetées -> {rejects_path}")
    orders_positive = orders_flat.filter(col("item_unit_price") >= 0)

    window = Window.partitionBy("order_id").orderBy(col("created_at").asc())
    orders_dedup = (
        orders_positive.withColumn("rn", row_number().over(window))
        .filter(col("rn") == 1)
        .drop("rn")
    )

    orders_dedup = orders_dedup.withColumn(
        "line_gross", col("item_qty") * col("item_unit_price")
    )
    per_order = orders_dedup.groupBy(
        "order_id", "customer_id", "channel", "created_at"
    ).agg(
        F_sum("item_qty").alias("items_sold"),
        F_sum("line_gross").alias("gross_revenue_eur"),
    )

    per_order_active = (
        per_order.join(customers_clean, on="customer_id", how="left")
        .filter(col("is_active") == True)  # noqa: E712
        .withColumn("order_date", order_date_udf(col("created_at")))
    )

    refunds_sum = refunds_clean.groupBy("order_id").agg(
        F_sum("amount").alias("refunds_eur")
    )
    per_order_ref = (
        per_order_active.join(refunds_sum, on="order_id", how="left")
        .na.fill({"refunds_eur": 0.0})
        .select(
            "order_id",
            "customer_id",
            "city",
            "channel",
            "order_date",
            "created_at",
            "items_sold",
            "gross_revenue_eur",
            "refunds_eur",
        )
    )

    agg_df = per_order_ref.groupBy("order_date", "city", "channel").agg(
        countDistinct("order_id").alias("orders_count"),
        countDistinct("customer_id").alias("unique_customers"),
        F_sum("items_sold").alias("items_sold"),
        F_sum("gross_revenue_eur").alias("gross_revenue_eur"),
        F_sum("refunds_eur").alias("refunds_eur"),
    )
    agg_df = agg_df.withColumn(
        "net_revenue_eur", col("gross_revenue_eur") + col("refunds_eur")
    ).withColumnRenamed("order_date", "date")

    per_order_pd = per_order_ref.select(
        "order_id",
        "customer_id",
        "city",
        "channel",
        "order_date",
        "items_sold",
        "gross_revenue_eur",
    ).toPandas()
    agg_pd = agg_df.orderBy("date", "city", "channel").toPandas()

    with sqlite3.connect(db_path) as conn:
        per_order_pd.to_sql("orders_clean", conn, if_exists="replace", index=False)
        agg_pd.to_sql("daily_city_sales", conn, if_exists="replace", index=False)

    for date_value, sub_df in agg_pd.groupby("date"):
        out_path = output_dir / f"daily_summary_{date_value.replace('-', '')}.csv"
        sub_df[
            [
                "date",
                "city",
                "channel",
                "orders_count",
                "unique_customers",
                "items_sold",
                "gross_revenue_eur",
                "refunds_eur",
                "net_revenue_eur",
            ]
        ].to_csv(
            out_path,
            index=False,
            sep=csv_sep,
            encoding=csv_encoding,
            float_format=csv_float_format,
        )
        print(f"Ecriture du résumé : {out_path}")

    spark.stop()
