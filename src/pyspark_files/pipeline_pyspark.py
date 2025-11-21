from __future__ import annotations
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import yaml
import pandas as pd
from pyspark.sql import SparkSession, DataFrame as SparkDataFrame
from pyspark.sql.functions import (
    col,
    countDistinct,
    explode,
    expr,
    row_number,
    sum as F_sum,
    when,
    lower,
    trim,
    lit,
    to_date,
    coalesce,
    date_format,
)
from pyspark.sql.window import Window


# -------------------------
# Helpers / small utilities
# -------------------------
def load_settings(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def resolve_path(base_dir: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    return candidate if candidate.is_absolute() else (base_dir / candidate)


# -------------------------
# Spark session management
# -------------------------
def build_spark(app_name: str = "PipelinePySpark") -> SparkSession:
    return SparkSession.builder.appName(app_name).master("local[*]").getOrCreate()


# -------------------------
# IO / Reading
# -------------------------
def list_order_files(input_dir: Path) -> list[Path]:
    return sorted(input_dir.glob("orders_*.json"))


def read_sources(
    spark: SparkSession,
    orders_files: Iterable[Path],
    customers_file: Path,
    refunds_file: Path,
    multi_line: bool = True,
) -> Tuple[SparkDataFrame, SparkDataFrame, SparkDataFrame]:
    orders_df = spark.read.option("multiLine", "true" if multi_line else "false").json(
        [str(p) for p in orders_files]
    )
    refunds_df = spark.read.csv(str(refunds_file), header=True, inferSchema=True)
    customers_df = spark.read.csv(str(customers_file), header=True, inferSchema=True)
    return orders_df, refunds_df, customers_df


# -------------------------
# Cleaning / transformations
# -------------------------
def clean_customers(customers_df: SparkDataFrame) -> SparkDataFrame:
    """
    Nettoie les clients et normalise la colonne is_active à un booléen
    en pur Spark (sans UDF Python) pour éviter les problèmes de pickling
    et de chemins de modules dans les workers.
    """
    with_num = customers_df.withColumn(
        "is_active_num", expr("try_cast(is_active AS double)")
    )
    return with_num.withColumn(
        "is_active",
        when(col("is_active").isNull(), lit(False))
        .when(col("is_active_num").isNotNull(), col("is_active_num") != 0.0)
        .when(
            lower(trim(col("is_active"))).isin("1", "true", "yes", "y", "t"),
            lit(True),
        )
        .otherwise(lit(False)),
    ).select("customer_id", "city", "is_active")


def clean_refunds(refunds_df: SparkDataFrame) -> SparkDataFrame:
    return (
        refunds_df.withColumn("amount", expr("try_cast(amount AS double)"))
        .na.fill({"amount": 0.0})
        .select("order_id", "amount")
    )


def extract_flat_orders(orders_df: SparkDataFrame) -> SparkDataFrame:
    exploded = orders_df.filter(col("payment_status") == "paid").withColumn(
        "item", explode("items")
    )
    flat = exploded.select(
        "order_id",
        "customer_id",
        "channel",
        "created_at",
        col("item.sku").alias("item_sku"),
        col("item.qty").alias("item_qty"),
        col("item.unit_price").alias("item_unit_price"),
    )
    return flat


def save_rejects_if_any(
    neg_df: SparkDataFrame, output_dir: Path, csv_encoding: str, csv_sep: str
) -> int:
    count = int(neg_df.count())
    if count:
        out_path = output_dir / "rejects_items.csv"
        output_dir.mkdir(parents=True, exist_ok=True)
        neg_df.toPandas().to_csv(
            out_path, index=False, encoding=csv_encoding, sep=csv_sep
        )
        print(f"{count} lignes rejetées -> {out_path}")
    return count


def keep_positive_items(flat_df: SparkDataFrame) -> SparkDataFrame:
    return flat_df.filter(col("item_unit_price") >= 0)


def deduplicate_orders(flat_df: SparkDataFrame) -> SparkDataFrame:
    window = Window.partitionBy("order_id").orderBy(col("created_at").asc())
    dedup = (
        flat_df.withColumn("rn", row_number().over(window))
        .filter(col("rn") == 1)
        .drop("rn")
    )
    return dedup


def compute_per_order(dedup_df: SparkDataFrame) -> SparkDataFrame:
    dedup_with_gross = dedup_df.withColumn(
        "line_gross", col("item_qty") * col("item_unit_price")
    )
    per_order = dedup_with_gross.groupBy(
        "order_id", "customer_id", "channel", "created_at"
    ).agg(
        F_sum("item_qty").alias("items_sold"),
        F_sum("line_gross").alias("gross_revenue_eur"),
    )
    return per_order


def join_customers_and_add_date(
    per_order_df: SparkDataFrame, customers_clean: SparkDataFrame
) -> SparkDataFrame:
    """
    Joint les clients et ajoute une colonne order_date (yyyy-MM-dd) en pur Spark,
    compatible avec les deux formats possibles de created_at :
    - 'YYYY-MM-DD HH:MM:SS'
    - 'YYYY-MM-DD'
    """
    joined = per_order_df.join(customers_clean, on="customer_id", how="left").filter(
        col("is_active") == True  # noqa: E712
    )
    parsed_date = coalesce(
        to_date(col("created_at"), "yyyy-MM-dd HH:mm:ss"),
        to_date(col("created_at"), "yyyy-MM-dd"),
    )
    return joined.withColumn("order_date", date_format(parsed_date, "yyyy-MM-dd"))


def merge_refunds(
    per_order_active: SparkDataFrame, refunds_clean: SparkDataFrame
) -> SparkDataFrame:
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
    return per_order_ref


def aggregate_daily_city(per_order_ref: SparkDataFrame) -> SparkDataFrame:
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
    return agg_df


# -------------------------
# Persistence
# -------------------------
def persist_to_sqlite(
    per_order_ref: SparkDataFrame, agg_df: SparkDataFrame, db_path: Path
) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        per_order_pd.to_sql("orders_clean", conn, if_exists="replace", index=False)
        agg_pd.to_sql("daily_city_sales", conn, if_exists="replace", index=False)
    return per_order_pd, agg_pd


def export_daily_csvs(
    agg_pd: pd.DataFrame,
    output_dir: Path,
    csv_sep: str,
    csv_encoding: str,
    csv_float_format: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
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


# -------------------------
# pipeline execution function
# -------------------------
def run_pyspark_pipeline(
    repo_root: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Exécute la pipeline PySpark et retourne :
    - per_order_pd : commandes nettoyées au niveau commande (Pandas)
    - agg_pd       : agrégat daily_city_sales (Pandas)
    """
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]

    cfg_path = repo_root / "settings.yaml"
    cfg = load_settings(cfg_path)

    input_dir = resolve_path(repo_root, cfg.get("input_dir", "./data/march-input"))
    output_dir = resolve_path(repo_root, cfg.get("output_dir", "./data/out"))
    csv_sep = cfg.get("csv_sep", ";")
    csv_encoding = cfg.get("csv_encoding", "utf-8")

    output_dir.mkdir(parents=True, exist_ok=True)

    orders_files = list_order_files(input_dir)
    if not orders_files:
        raise FileNotFoundError(f"Aucun fichier orders_*.json dans {input_dir}")

    customers_file = input_dir / "customers.csv"
    refunds_file = input_dir / "refunds.csv"
    if not customers_file.exists():
        raise FileNotFoundError(f"Fichier clients introuvable: {customers_file}")
    if not refunds_file.exists():
        raise FileNotFoundError(f"Fichier remboursements introuvable: {refunds_file}")

    spark = build_spark()

    try:
        # Lecture
        orders_df, refunds_df, customers_df = read_sources(
            spark, orders_files, customers_file, refunds_file
        )

        # Nettoyage / transfo
        customers_clean = clean_customers(customers_df)
        refunds_clean = clean_refunds(refunds_df)

        flat_orders = extract_flat_orders(orders_df)

        neg_df = flat_orders.filter(col("item_unit_price") < 0)
        save_rejects_if_any(neg_df, output_dir, csv_encoding, csv_sep)

        positive_orders = keep_positive_items(flat_orders)
        dedup = deduplicate_orders(positive_orders)
        per_order = compute_per_order(dedup)
        per_order_active = join_customers_and_add_date(per_order, customers_clean)
        per_order_ref = merge_refunds(per_order_active, refunds_clean)

        agg_df = aggregate_daily_city(per_order_ref)

        # Conversion en Pandas pour les tests
        per_order_pd = per_order_ref.select(
            "order_id",
            "customer_id",
            "city",
            "channel",
            "order_date",
            "items_sold",
            "gross_revenue_eur",
            "refunds_eur",
        ).toPandas()

        agg_pd = agg_df.orderBy("date", "city", "channel").toPandas()

        return per_order_pd, agg_pd
    finally:
        spark.stop()


# -------------------------
# Orchestration (main)
# -------------------------
def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    cfg_path = repo_root / "settings.yaml"
    cfg = load_settings(cfg_path)

    db_path = resolve_path(repo_root, cfg.get("db_path", "./data/sales_db.db"))
    output_dir = resolve_path(repo_root, cfg.get("output_dir", "./data/out"))
    csv_sep = cfg.get("csv_sep", ";")
    csv_encoding = cfg.get("csv_encoding", "utf-8")
    csv_float_format = cfg.get("csv_float_format", "%.2f")

    per_order_pd, agg_pd = run_pyspark_pipeline(repo_root)

    # Persistance SQLite
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        per_order_pd.to_sql("orders_clean", conn, if_exists="replace", index=False)
        agg_pd.to_sql("daily_city_sales", conn, if_exists="replace", index=False)

    # CSV journaliers
    export_daily_csvs(agg_pd, output_dir, csv_sep, csv_encoding, csv_float_format)


if __name__ == "__main__":
    main()
