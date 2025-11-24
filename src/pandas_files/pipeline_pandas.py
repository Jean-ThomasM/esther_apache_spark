import os
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml


# Configuration / defaults
def load_settings(path: str = "settings.yaml") -> dict:
    """
    Charge le fichier de configuration en étant robuste au répertoire courant.
    Le chemin est résolu à partir de la racine du repo (deux niveaux au-dessus
    de ce fichier) si `path` est relatif.
    """
    repo_root = Path(__file__).resolve().parents[2]
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = repo_root / cfg_path
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


##### ---- variables globales ---- #####
REPO_ROOT = Path(__file__).resolve().parents[2]
cfg = load_settings()

IN_DIR = REPO_ROOT / cfg.get("input_dir", "./data/march-input")
OUT_DIR = REPO_ROOT / cfg.get("output_dir", "./data/out")
DB_PATH = REPO_ROOT / cfg.get("db_path", "./data/sales_db.db")

CSV_SEP = cfg.get("csv_sep", ";")
CSV_ENCODING = cfg.get("csv_encoding", "utf-8")
CSV_FLOAT_FMT = cfg.get("csv_float_format", "%.2f")

OUT_DIR.mkdir(parents=True, exist_ok=True)
##### ---------------------------- #####


# IO helpers
def read_csv_if_exists(path, **kwargs):
    if not os.path.exists(path):
        print(f"Fichier manquant : `{path}`.")
        return None
    return pd.read_csv(path, **kwargs)


def read_json_if_exists(path, **kwargs):
    if not os.path.exists(path):
        print(f"Fichier manquant : `{path}`.")
        return None
    return pd.read_json(path, **kwargs)


# Loaders
def load_customers(in_dir=IN_DIR):
    path = os.path.join(in_dir, "customers.csv")
    df = read_csv_if_exists(path)
    return df


def load_refunds(in_dir=IN_DIR):
    path = os.path.join(in_dir, "refunds.csv")
    df = read_csv_if_exists(path)
    return df


def load_orders_all(in_dir=IN_DIR, year=2025, month=3):
    frames = []
    for day in range(1, 32):
        fn = f"orders_{year}-{month:02d}-{day:02d}.json"
        path = os.path.join(in_dir, fn)
        df = read_json_if_exists(path)
        if df is None:
            continue
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# Cleaning / conversions
def controle_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if v is None:
        return False
    s = str(v).strip().lower()
    return s in ("1", "true", "yes", "y", "t")


def clean_customers(customers):
    if customers is None or customers.empty:
        return pd.DataFrame()
    customers = customers.copy()
    customers["is_active"] = customers["is_active"].apply(controle_bool)
    customers = customers.astype({"customer_id": "string", "city": "string"})
    return customers


def clean_refunds(refunds):
    if refunds is None or refunds.empty:
        return pd.DataFrame(columns=["order_id", "amount", "created_at"])
    refunds = refunds.copy()
    refunds["amount"] = pd.to_numeric(refunds["amount"], errors="coerce").fillna(0.0)
    refunds["created_at"] = refunds["created_at"].astype("string")
    return refunds


# Orders processing
def filter_paid_orders(orders):
    if orders is None or orders.empty:
        return pd.DataFrame()
    return orders[orders["payment_status"] == "paid"].copy()


def explode_orders_items(orders):
    if orders is None or orders.empty:
        return pd.DataFrame()
    df = orders.explode("items", ignore_index=True)
    items = pd.json_normalize(df["items"]).add_prefix("item_")
    df = pd.concat([df.drop(columns=["items"]), items], axis=1)
    return df


def remove_negative_items(orders_df, out_dir=OUT_DIR, encoding=CSV_ENCODING):
    if orders_df is None or orders_df.empty:
        return orders_df
    neg_mask = orders_df["item_unit_price"] < 0
    n_neg = int(neg_mask.sum())
    if n_neg > 0:
        rejects = orders_df.loc[neg_mask].copy()
        rejects_path = os.path.join(out_dir, "rejects_items.csv")
        rejects.to_csv(rejects_path, index=False, encoding=encoding)
        print(f"Rejets sauvegardés : `{rejects_path}`")
    return orders_df.loc[~neg_mask].copy()


def deduplicate_orders_keep_first(orders_df):
    if orders_df is None or orders_df.empty:
        return orders_df
    before = len(orders_df)
    df = orders_df.sort_values(["order_id", "created_at"]).drop_duplicates(
        subset=["order_id"], keep="first"
    )
    after = len(df)
    print(f"Déduplication : {before} → {after}")
    return df


def compute_line_and_per_order(orders_df):
    if orders_df is None or orders_df.empty:
        return pd.DataFrame()
    orders_df = orders_df.copy()
    orders_df["line_gross"] = orders_df["item_qty"] * orders_df["item_unit_price"]
    per_order = orders_df.groupby(
        ["order_id", "customer_id", "channel", "created_at"], as_index=False
    ).agg(
        items_sold=("item_qty", "sum"),
        gross_revenue_eur=("line_gross", "sum"),
    )
    return per_order


def join_customers_filter_active(per_order, customers):
    if per_order is None or per_order.empty:
        return per_order
    df = per_order.merge(
        customers[["customer_id", "city", "is_active"]], on="customer_id", how="left"
    )
    df = df[df["is_active"]].copy()
    return df


def to_date_str(s):
    s = str(s)
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue
    raise ValueError(f"Format de date non reconnu: {s}")


def add_order_date(per_order):
    if per_order is None or per_order.empty:
        return per_order
    per_order = per_order.copy()
    per_order["order_date"] = per_order["created_at"].apply(to_date_str)
    return per_order


def merge_refunds_sum(per_order, refunds):
    if per_order is None or per_order.empty:
        return per_order
    if refunds is None or refunds.empty:
        per_order = per_order.copy()
        per_order["refunds_eur"] = 0.0
        return per_order
    refunds_sum = (
        refunds.groupby("order_id", as_index=False)["amount"]
        .sum()
        .rename(columns={"amount": "refunds_eur"})
    )
    return per_order.merge(refunds_sum, on="order_id", how="left").fillna(
        {"refunds_eur": 0.0}
    )


def save_orders_to_sqlite(per_order, db_path=DB_PATH):
    if per_order is None or per_order.empty:
        print("Aucune commande à sauvegarder dans SQLite.")
        return
    conn = sqlite3.connect(db_path)
    per_order_save = per_order[
        [
            "order_id",
            "customer_id",
            "city",
            "channel",
            "order_date",
            "items_sold",
            "gross_revenue_eur",
        ]
    ].copy()
    per_order_save.to_sql("orders_clean", conn, if_exists="replace", index=False)
    conn.close()
    print("Table `orders_clean` sauvegardée dans SQLite")


def aggregate_daily_city(per_order):
    if per_order is None or per_order.empty:
        return pd.DataFrame()
    agg = per_order.groupby(["order_date", "city", "channel"], as_index=False).agg(
        orders_count=("order_id", "nunique"),
        unique_customers=("customer_id", "nunique"),
        items_sold=("items_sold", "sum"),
        gross_revenue_eur=("gross_revenue_eur", "sum"),
        refunds_eur=("refunds_eur", "sum"),
    )
    agg["net_revenue_eur"] = agg["gross_revenue_eur"] + agg["refunds_eur"]
    agg = (
        agg.rename(columns={"order_date": "date"})
        .sort_values(["date", "city", "channel"])
        .reset_index(drop=True)
    )
    return agg


def save_aggregates_csv(
    agg, out_dir=OUT_DIR, sep=CSV_SEP, encoding=CSV_ENCODING, float_fmt=CSV_FLOAT_FMT
):
    if agg is None or agg.empty:
        return
    for d, sub in agg.groupby("date"):
        out_path = os.path.join(out_dir, f"daily_summary_{d.replace('-', '')}.csv")
        sub[
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
            out_path, index=False, sep=sep, encoding=encoding, float_format=float_fmt
        )
    all_path = os.path.join(out_dir, "daily_summary_all.csv")
    agg.to_csv(
        all_path, index=False, sep=sep, encoding=encoding, float_format=float_fmt
    )
    print(f"Exports CSV dans `{out_dir}`")


def run_pandas_pipeline(in_dir=IN_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Exécute toute la pipeline pandas et retourne :
    - per_order : commandes nettoyées au niveau commande
    - agg       : agrégat daily_city_sales
    """
    customers = clean_customers(load_customers(in_dir))
    refunds = clean_refunds(load_refunds(in_dir))
    orders_raw = load_orders_all(in_dir)

    paid = filter_paid_orders(orders_raw)
    exploded = explode_orders_items(paid)
    no_neg = remove_negative_items(exploded)
    dedup = deduplicate_orders_keep_first(no_neg)
    per_order = compute_line_and_per_order(dedup)
    per_order = join_customers_filter_active(per_order, customers)
    per_order = add_order_date(per_order)
    per_order = merge_refunds_sum(per_order, refunds)

    agg = aggregate_daily_city(per_order)
    return per_order, agg


# Orchestration
def main():
    per_order, agg = run_pandas_pipeline(IN_DIR)

    save_orders_to_sqlite(per_order, DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    agg.to_sql("daily_city_sales", conn, if_exists="replace", index=False)
    conn.close()

    save_aggregates_csv(agg, OUT_DIR)


if __name__ == "__main__":
    main()
