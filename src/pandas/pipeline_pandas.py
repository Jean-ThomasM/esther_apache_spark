import os, glob, json, sqlite3
import pandas as pd
import yaml
from datetime import datetime
from pathlib import Path
from IPython.display import display, Markdown

def load_settings(path="settings.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    
cfg = load_settings()
in_dir = cfg.get("input_dir", "./data/march-input")
out_dir = cfg.get("output_dir", "./data/out")
db_path = cfg.get("db_path", "./data/sales_db.db")
sep = cfg.get("csv_sep",";")
enc = cfg.get("csv_encoding","utf-8")
ffmt = cfg.get("csv_float_format","%.2f")
Path(out_dir).mkdir(parents=True, exist_ok=True)
# display(Markdown(
#     f"Input dir: `{in_dir}`\n"
#     f"Output dir: `{out_dir}`\n"
#     f"DB (SQLite) : `{db_path}`"
# ))

customers_path = os.path.join(in_dir, "customers.csv")
#customers_path = ("/chemin vers /customers.csv")
if not os.path.exists(customers_path):
    print(f" Fichier manquant : `{customers_path}`.")
else:
    customers = pd.read_csv(customers_path)
    print(Markdown(f"Taille: {customers.shape}"))

refunds_path = os.path.join(in_dir, "refunds.csv")
#refunds_path = ("/chemin vers /refunds.csv")
if not os.path.exists(refunds_path):
    # display(Markdown(f" Fichier manquant : `{refunds_path}`."))
    print(f" Fichier manquant : `{refunds_path}`.")
else:
    refunds = pd.read_csv(refunds_path)
    # display(refunds.head())
    # display(Markdown(f"Taille: {refunds.shape}"))
    print(f"Taille: {refunds.shape}")

order_path = os.path.join(in_dir, "orders_2025-03-01.json")
#order_path = ("/Users/estherchabi/Documents/fr/Caplogy/2025-2026/Simplon/Inter UI ARA HDF /Esther_Phase0/Brief1111/projet_python/data/march-input/orders_2025-03-01.json")

if not os.path.exists(order_path):
    # display(Markdown(f" Fichier manquant : `{order_path}`."))
    print(f" Fichier manquant : `{order_path}`.")
else:
    order = pd.read_json(order_path)
    # display(order.head())
    # display(Markdown(f"Taille: {order.shape}"))
    print(f"Taille: {order.shape}")

liste = []

for day in range(1, 32): 
    order_path = os.path.join(in_dir, f"orders_2025-03-{day:02d}.json")

    if not os.path.exists(order_path):
        # display(Markdown(f" Fichier manquant : `{order_path}`."))
        print(f" Fichier manquant : `{order_path}`.")
        continue
    else:
        order = pd.read_json(order_path)
        #display(Markdown(f"Taille: {order.shape}")) # Taille: (103, 6)

    liste.append(order)
    #display(Markdown(f"Taille : {liste}"))

orders = pd.concat(liste) 

# display(orders.head())
# display(Markdown(f"Taille: {orders.shape}")) # Taille: (103x31, 6) = (3193, 6)

def controle_bool(v):
    if isinstance(v, bool): return v
    if isinstance(v, (int, float)): return bool(v)
    if v is None: return False
    s = str(v).strip().lower()
    return s in ("1","true","yes","y","t")

customers["is_active"] = customers["is_active"].apply(controle_bool)
customers = customers.astype({"customer_id":"string","city":"string"})
# display(Markdown("Affichage clients (après nettoyage)"))
# display(customers.head(30))
# display(Markdown(f"Taille: {customers.shape}"))

#refunds = pd.read_csv(refunds_path)
refunds["amount"] = pd.to_numeric(refunds["amount"], errors="coerce").fillna(0.0)
refunds["created_at"] = refunds["created_at"].astype("string")
# display(Markdown("Aperçu remboursements (après coercition numérique)"))
# display(refunds.head())
# display(Markdown(f"Taille: {refunds.shape}"))

#Filtrer les commandes payées (payment_status == 'paid')
# On peut aussi ecrire une fonction qui verifie que le status est bien 'paid'
# ex: p,ok,yes => paid
# def controle_paid(v):
#     v = v.lower()
#     if v in ['p', 'ok', 'yes']:
#         return 'paid'
#     return 'other'   
# orders["payment_status"] = orders["payment_status"].apply(controle_paid) 
ln_initial = len(orders)
orders = orders[orders["payment_status"]=="paid"].copy()
ln_final = len(orders)
# display(Markdown(f"Filtrage payées : {ln_initial} → {ln_final}"))
# display(orders.head())

orders2 = orders
# display(Markdown("Avant explosion des items"))
# display(orders2.head())
#display(Markdown(f"Colonnes: {list(orders2.columns)[:12]}

orders2 = orders2.explode("items", ignore_index=True)
# display(orders.head())
items = pd.json_normalize(orders2["items"]).add_prefix("item_")
# display(items)
orders2 = pd.concat([orders2.drop(columns=["items"]), items], axis=1)
# display(Markdown("Après explosion des items"))
# display(orders2.head())
# display(Markdown(f"Colonnes: {list(orders2.columns)[:12]} ..."))

neg_mask = orders2["item_unit_price"] < 0
n_neg = int(neg_mask.sum())
# display(Markdown(f"Lignes prix négatifs : {n_neg}"))
if n_neg > 0:
    rejects_items = orders2.loc[neg_mask].copy()
    rejects_path = os.path.join(out_dir, "rejects_items.csv")
    rejects_items.to_csv(rejects_path, index=False, encoding=enc)
    # display(Markdown(f" Rejets sauvegardés : `{rejects_path}`"))
    print(f" Rejets sauvegardés : `{rejects_path}`")
    orders2 = orders2.loc[~neg_mask].copy()
# display(orders2.head())

before = len(orders2)
orders3 = orders2.sort_values(["order_id","created_at"]).drop_duplicates(subset=["order_id"], keep="first")
after = len(orders3)
# display(Markdown(f"Déduplication : **{before} → {after}**"))
# display(orders3.head())

orders3["line_gross"] = orders3["item_qty"] * orders3["item_unit_price"]
per_order = orders3.groupby(["order_id","customer_id","channel","created_at"], as_index=False).agg(
    items_sold=("item_qty","sum"),
    gross_revenue_eur=("line_gross","sum")
)
# display(Markdown("Aperçu `per_order`"))
# display(per_order.head())
# display(Markdown(f"Taille: {per_order.shape}"))

len_init = len(per_order)
per_order = per_order.merge(customers[["customer_id","city","is_active"]], on="customer_id", how="left")
per_order = per_order[per_order["is_active"]==True].copy() # Important pour respecter le cahier de charge
ln_aft = len(per_order)
# display(Markdown(f"Après jointure+filtre actifs : **{len_init} → {ln_aft}**"))
# display(per_order.head())

def to_date(s):
    s = str(s)
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue
    raise ValueError(f"Format de date non reconnu: {s}")

per_order["order_date"] = per_order["created_at"].apply(to_date)
# display(per_order[["order_id","created_at","order_date"]].head())

refunds_sum = refunds.groupby("order_id", as_index=False)["amount"].sum().rename(columns={"amount":"refunds_eur"}) # Somme des remboursements par order_id (commande)
per_order = per_order.merge(refunds_sum, on="order_id", how="left").fillna({"refunds_eur":0.0})
# display(per_order.head())
conn = sqlite3.connect(db_path)
per_order_save = per_order[["order_id","customer_id","city","channel","order_date","items_sold","gross_revenue_eur"]].copy()
per_order_save.to_sql("orders_clean", conn, if_exists="replace", index=False)
# display(Markdown("Table `orders_clean` sauvegardée dans SQLite"))
conn.close()

agg = per_order.groupby(["order_date","city","channel"], as_index=False).agg(
    orders_count=("order_id","nunique"),
    unique_customers=("customer_id","nunique"),
    items_sold=("items_sold","sum"),
    gross_revenue_eur=("gross_revenue_eur","sum"),
    refunds_eur=("refunds_eur","sum")
)
agg["net_revenue_eur"] = agg["gross_revenue_eur"] + agg["refunds_eur"]
agg = agg.rename(columns={"order_date":"date"}).sort_values(["date","city","channel"]).reset_index(drop=True)
# display(agg.head())
# display(Markdown(f"Taille: {agg.shape}"))

conn = sqlite3.connect(db_path)
agg.to_sql("daily_city_sales", conn, if_exists="replace", index=False)
conn.close()
# display(Markdown("Table `daily_city_sales` écrite dans SQLite"))
# Exports CSV
for d, sub in agg.groupby("date"):
    out_path = os.path.join(out_dir, f"daily_summary_{d.replace('-','')}.csv")
    sub[[
        "date","city","channel","orders_count","unique_customers","items_sold",
        "gross_revenue_eur","refunds_eur","net_revenue_eur"
    ]].to_csv(out_path, index=False, sep=sep, encoding=enc, float_format=ffmt)
all_path = os.path.join(out_dir, "daily_summary_all.csv")
agg.to_csv(all_path, index=False, sep=sep, encoding=enc, float_format=ffmt)
# display(Markdown(f"Exports CSV dans `{out_dir}`"))
print(f"Exports CSV dans `{out_dir}`")