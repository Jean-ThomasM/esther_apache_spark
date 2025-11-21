import pandas as pd
from pandas.testing import assert_frame_equal

from pandas_files.pipeline_pandas import run_pandas_pipeline
from pyspark_files.pipeline_pyspark import run_pyspark_pipeline


def _sorted_by_key(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["date", "city", "channel"]).reset_index(drop=True)


def _sorted_per_by_key(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(
        ["order_id", "customer_id", "channel", "order_date"]
    ).reset_index(drop=True)


def test_pytest_is_configured():
    assert 1 + 1 == 2


def test_pipelines_run_and_return_dataframes():
    # 1. On exécute la pipeline Pandas
    per_pd, agg_pd = run_pandas_pipeline()
    # 2. On exécute la pipeline PySpark
    per_sp, agg_sp = run_pyspark_pipeline()

    # 3. Vérifications très simples : ce sont bien des DataFrames pandas
    assert isinstance(per_pd, pd.DataFrame)
    assert isinstance(agg_pd, pd.DataFrame)
    assert isinstance(per_sp, pd.DataFrame)
    assert isinstance(agg_sp, pd.DataFrame)

    # 4. Et elles ne sont pas vides (au moins 1 ligne)
    assert len(agg_pd) > 0
    assert len(agg_sp) > 0


def test_pipeline_equivalences():
    """
    Vérifie que les agrégats quotidiens Pandas et PySpark
    sont équivalents sur les colonnes métiers principales.
    """
    _, agg_pd = run_pandas_pipeline()
    _, agg_sp = run_pyspark_pipeline()

    cols = [
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

    agg_pd_filtered = agg_pd[cols].copy()
    agg_sp_filtered = agg_sp[cols].copy()

    agg_pd_sorted = _sorted_by_key(agg_pd_filtered)
    agg_sp_sorted = _sorted_by_key(agg_sp_filtered)

    assert_frame_equal(agg_pd_sorted, agg_sp_sorted, check_dtype=False, atol=1e-6)


def test_per_order_equivalence():
    """
    Vérifie que les DataFrames niveau commande (per_order)
    sont équivalents entre Pandas et PySpark.
    """
    per_pd, _ = run_pandas_pipeline()
    per_sp, _ = run_pyspark_pipeline()

    cols = [
        "order_id",
        "customer_id",
        "city",
        "channel",
        "order_date",
        "items_sold",
        "gross_revenue_eur",
        "refunds_eur",
    ]

    per_pd_filtered = per_pd[cols].copy()
    per_sp_filtered = per_sp[cols].copy()

    per_pd_sorted = _sorted_per_by_key(per_pd_filtered)
    per_sp_sorted = _sorted_per_by_key(per_sp_filtered)

    assert_frame_equal(per_pd_sorted, per_sp_sorted, check_dtype=False, atol=1e-6)


def test_aggregate_sanity_checks():
    """
    Sanity checks métier sur les agrégats des deux pipelines.
    """
    _, agg_pd = run_pandas_pipeline()
    _, agg_sp = run_pyspark_pipeline()

    for agg in (agg_pd, agg_sp):
        # net_revenue_eur ≈ gross_revenue_eur + refunds_eur
        diff = (
            agg["net_revenue_eur"] - agg["gross_revenue_eur"] - agg["refunds_eur"]
        ).abs()
        assert (diff < 1e-6).all()

        # bornes minimales cohérentes
        assert (agg["orders_count"] >= agg["unique_customers"]).all()
        assert (agg["items_sold"] >= 0).all()
        assert (agg["gross_revenue_eur"] >= 0).all()
