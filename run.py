import pandas as pd
from functools import reduce
from functools import lru_cache
import numpy as np
import joblib
import datetime
from gocardless.config.config import CREDITOR_COLS, MANDATE_COLS, PAYMENT_COL

model = joblib.load("config/churn_model.pkl")


@lru_cache()
def get_data(key: str) -> pd.DataFrame:
    """
    Get the data and return pd.DataFrame
    :param key: table names
    :return: pd.DataFrame
    """
    query = f"""
        SELECT *
        from gc_data_science.{key}
    """
    df = pd.read_gbq(query, project_id="gc-prd-ext-data-test-prod-906c")
    return df


def merge_data(creditors: pd.DataFrame, mandates: pd.DataFrame, payments: pd.DataFrame) -> pd.DataFrame:
    """
    Joins three tables together
    :param creditors: pd.DataFrame with creditor
    :param mandates: pd.DataFrame with mandates
    :param payments: pd.DataFrame with payments
    :return: pd.DataFrame with the joined table
    """
    merged = (
        creditors.rename(
            columns={
                CREDITOR_COLS.id: CREDITOR_COLS.creditor_id,
                CREDITOR_COLS.created_at: CREDITOR_COLS.creditor_created_at,
            }
        )
        .merge(
            mandates.rename(
                columns={
                    MANDATE_COLS.id: MANDATE_COLS.mandate_id,
                    MANDATE_COLS.created_at: MANDATE_COLS.mandate_created_at,
                }
            ),
            on=CREDITOR_COLS.creditor_id,
        )
        .merge(
            payments.rename(
                columns={PAYMENT_COL.id: PAYMENT_COL.payment_id, PAYMENT_COL.created_at: PAYMENT_COL.payment_created_at}
            ),
            on=(CREDITOR_COLS.creditor_id, MANDATE_COLS.mandate_id),
        )
    )
    return merged


def process_creditor_data(creditor_payments: pd.DataFrame) -> pd.DataFrame:
    """
    Add payment statistic and mandate statistic to creditor data
    :param creditor_payments: pd.DataFrame: merged table data
    :return: pd.DataFrame with all selected features
    """
    # payment stats
    payments_stats = (
        creditor_payments[
            [
                CREDITOR_COLS.creditor_id,
                PAYMENT_COL.amount_gbp,
                PAYMENT_COL.payment_created_at,
                PAYMENT_COL.has_reference,
                PAYMENT_COL.has_description,
                PAYMENT_COL.source,
            ]
        ]
        .groupby(CREDITOR_COLS.creditor_id)
        .agg(
            amount_sum=(PAYMENT_COL.amount_gbp, np.sum),
            num_payments=(PAYMENT_COL.amount_gbp, len),
            active_aging=(PAYMENT_COL.payment_created_at, lambda x: np.ptp(x).days + 1),
            pct_has_ref=(PAYMENT_COL.has_reference, np.mean),
            pct_source_api=(PAYMENT_COL.source, lambda x: sum(1 for s in x if s == "api") / len(x),),
            pct_source_app=(PAYMENT_COL.source, lambda x: sum(1 for s in x if s == "app") / len(x),),
        )
        .reset_index()
    )

    # mandate stats
    creditors_mandates = creditor_payments[
        [
            CREDITOR_COLS.creditor_id,
            MANDATE_COLS.mandate_id,
            MANDATE_COLS.payments_require_approval,
            MANDATE_COLS.is_business_customer_type,
            MANDATE_COLS.scheme,
        ]
    ].drop_duplicates()
    mandates_stats = (
        creditors_mandates[
            [
                CREDITOR_COLS.creditor_id,
                MANDATE_COLS.mandate_id,
                MANDATE_COLS.payments_require_approval,
                MANDATE_COLS.is_business_customer_type,
                MANDATE_COLS.scheme,
            ]
        ]
        .groupby(CREDITOR_COLS.creditor_id)
        .agg(
            pct_payments_require_approval=(MANDATE_COLS.payments_require_approval, np.mean),
            pct_is_business_customer_type=(MANDATE_COLS.is_business_customer_type, np.mean),
            num_mandates=(MANDATE_COLS.mandate_id, len),
            pct_scheme_bacs=(MANDATE_COLS.scheme, lambda x: sum(1 for s in x if s == "bacs") / len(x),),
        )
        .reset_index()
    )

    dfs = [
        payments_stats,
        mandates_stats,
        creditor_payments[
            [
                CREDITOR_COLS.creditor_id,
                CREDITOR_COLS.has_logo,
                CREDITOR_COLS.merchant_type,
                CREDITOR_COLS.refunds_enabled,
            ]
        ].drop_duplicates(),
    ]
    data = reduce(lambda left, right: left.merge(right, how="left"), dfs)

    return data


def extract_data(merged: pd.DataFrame) -> pd.DataFrame:
    """
     Extract the test data:
     The creditor ids that are active since Q4 2016 will be the ones we want to predict for Q1 2017.
     We will use those creditor id's data records for the latest 6 months (from 2016-7-1 to 2016-12-31) for prediction.
    :param merged: pd.DataFrame with merged table
    :return: pd.DataFrame with the data that will be used for prediction
    """
    cond = merged[PAYMENT_COL.payment_created_at] >= datetime.datetime(2016, 10, 1, tzinfo=datetime.timezone.utc)
    churned_creditor_ids_pre_q4 = set(merged[CREDITOR_COLS.creditor_id]) - set(merged[cond][CREDITOR_COLS.creditor_id])

    start = datetime.datetime(2016, 7, 1, tzinfo=datetime.timezone.utc)
    df_6months = merged[(start <= merged[PAYMENT_COL.payment_created_at])]
    df_test = df_6months[~df_6months[CREDITOR_COLS.creditor_id].isin(churned_creditor_ids_pre_q4)]

    return df_test


def run_pipeline():
    """
    Step 1: Getting the data from three tables
    Step 2: Joining three tables to one table
    Step 3: Extract the data that will be used for prediction
    Step 4: Process the data by adding payment and mandate statistic
    Step 5: Run model to get the probability output
    """
    creditors = get_data(key="creditors")
    mandates = get_data(key="mandates")
    payments = get_data(key="payments")

    merged = merge_data(creditors, mandates, payments)

    df_test = extract_data(merged)

    X_test = process_creditor_data(df_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    result = dict(zip(X_test["creditor_id"], y_proba))
    res = pd.DataFrame(result.items(), columns=["id", "probability"])
    res.to_csv("prediction.csv")


if __name__ == "__main__":
    run_pipeline()
