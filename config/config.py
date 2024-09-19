from typing import NamedTuple


class CreditorColumnNames(NamedTuple):
    id: str
    creditor_id: str
    created_at: str
    creditor_created_at: str
    has_logo: str
    merchant_type: str
    refunds_enabled: str


class MandateColumnNames(NamedTuple):
    id: str
    mandate_id: str
    created_at: str
    mandate_created_at: str
    payments_require_approval: str
    is_business_customer_type: str
    scheme: str


class PaymentsColumnNames(NamedTuple):
    id: str
    payment_id: str
    created_at: str
    payment_created_at: str
    amount_gbp: str
    has_reference: str
    has_description: str
    source: str


CREDITOR_COLS = CreditorColumnNames(
    id="id",
    creditor_id="creditor_id",
    created_at="created_at",
    creditor_created_at="creditor_created_at",
    has_logo="has_logo",
    merchant_type="merchant_type",
    refunds_enabled="refunds_enabled",
)

MANDATE_COLS = MandateColumnNames(
    id="id",
    mandate_id="mandate_id",
    created_at="mandate_id",
    mandate_created_at="mandate_created_at",
    payments_require_approval="payments_require_approval",
    is_business_customer_type="is_business_customer_type",
    scheme="scheme",
)

PAYMENT_COL = PaymentsColumnNames(
    id="id",
    payment_id="payment_id",
    created_at="created_at",
    payment_created_at="payment_created_at",
    amount_gbp="amount_gbp",
    has_reference="has_reference",
    has_description="has_description",
    source="source",
)
