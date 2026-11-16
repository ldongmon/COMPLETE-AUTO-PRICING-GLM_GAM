"""
commercial_premium.py
=====================
Phase 11: from the technical premium (expected claims) to the commercial
premium (what the customer pays), by adding the expenses.

THE LOADING
-----------
We want, on average, claims + expenses = premium. With a commercial premium
P = technical premium x (1 + loading):

    expected claims = technical premium        (expected LR on the technical premium = 100%)
    expenses        = 27% x P
    claims + expenses = P   =>   technical premium = (1 - 0.27) x P
                            =>   1 + loading = 1 / 0.73   =>   loading = 37.0%

General formula: loading = expected LR / (1 - expense ratio) - 1.

Example: technical premium EUR 150 -> commercial premium 150 x 1.37 = EUR 205.48;
expenses 27% x 205.48 = 55.48; 205.48 - 55.48 = 150 left for the claims.

The book's realized LR (112.7%, Phase 8) is NOT used: it is pushed up by a few
giant claims in one data set, and would charge that bad luck to everybody.
"""


def calibrate_breakeven_loading(loss_ratio, expense_ratio):
    """loading = LR / (1 - expense ratio) - 1  (37.0% for LR 100% and expenses 27%)."""
    return loss_ratio / (1 - expense_ratio) - 1


def add_commercial_premium(df, loading):
    """Adds commercial_premium = technical_premium x (1 + loading) (a copy, df is not changed)."""
    out = df.copy()
    out["commercial_premium"] = out["technical_premium"] * (1 + loading)
    return out
