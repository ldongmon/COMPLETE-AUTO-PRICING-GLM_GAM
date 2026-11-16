"""
load_data.py
============
Step 1 of the project: get the two source tables and clean them.

The data is the French Motor Third-Party Liability dataset, published on
OpenML (a public data website):

    freMTPL2freq  one row per POLICY  (exposure, claim count, rating factors)
    freMTPL2sev   one row per PAYMENT (the amount paid for one claim)

The first call downloads both tables and saves them in data/raw/ as parquet
files (a fast, compact file format). Every later call simply reads those
files, so the internet is used only once.

This module also contains the two functions that detect and remove the
"claims-only block": the first 9,387 policies of the file, which all carry a
claim and are not insured policies like the others (see Phase 1).

Usage in a notebook:

    from src.data.load_data import load_freq_raw, load_sev_raw
    freq = load_freq_raw()      # 678,013 policies
    sev = load_sev_raw()        # 26,639 claim payments
"""

from pathlib import Path  # Path makes file paths work the same way on Windows, Mac and Linux

import pandas as pd  # pandas is the standard Python library for tables (DataFrames)
from sklearn.datasets import fetch_openml  # downloads a dataset from OpenML by its id


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #

OPENML_ID_FREQ = 41214  # OpenML id of the policy table (ids never change; names can)
OPENML_ID_SEV = 41215  # OpenML id of the payment table

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # this file is in src/data/, so the root is two levels up
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"  # folder where the downloaded tables are saved
FREQ_CACHE_PATH = RAW_DATA_DIR / "freMTPL2freq.parquet"  # saved copy of the policy table
SEV_CACHE_PATH = RAW_DATA_DIR / "freMTPL2sev.parquet"  # saved copy of the payment table

# The columns we expect in each table, and the type we want for each one.
# "int64" = whole number, "float64" = decimal number, "category" = a label (e.g. a region code).
FREQ_DTYPES = {
    "IDpol": "int64",  # policy number
    "ClaimNb": "int64",  # number of claims reported during the year
    "Exposure": "float64",  # time on risk, in years (0.5 = six months)
    "Area": "category",  # area code A to F (a coarse version of Density)
    "VehPower": "int64",  # fiscal power of the vehicle
    "VehAge": "int64",  # age of the vehicle, in years
    "DrivAge": "int64",  # age of the driver, in years
    "BonusMalus": "int64",  # French bonus-malus coefficient (50 = best, 100 = neutral)
    "VehBrand": "category",  # vehicle brand (anonymised: B1, B2, ...)
    "VehGas": "category",  # fuel: Diesel or Regular
    "Density": "float64",  # inhabitants per km2 in the policyholder's town
    "Region": "category",  # French region code
}
SEV_DTYPES = {
    "IDpol": "int64",  # policy number, the link to the policy table
    "ClaimAmount": "float64",  # amount paid, in euros
}


# --------------------------------------------------------------------------- #
# Download and cleaning (used only the first time)
# --------------------------------------------------------------------------- #


def clean_text_column(column):
    """OpenML sometimes returns labels with quotes around them ("'B12'"); remove them."""
    text = column.astype(str)  # make sure every value is a piece of text
    text = text.str.strip("'")  # remove a quote at the start or the end of each value
    return text.astype("category")  # store the result as a category (a label)


def download_table(openml_id, expected_types):
    """Download one table from OpenML, keep the expected columns and set their types."""
    downloaded = fetch_openml(data_id=openml_id, as_frame=True, parser="liac-arff")  # call the OpenML website
    table = downloaded.frame.copy()  # the table itself is in the .frame attribute

    missing_columns = [name for name in expected_types if name not in table.columns]  # columns we expected but did not get
    if len(missing_columns) > 0:  # stop with a clear message if the source has changed
        raise ValueError(f"OpenML table {openml_id} is missing the columns {missing_columns}")

    table = table[list(expected_types)]  # keep only the expected columns, in the expected order
    for name, wanted_type in expected_types.items():  # go through the columns one by one
        if wanted_type == "category":  # text labels need their quotes removed first
            table[name] = clean_text_column(table[name])
    return table.astype(expected_types)  # convert every column to its expected type


def load_table(cache_path, openml_id, expected_types, force_download=False):
    """Read the saved copy of a table, or download it the first time."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)  # create data/raw/ if it does not exist yet
    if cache_path.exists() and not force_download:  # a saved copy is already there
        return pd.read_parquet(cache_path)  # read it: no internet needed
    table = download_table(openml_id, expected_types)  # first time: download
    table.to_parquet(cache_path, index=False)  # and save it for next time
    return table


def load_freq_raw(force_download=False):
    """The policy table, one row per policy (678,013 rows)."""
    return load_table(FREQ_CACHE_PATH, OPENML_ID_FREQ, FREQ_DTYPES, force_download)


def load_sev_raw(force_download=False):
    """The payment table, one row per claim payment (26,639 rows)."""
    return load_table(SEV_CACHE_PATH, OPENML_ID_SEV, SEV_DTYPES, force_download)


# --------------------------------------------------------------------------- #
# The claims-only block (data defect found in Phase 1)
# --------------------------------------------------------------------------- #
# In a normal portfolio, most policies have NO claim in the year (here 96%).
# Yet every policy at the start of the file, up to IDpol 24,951, has a claim,
# and only 3% of those claims were ever paid. This block is a piece of a
# claims file pasted into the policy file, not a group of insured drivers.
# We find where it ends from the data itself: it stops just before the first
# policy without a claim.


def find_claims_only_block_end(freq):
    """Return the last IDpol of the leading block in which every policy has a claim."""
    ordered = freq.sort_values("IDpol")  # sort the policies by number
    claim_free = ordered[ordered["ClaimNb"] == 0]  # the policies with no claim
    first_claim_free_id = claim_free["IDpol"].min()  # the first of them (24,952 in this dataset)
    block = ordered[ordered["IDpol"] < first_claim_free_id]  # every policy before it
    if len(block) == 0:  # no block at all: nothing to remove
        return 0
    return int(block["IDpol"].max())  # the last policy number of the block (24,951)


def drop_claims_only_block(df, block_end):
    """Remove every policy with IDpol up to `block_end`."""
    kept = df[df["IDpol"] > block_end]  # keep only the policies after the block
    return kept.reset_index(drop=True)  # renumber the rows 0, 1, 2, ...


# --------------------------------------------------------------------------- #
# Command line: `python -m src.data.load_data` downloads the data and prints a summary
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    freq = load_freq_raw()  # download or read the policy table
    sev = load_sev_raw()  # download or read the payment table
    print(f"freMTPL2freq: {len(freq):,} policies, {freq['IDpol'].nunique():,} unique policy numbers")
    print(f"freMTPL2sev : {len(sev):,} payments, on {sev['IDpol'].nunique():,} policies")
