"""ingest.py - Load the 12 raw Zomato CSVs.

All columns are read as strings first (dtype=str) so that malformed values
(e.g. phone "abc", mixed date formats) survive loading; typing/validation is
the job of clean.py.  Filenames are canonical (no "(1)" browser suffixes).
"""
from pathlib import Path
import logging
import pandas as pd

LOG = logging.getLogger("ingest")

TABLES = ["cities", "customers", "restaurants", "menu", "delivery_partners",
          "promotions", "orders", "order_items", "payments",
          "customer_feedback", "weather", "traffic"]

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
CLEAN_DIR = ROOT / "data" / "cleaned"


def load_table(name: str, directory: Path = RAW_DIR) -> pd.DataFrame:
    """Load one CSV as strings. Raises FileNotFoundError with a clear message."""
    path = Path(directory) / f"{name}.csv"
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=True, skipinitialspace=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing dataset: {path}") from exc
    LOG.info("loaded %-18s rows=%d cols=%d", name, *df.shape)
    return df


def load_raw(directory: Path = RAW_DIR) -> dict:
    """Load all 12 raw tables into a dict {table_name: DataFrame(str)}."""
    return {t: load_table(t, directory) for t in TABLES}


def load_cleaned(directory: Path = CLEAN_DIR) -> dict:
    """Load cleaned tables with proper dtypes (dates parsed)."""
    date_cols = {"customers": ["RegistrationDate"], "delivery_partners": ["JoiningDate"],
                 "promotions": ["StartDate", "EndDate"], "orders": ["OrderDate"],
                 "payments": ["PaymentDate"], "weather": ["Date"], "traffic": ["Date"]}
    out = {}
    for t in TABLES:
        out[t] = pd.read_csv(Path(directory) / f"{t}.csv", parse_dates=date_cols.get(t, []))
    return out
