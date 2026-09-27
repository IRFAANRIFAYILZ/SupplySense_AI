"""
helpers.py
----------
Small shared utilities: cached data loading, formatting, and basic input
validation used across the Streamlit app and other modules.
"""

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

ALLOWED_DATA_FILES = {"suppliers.csv", "orders.csv", "inventory.csv"}


def load_csv_safe(filename: str) -> pd.DataFrame:
    """Loads a CSV from the data/ directory, but ONLY if it's on the explicit
    allow-list. This prevents arbitrary file-path access if this function is
    ever wired up to user input (e.g. a future 'upload your own data' feature)."""
    if filename not in ALLOWED_DATA_FILES:
        raise ValueError(f"'{filename}' is not an allowed data file.")
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python data/generate_data.py` first."
        )
    return pd.read_csv(path)


def format_currency(value: float) -> str:
    return f"${value:,.0f}"


def format_percent(value: float, already_fraction: bool = True) -> str:
    pct = value * 100 if already_fraction else value
    return f"{pct:.1f}%"


def inventory_value(inventory_df: pd.DataFrame) -> float:
    return float((inventory_df["current_stock"] * inventory_df["unit_cost"]).sum())


def validate_uploaded_file(filename: str, max_size_bytes: int = 5_000_000) -> None:
    """Basic file-type / size validation for any future file-upload feature
    (e.g. uploading a new policy document for RAG ingestion)."""
    allowed_extensions = {".csv", ".md", ".txt", ".pdf"}
    ext = Path(filename).suffix.lower()
    if ext not in allowed_extensions:
        raise ValueError(f"File type '{ext}' is not permitted. Allowed: {allowed_extensions}")
