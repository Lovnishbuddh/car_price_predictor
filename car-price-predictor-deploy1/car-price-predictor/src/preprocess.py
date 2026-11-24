"""
preprocess.py
=============
Data loading, cleaning and feature engineering for the Car Price Predictor.

This module is shared by eda.py and train_model.py so the data is prepared
in EXACTLY the same way during exploration, training, and (at prediction
time) inside the Flask app.

Why a separate file?
    - "Write once, use everywhere" avoids subtle bugs where the training
      data is cleaned one way and the live prediction data another.
    - It makes the project structure professional and easy to explain.

DATASET NOTE
    We use the "Quikr Cars" dataset (quikr_car.csv), a web-scraped
    collection of used-car listings from quikr.com. It is intentionally
    messy (real scraped data always is), which makes data cleaning the
    most important - and most demonstrable - part of this project.

Author : <Your Name>  |  Internship Mini Project
"""

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration constants (single source of truth)
# ---------------------------------------------------------------------------
DATASET_PATH = "dataset/quikr_car.csv"

# Features the model uses (after cleaning and feature engineering)
NUMERIC_FEATURES = ["Car_Age", "kms_driven"]
CATEGORICAL_FEATURES = ["name", "company", "fuel_type"]

TARGET = "Price"  # resale price in Rupees (e.g. 425000 = Rs 4,25,000)

# Sanity ranges used to remove junk / impossible values
YEAR_MIN, YEAR_MAX = 1995, 2020          # valid purchase years
PRICE_MIN, PRICE_MAX = 30_000, 30_00_000  # Rs 30,000 to Rs 30 lakh
KMS_MAX = 250_000                        # above this = data error

# A company needs at least this many listings to be kept; single-listing
# "companies" in this dataset are usually seller notes, not real brands.
MIN_COMPANY_LISTINGS = 5


def load_raw_data(path: str = DATASET_PATH) -> pd.DataFrame:
    """Step 1: Load the raw CSV into a pandas DataFrame."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 2: Data cleaning - the heart of this project.

    The raw Quikr data is scraped from web listings and contains many
    errors. Each rule below fixes one specific problem we actually
    observed in the data (you can show these rows in your presentation):

      a) 'year'          : some rows contain text like 'r 15', 'TOUR',
                           '150k' instead of a year  -> keep only 4-digit
                           years in a sensible range.
      b) 'Price'         : 35 rows say 'Ask For Price'; prices are stored
                           as strings like '4,25,000'; one row has an
                           impossible 85,00,003  -> keep only real numeric
                           prices within a sensible range.
      c) 'kms_driven'    : stored as '45,000 kms' strings; some blank  ->
                           strip text, convert to numbers, drop rows we
                           cannot parse; drop impossible readings.
      d) 'fuel_type'     : 55 rows are blank -> drop them.
      e) 'name'          : full listing titles vary too much
                           ('Hyundai i20 Sportz 1.2 Kappa') -> keep the
                           first 3 words so the same model maps to the
                           same value.
      f) 'company'       : inconsistent case ('MARUTI', 'tata', 'Maruti')
                           and junk sellers ('URJENT', 'selling car...')
                           -> normalise case and keep only companies with
                           enough listings to learn from.
    """
    df = df.copy()

    # (a) year: keep only numeric 4-digit years in range
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df[df["year"].between(YEAR_MIN, YEAR_MAX)]

    # (b) Price: drop 'Ask For Price', strip commas, keep sane range
    df = df[df["Price"].astype(str) != "Ask For Price"]
    df["Price"] = (df["Price"].astype(str)
                   .str.replace(",", "", regex=False))
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df = df[df["Price"].between(PRICE_MIN, PRICE_MAX)]

    # (c) kms_driven: '45,000 kms' -> 45000
    df["kms_driven"] = (df["kms_driven"].astype(str)
                        .str.replace(" kms", "", regex=False)
                        .str.replace(",", "", regex=False))
    df["kms_driven"] = pd.to_numeric(df["kms_driven"], errors="coerce")
    df = df.dropna(subset=["kms_driven"])
    df = df[df["kms_driven"] <= KMS_MAX]

    # (d) fuel_type: drop rows with missing fuel type
    df = df.dropna(subset=["fuel_type"])

    # (e) name: keep first 3 words of the listing title
    df["name"] = (df["name"].astype(str).str.split()
                  .str.slice(0, 3).str.join(" "))

    # (f) company: normalise case, keep only real brands
    df["company"] = df["company"].astype(str).str.strip().str.title()
    counts = df["company"].value_counts()
    real_brands = counts[counts >= MIN_COMPANY_LISTINGS].index
    df = df[df["company"].isin(real_brands)]

    return df.reset_index(drop=True)


def add_features(df: pd.DataFrame, current_year: int = 2026) -> pd.DataFrame:
    """
    Step 3: Feature engineering.

    We convert 'year' (year of purchase) into 'Car_Age' because a model
    understands "how old is the car" much better than a calendar year -
    depreciation depends on age, not on the year itself.

        Car_Age = current_year - year
    """
    df = df.copy()
    df["Car_Age"] = current_year - df["year"]
    return df


def prepare_dataset(path: str = DATASET_PATH, current_year: int = 2026) -> pd.DataFrame:
    """Run the complete preprocessing pipeline: load -> clean -> features."""
    df = load_raw_data(path)
    df = clean_data(df)
    df = add_features(df, current_year=current_year)
    return df


def prepare_single_record(name, company, year, kms_driven, fuel_type,
                          current_year: int = 2026) -> pd.DataFrame:
    """
    Build a one-row DataFrame from the web form, using the SAME column
    names and preprocessing as the training data. The Flask app calls
    this before model.predict().
    """
    record = {
        "name": name,
        "company": company,
        "fuel_type": fuel_type,
        "kms_driven": float(kms_driven),
        "Car_Age": current_year - int(year),
    }
    return pd.DataFrame([record])


if __name__ == "__main__":
    # Quick self-test when run directly:  python -m src.preprocess
    raw = load_raw_data()
    data = prepare_dataset()
    print(f"Raw rows     : {len(raw)}")
    print(f"Clean rows   : {len(data)}  ({len(raw) - len(data)} rows removed)")
    print(f"Columns      : {list(data.columns)}")
    print(f"Companies    : {data['company'].nunique()}")
    print(f"Models       : {data['name'].nunique()}")
    print(f"Price range  : Rs {data['Price'].min():,.0f} - "
          f"Rs {data['Price'].max():,.0f}")
    print(f"Missing cells: {data.isnull().sum().sum()}")
