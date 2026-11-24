"""
eda.py
======
Exploratory Data Analysis for the Car Price Predictor (Quikr dataset).

Generates charts (saved into static/images/eda/) and prints key insights
about the dataset. These charts are shown on the website's "Project
Details" page and can be pasted directly into your presentation slides.

Run:  python -m src.eda
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # render charts to files, no screen needed
import matplotlib.pyplot as plt
import seaborn as sns

from src.preprocess import prepare_dataset, load_raw_data

OUTPUT_DIR = os.path.join("static", "images", "eda")

# A clean, professional colour palette
COLOR_MAIN = "#2563eb"      # blue
COLOR_ACCENT = "#16a34a"    # green
COLOR_WARM = "#ea580c"      # orange
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.titleweight"] = "bold"


def lakh(x, pos=None):
    """Format an axis in lakhs (Rs 4,25,000 -> 4.25 L)."""
    return f"{x/100000:.0f}L"


def save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    insights = {}

    raw = load_raw_data()
    df = prepare_dataset()

    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS - Used Car Dataset (Quikr.com)")
    print("=" * 60)
    print(f"Raw rows: {len(raw)}   Clean rows: {len(df)}")

    # ------------------------------------------------------------------ 1
    fig, ax = plt.subplots()
    sns.histplot(df["Price"] / 100000, bins=40, kde=True, color=COLOR_MAIN, ax=ax)
    ax.set_title("Distribution of Car Prices")
    ax.set_xlabel("Price (lakh rupees)")
    ax.set_ylabel("Number of cars")
    save(fig, "01_price_distribution.png")
    insights["price_mean_lakh"] = round(float(df["Price"].mean() / 100000), 2)
    insights["price_median_lakh"] = round(float(df["Price"].median() / 100000), 2)

    # ------------------------------------------------------------------ 2
    fig, ax = plt.subplots()
    grouped = df.groupby("Car_Age")["Price"].mean()
    ax.bar(grouped.index, grouped.values, color=COLOR_ACCENT)
    ax.set_title("Average Price by Car Age")
    ax.set_xlabel("Age of car (years)")
    ax.set_ylabel("Average price (Rs)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lakh))
    save(fig, "02_age_vs_price.png")

    # ------------------------------------------------------------------ 3
    fig, ax = plt.subplots()
    order = df.groupby("company")["Price"].count().sort_values(
        ascending=False).index[:12]
    med = df[df["company"].isin(order)].groupby("company")["Price"].median()
    med = med.loc[med.abs().sort_values().index]
    ax.barh(med.index, med.values, color=COLOR_MAIN)
    ax.set_title("Median Price by Company (top 12 by listings)")
    ax.set_xlabel("Median price (Rs)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lakh))
    save(fig, "03_company_vs_price.png")

    # ------------------------------------------------------------------ 4
    fig, ax = plt.subplots()
    order = df.groupby("fuel_type")["Price"].median().sort_values().index
    sns.boxplot(data=df, x="fuel_type", y="Price", order=order,
                hue="fuel_type", palette="Blues", legend=False, ax=ax)
    ax.set_title("Price by Fuel Type")
    ax.set_xlabel("Fuel type")
    ax.set_ylabel("Price (Rs)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lakh))
    save(fig, "04_fuel_type.png")

    # ------------------------------------------------------------------ 5
    fig, ax = plt.subplots()
    ax.scatter(df["kms_driven"] / 1000, df["Price"] / 100000, alpha=0.5,
               color=COLOR_WARM, edgecolors="white")
    ax.set_title("Price vs Kilometres Driven")
    ax.set_xlabel("Kilometres driven (thousands)")
    ax.set_ylabel("Price (lakh rupees)")
    save(fig, "05_kms_vs_price.png")
    insights["corr_kms_price"] = round(
        float(df["kms_driven"].corr(df["Price"])), 3)

    # ------------------------------------------------------------------ 6
    fig, ax = plt.subplots(figsize=(7, 5))
    corr = df[["Price", "kms_driven", "Car_Age"]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                square=True, ax=ax, cbar_kws={"label": "Correlation"})
    ax.set_title("Correlation Heatmap (numeric features)")
    save(fig, "06_correlation_heatmap.png")
    insights["corr_age_price"] = round(
        float(df["Car_Age"].corr(df["Price"])), 3)

    # ------------------------------------------------------------------ 7
    top = df["name"].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.barh(top.index[::-1], top.values[::-1], color=COLOR_MAIN)
    ax.set_title("Top 10 Most Frequent Car Models")
    ax.set_xlabel("Number of listings")
    save(fig, "07_top_models.png")

    # ------------------------------------------------------------------ 8
    fig, ax = plt.subplots()
    counts = df["company"].value_counts().head(10)
    ax.bar(counts.index, counts.values, color=COLOR_ACCENT)
    ax.set_title("Listings per Company (top 10)")
    ax.set_ylabel("Number of listings")
    ax.tick_params(axis="x", rotation=45)
    save(fig, "08_company_counts.png")

    # ------------------------------------------------------------- summary
    print("\nKEY INSIGHTS")
    print("-" * 60)
    print(f"Average price   : Rs {insights['price_mean_lakh']} lakh")
    print(f"Median price    : Rs {insights['price_median_lakh']} lakh")
    print(f"Corr age vs price    : {insights['corr_age_price']}")
    print(f"Corr kms vs price    : {insights['corr_kms_price']}")
    print(f"Most common fuel: {df['fuel_type'].value_counts().idxmax()}")
    print(f"Companies       : {df['company'].nunique()}")
    print(f"Models          : {df['name'].nunique()}")

    insights["n_raw"] = int(len(raw))
    insights["n_clean"] = int(len(df))
    insights["n_models"] = int(df["name"].nunique())
    insights["year_min"] = int(df["year"].min())
    insights["year_max"] = int(df["year"].max())
    with open(os.path.join(OUTPUT_DIR, "insights.json"), "w") as f:
        json.dump(insights, f, indent=2)
    print(f"\nAll charts saved in {OUTPUT_DIR}/ and insights.json written.")


if __name__ == "__main__":
    main()
