import os
import pandas as pd

GBP_TO_INR = 105.50

RATING_MAP = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5
}


def clean_books():
    input_file = "data_pipeline/data/raw_books.csv"
    output_file = "data_pipeline/data/cleaned_books.csv"

    df = pd.read_csv(input_file)

    # ----------------------------
    # Price (clean and convert)
    # ----------------------------
    df["price_gbp"] = (
        df["price"]
        .astype(str)
        .str.replace(r"[^\d.]", "", regex=True)   # Remove everything except digits and decimal point
    )

    # Convert to numeric
    df["price_gbp"] = pd.to_numeric(df["price_gbp"], errors="coerce")

    # ----------------------------
    # Rating
    # ----------------------------
    df["rating"] = df["star_rating"].map(RATING_MAP)

    # ----------------------------
    # Stock Availability
    # ----------------------------
    df["in_stock"] = (
        df["availability"]
        .str.contains("In stock", case=False)
    )

    # ----------------------------
    # INR Conversion
    # ----------------------------
    df["price_inr"] = (
        df["price_gbp"] * GBP_TO_INR
    ).round(2)

    # ----------------------------
    # Handle parsing issues
    # ----------------------------
    before = len(df)

    df = df.dropna()

    after = len(df)

    print(f"Dropped {before-after} invalid rows.")

    # Keep only useful columns
    df = df[
        [
            "title",
            "category",
            "price_gbp",
            "price_inr",
            "rating",
            "in_stock"
        ]
    ]

    os.makedirs("data_pipeline/data", exist_ok=True)

    df.to_csv(output_file, index=False)

    print("\nCleaning Complete!")
    print(df.head())

    print(f"\nSaved cleaned dataset to:\n{output_file}")


if __name__ == "__main__":
    clean_books()