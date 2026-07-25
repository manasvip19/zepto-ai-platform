import sqlite3
import pandas as pd
import os

DB_PATH = "data_pipeline/data/books.db"
CSV_PATH = "data_pipeline/data/cleaned_books.csv"


def create_database():

    os.makedirs("data_pipeline/data", exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Enable foreign keys
    cursor.execute("PRAGMA foreign_keys = ON;")

    # Drop old tables if they exist
    cursor.execute("DROP TABLE IF EXISTS books")
    cursor.execute("DROP TABLE IF EXISTS categories")

    # Categories table
    cursor.execute("""
        CREATE TABLE categories (
            category_id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_name TEXT UNIQUE
        )
    """)

    # Books table
    cursor.execute("""
        CREATE TABLE books (
            book_id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            price_gbp REAL,
            price_inr REAL,
            rating INTEGER,
            in_stock INTEGER,
            category_id INTEGER,
            FOREIGN KEY(category_id)
                REFERENCES categories(category_id)
        )
    """)

    df = pd.read_csv(CSV_PATH)

    # Insert categories
    categories = sorted(df["category"].unique())

    for category in categories:
        cursor.execute(
            "INSERT INTO categories(category_name) VALUES (?)",
            (category,)
        )

    conn.commit()

    # Get category IDs
    category_map = {}

    cursor.execute(
        "SELECT category_id, category_name FROM categories"
    )

    for cid, cname in cursor.fetchall():
        category_map[cname] = cid

    # Insert books
    for _, row in df.iterrows():

        cursor.execute("""
            INSERT INTO books
            (
                title,
                price_gbp,
                price_inr,
                rating,
                in_stock,
                category_id
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            row["title"],
            row["price_gbp"],
            row["price_inr"],
            int(row["rating"]),
            int(row["in_stock"]),
            category_map[row["category"]]
        ))

    conn.commit()

    # Print summary
    cursor.execute("SELECT COUNT(*) FROM books")
    books = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM categories")
    cats = cursor.fetchone()[0]

    print(f"\nDatabase created successfully!")
    print(f"Books inserted      : {books}")
    print(f"Categories inserted : {cats}")

    conn.close()


if __name__ == "__main__":
    create_database()