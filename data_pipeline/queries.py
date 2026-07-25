import sqlite3
import pandas as pd

DB_PATH = "data_pipeline/data/books.db"

conn = sqlite3.connect(DB_PATH)

queries = {
    "Query 1 - SELECT + WHERE": """
        SELECT title, rating
        FROM books
        WHERE rating = 5;
    """,

    "Query 2 - ORDER BY": """
        SELECT title, price_gbp
        FROM books
        ORDER BY price_gbp DESC;
    """,

    "Query 3 - LIMIT": """
        SELECT title, category_name
        FROM books
        JOIN categories
        ON books.category_id = categories.category_id
        LIMIT 10;
    """,

    "Query 4 - DISTINCT": """
        SELECT DISTINCT category_name
        FROM categories;
    """,

    "Query 5 - BETWEEN": """
        SELECT title, price_gbp
        FROM books
        WHERE price_gbp BETWEEN 20 AND 40;
    """,

    "JOIN Query": """
        SELECT
            books.title,
            categories.category_name,
            books.rating,
            books.price_gbp
        FROM books
        JOIN categories
        ON books.category_id = categories.category_id
        ORDER BY categories.category_name, books.rating DESC;
    """
}

print("=" * 60)

for name, query in queries.items():

    print(f"\n{name}")
    print("-" * 60)

    df = pd.read_sql(query, conn)

    print(df.head())

    if name != "JOIN Query":
        output = name.lower().replace(" ", "_").replace("-", "")
        df.to_csv(f"data_pipeline/outputs/{output}.csv", index=False)

# --------------------------------------------------
# Verify JOIN using pandas.merge()
# --------------------------------------------------

print("\n" + "=" * 60)
print("Comparing SQL JOIN vs pandas.merge()")
print("=" * 60)

books = pd.read_sql("SELECT * FROM books", conn)
categories = pd.read_sql("SELECT * FROM categories", conn)

sql_join = pd.read_sql(
    queries["JOIN Query"],
    conn
)

merge_join = books.merge(
    categories,
    on="category_id"
)[
    [
        "title",
        "category_name",
        "rating",
        "price_gbp"
    ]
].sort_values(
    ["category_name", "rating"],
    ascending=[True, False]
).reset_index(drop=True)

sql_join = sql_join.reset_index(drop=True)

print("\nSQL JOIN")
print(sql_join.head())

print("\npandas.merge()")
print(merge_join.head())

print("\nEquivalent:",
      sql_join.equals(merge_join))

sql_join.to_csv(
    "data_pipeline/outputs/sql_join.csv",
    index=False
)

merge_join.to_csv(
    "data_pipeline/outputs/pandas_merge.csv",
    index=False
)

conn.close()