import os
import requests
import pandas as pd
from bs4 import BeautifulSoup

BASE_URL = "https://books.toscrape.com/"


def get_soup(url):
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def get_category_links():
    """Get the first 3 category links."""
    soup = get_soup(BASE_URL)

    categories = {}

    category_list = soup.select("div.side_categories ul li ul li a")

    for category in category_list[:3]:
        name = category.text.strip()
        link = BASE_URL + category["href"]
        categories[name] = link

    return categories


def scrape_category(category_name, category_url):
    books = []

    next_page = category_url

    while next_page:

        soup = get_soup(next_page)

        for book in soup.select("article.product_pod"):

            title = book.h3.a["title"]

            price = book.select_one(".price_color").text.strip()

            availability = book.select_one(".instock.availability").text.strip()

            rating = book.p["class"][1]

            books.append(
                {
                    "title": title,
                    "price": price,
                    "star_rating": rating,
                    "availability": availability,
                    "category": category_name,
                }
            )

        next_btn = soup.select_one("li.next a")

        if next_btn:
            next_page = next_page.rsplit("/", 1)[0] + "/" + next_btn["href"]
        else:
            next_page = None

    return books


def main():

    os.makedirs("data_pipeline/data", exist_ok=True)

    categories = get_category_links()

    all_books = []

    for name, url in categories.items():
        print(f"Scraping {name}...")
        all_books.extend(scrape_category(name, url))

    df = pd.DataFrame(all_books)

    output = "data_pipeline/data/raw_books.csv"

    df.to_csv(output, index=False)

    print(f"\nCollected {len(df)} books.")
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()