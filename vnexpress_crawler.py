import csv
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
MAX_PAGES = 10
MAX_WORKERS = 10
CSV_FILE = "data/vnexpress_dataset.csv"
# Rate limiting parameters
MIN_DELAY = 1.0
MAX_DELAY = 3.0
URL = "https://vnexpress.net/"


def create_session():
    """Create a requests session with retry logic"""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        status_forcelist=[429, 500, 502, 503, 504],
        backoff_factor=1,
        respect_retry_after_header=True
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update(HEADERS)
    return session


def get_categories(session):
    res = session.get(URL)
    soup = BeautifulSoup(res.text, "lxml")
    menu_items = soup.select("nav[class*=main-nav] ul.parent li")

    categories = {}
    for item in menu_items:
        name = item.get("class", None)[0] if item.get("class", None) else None
        link = item.a["href"] if item.a else None
        categories[name] = link
    return categories


def get_articles_from_category(session, category_url, max_pages=MAX_PAGES):
    article_urls = set()

    for page in tqdm(range(1, max_pages + 1), desc="Pages", leave=False):
        url = f"{category_url}-p{page}" if page > 1 else category_url
        try:
            res = session.get(url, timeout=10)
            soup = BeautifulSoup(res.text, "lxml")
            items = soup.select("article.item-news h3.title-news a")
            for tag in items:
                article_urls.add(tag["href"])
            # Rate limiting with randomized delay
            time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
        except Exception as e:
            print(f"Error accessing {url}: {e}")

    return list(article_urls)


def get_article_content(session, article_url):
    try:
        res = session.get(article_url, timeout=10)
        soup = BeautifulSoup(res.text, "lxml")

        title_tag = soup.select_one("h1.title-detail")
        content_div = soup.select_one("article.fck_detail")
        views_tag = soup.select_one("span.icon-view")
        comments_tag = soup.select_one("span.txt-comment")

        if not title_tag or not content_div:
            return None

        paragraphs = content_div.find_all("p")
        content = "\n".join(p.get_text(strip=True) for p in paragraphs)

        return {
            "Title": title_tag.get_text(strip=True),
            "Link": article_url,
            "Views": views_tag.get_text(strip=True).replace(".", "") if views_tag else "0",
            "Comments": comments_tag.get_text(strip=True).replace("Bình luận", "").strip() if comments_tag else "0",
            "Content": content
        }
    except Exception as e:
        print(f"Error accessing article {article_url}: {e}")
        return None


def load_existing_urls(csv_file):
    urls = set()
    if os.path.exists(csv_file):
        with open(csv_file, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                urls.add(row["Link"])
    return urls


def save_to_csv(data: List[Dict[str, str]], filename: str = CSV_FILE) -> None:
    if not data:
        return

    fieldnames = ["Title", "Link", "Views", "Comments", "Content"]
    file_exists = os.path.exists(filename)

    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Use a context manager with explicit typing
    with open(filename, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)  # type: ignore
        if not file_exists:
            writer.writeheader()
        for item in data:
            writer.writerow(item)


def crawl_category(session, category_name, category_url, existing_urls):
    print(f"\n📌 Crawling category: {category_name}")
    article_urls = get_articles_from_category(session, category_url)
    new_urls = [url for url in article_urls if url not in existing_urls]

    print(f"✅ Found {len(new_urls)} new articles in category {category_name}")
    results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(get_article_content, session, url): url for url in new_urls}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Articles", leave=False):
            url = futures[future]
            try:
                data = future.result()
                if data:
                    results.append(data)
                # Rate limiting with randomized delay between requests
                time.sleep(random.uniform(MIN_DELAY / 2, MAX_DELAY / 2))
            except Exception as e:
                print(f"Error processing {url}: {e}")

    return results


def main():
    start_time = time.time()
    session = create_session()
    existing_urls = load_existing_urls(CSV_FILE)
    all_data = []
    categories = get_categories(session)

    print(f"Found {len(categories)} categories")
    for name, url in tqdm(categories.items(), desc="Categories"):
        category_data = crawl_category(session, name, URL[:-1] + url, existing_urls)
        all_data.extend(category_data)

        # Save incrementally after each category to prevent data loss
        if category_data:
            save_to_csv(category_data)
            print(f"✅ Wrote {len(category_data)} new articles from {name}")

        existing_urls.update([item["Link"] for item in category_data])

    total_time = time.time() - start_time
    print(f"\n✅ Completed crawling in {total_time:.2f} seconds")
    print(f"✅ Added {len(all_data)} new articles to {CSV_FILE}")
    return {"all_data": all_data, "total_time": total_time, "categories": categories}


if __name__ == "__main__":
    main()
