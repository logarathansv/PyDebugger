import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Define the target website (Replace with your target URL)
URL = "https://numpy.org/doc/stable/release/2.2.0-notes.html"
OUTPUT_FILE = "numpy2d2.txt"

def scrape_page(url):
    """Scrapes a single webpage and returns its text content and valid links."""
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to retrieve {url}")
        return "", []
    
    soup = BeautifulSoup(response.text, "html.parser")
    article_content = soup.find("article", class_="bd-article")
    
    if not article_content:
        print(f"Article section not found in {url}")
        return "", []
    
    content_text = " ".join(article_content.stripped_strings)
    links = []
    for a_tag in article_content.find_all("a", href=True):
        link = a_tag["href"]
        
        if link.startswith("#") or "pull" in link or "github.com" in link:
            continue  # Skip unwanted links

        full_link = urljoin(url, link)  # Ensure absolute URL
        if full_link.startswith("https"):
            links.append(full_link)
    return content_text, links

def scrape_website(url):
    """Scrapes the main page and follows links up to depth 1."""
    visited = set()
    all_content = []
    all_links = []
    
    # Scrape the main page
    print(f"Scraping: {url}")
    main_text, main_links = scrape_page(url)
    visited.add(url)
    all_content.append(f"Page: {url}\n{main_text}\n")
    all_links.extend(main_links)
    
    # Scrape linked pages (depth 1)
    for link in main_links:
        if link not in visited:
            print(f"Scraping (depth 1): {link}")
            text, _ = scrape_page(link)
            visited.add(link)
            all_content.append(f"Page: {link}\n{text}\n")
    
    # Write to a text file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        file.write("\n\n".join(all_content))
    
    print(f"Data saved to {OUTPUT_FILE}")

# Run the scraper
scrape_website(URL)
