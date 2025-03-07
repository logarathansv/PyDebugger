import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Define the target website (Replace with your target URL)
URL = "https://matplotlib.org/stable/api/prev_api_changes/api_changes_3.9.0.html"
OUTPUT_FILE = "matplotlib.txt"

def scrape_website(url):
    # Fetch the webpage
    response = requests.get(url)
    if response.status_code != 200:
        print("Failed to retrieve the webpage")
        return

    # Parse HTML content
    soup = BeautifulSoup(response.text, "html.parser")

    # Find the main content section
    main_content = soup.find("main", id="main-content", class_="bd-main")
    
    if not main_content:
        print("Main content section not found")
        return
    
    # Extract text from the main content
    content_text = main_content.get_text(separator=" ", strip=True)
    
    # Extract all links inside <main>
    links = []
    for a_tag in main_content.find_all("a", href=True):
        full_url = urljoin(url, a_tag["href"])  # Convert relative URLs to absolute
        links.append(full_url)
    
    # Write to a text file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        file.write(content_text + "\n\n")
    
    print(f"Data saved to {OUTPUT_FILE}")

# Run the scraper
scrape_website(URL)