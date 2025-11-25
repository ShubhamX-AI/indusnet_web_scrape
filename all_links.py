import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def get_all_links(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()  # Raise exception for bad status codes
        soup = BeautifulSoup(r.text, "html.parser")

        links = set()

        for a in soup.find_all("a", href=True):
            full_url = urljoin(url, a["href"])
            links.add(full_url)

        return list(links)
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the URL: {e}")
        return []

links = get_all_links("https://intglobal.com/")
print(f"Found {len(links)} links:")
for link in links:
    print(link)
