import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time

# --- Configuration ---
START_URL = "https://intglobal.com/"
DOMAIN_NAME = "intglobal.com" # Used to ensure we only stay on this site
OUTPUT_FILE = "intglobal_links.txt"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

def is_internal_link(url):
    """Checks if the URL belongs to the specific domain."""
    try:
        parsed_url = urlparse(url)
        # Check if the netloc (domain) ends with our target domain
        return parsed_url.netloc == DOMAIN_NAME or parsed_url.netloc.endswith("." + DOMAIN_NAME)
    except:
        return False

def load_existing_links():
    """Reads the file to check what we have already found to avoid duplicates."""
    try:
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            # Read lines and strip whitespace/newlines
            return set(line.strip() for line in f if line.strip())
    except FileNotFoundError:
        return set()

def save_link_to_file(url):
    """Appends a single new link to the file."""
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(url + "\n")

def crawl_website(start_url):
    # 1. Load all links currently in the file so we don't duplicate logic
    # This acts as our "Master List" of known URLs
    known_links = load_existing_links()
    
    # 2. Define a set for URLs we have actually downloaded and parsed
    # (Just because a link is in the file doesn't mean we have crawled IT yet)
    visited_links = set()

    # 3. Create a queue (list) of URLs to visit. 
    # If the file was empty, start with start_url. 
    # If the file had data, we might want to resume, but for simplicity, 
    # let's make sure the start_url is in the queue.
    urls_to_visit = [start_url]
    
    # Add start_url to known_links if not there
    if start_url not in known_links:
        save_link_to_file(start_url)
        known_links.add(start_url)

    print(f"--- Starting Crawl on {start_url} ---")

    # 4. The Loop: Go on until there are no URLs left to visit
    while urls_to_visit:
        # Get the next URL from the list
        current_url = urls_to_visit.pop(0)

        # Skip if we already scraped this specific page during this session
        if current_url in visited_links:
            continue

        print(f"Crawling: {current_url}")

        try:
            # Request the page
            r = requests.get(current_url, headers=HEADERS, timeout=10)
            
            # If the content type isn't HTML (e.g., PDF, JPG), skip parsing
            content_type = r.headers.get('Content-Type', '')
            if 'text/html' not in content_type:
                visited_links.add(current_url)
                continue

            soup = BeautifulSoup(r.text, "html.parser")
            visited_links.add(current_url)

            # Find all links on this page
            found_on_page = 0
            for a in soup.find_all("a", href=True):
                href = a["href"]
                
                # Make the URL absolute (handle relative paths like /about)
                full_url = urljoin(current_url, href)
                
                # Remove fragments (e.g., #section1) to avoid duplicates
                full_url = full_url.split('#')[0] 
                
                # Remove trailing slashes for consistency
                full_url = full_url.rstrip('/')

                # LOGIC:
                # 1. Must be internal (intglobal.com)
                # 2. Must NOT be in our known_links (file) already
                if is_internal_link(full_url) and full_url not in known_links:
                    
                    # Save to file immediately
                    save_link_to_file(full_url)
                    
                    # Add to memory set
                    known_links.add(full_url)
                    
                    # Add to queue to crawl this link later
                    urls_to_visit.append(full_url)
                    found_on_page += 1

            print(f"   -> Found {found_on_page} new links on this page.")
            
            # Be polite to the server, don't hammer it too fast
            time.sleep(1) 

        except requests.exceptions.RequestException as e:
            print(f"   -> Error accessing {current_url}: {e}")
        except KeyboardInterrupt:
            print("\nStopped by user.")
            break
        except Exception as e:
            print(f"   -> Unknown Error: {e}")

    print("--- Crawling Finished. All internal links saved. ---")

if __name__ == "__main__":
    crawl_website(START_URL)