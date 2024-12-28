from bs4 import BeautifulSoup
import requests
import pandas as pd
import json
import time
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

def parse_categories_from_text(text_content):
    """Parse the text file content into a dictionary of categories and their URLs."""
    categories_dict = {}
    current_category = None
    lines = text_content.split('\n')
    
    print("Starting to parse text content...")
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Debug print
        print(f"Processing line: {line[:100]}...")  # Print first 100 chars of line
            
        if ': [' in line:  # Changed condition to match the format
            # Extract category name from the line
            current_category = line.split(':')[0].strip("'")
            categories_dict[current_category] = []
            print(f"Found category: {current_category}")
            
        elif line.startswith('"http') or line.startswith('http'):
            url = line.strip('",')
            if current_category:
                categories_dict[current_category].append(url)
                print(f"Added URL to {current_category}: {url}")

    # Debug print final structure
    print("\nParsed Categories:")
    for category, urls in categories_dict.items():
        print(f"\n{category}:")
        print(f"Number of URLs: {len(urls)}")
        if urls:
            print(f"First URL: {urls[0]}")
    
    return categories_dict

def create_session_with_retries():
    """Create a session with retry strategy."""
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[500, 502, 503, 504]
    )
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session

def extract_data_from_html(html_content, category):
    """Extract structured information from the HTML content."""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Section mapping for better clarity
    section_map = {
        "Description": "description",
        "Directions": "directions",
        "Warnings": "warning",
        "Support Your Goals": "goals",
        "About the Brand": "about"
    }
    
    extracted_data = {"category": category}
    
    # Extract product title
    product_title = soup.find('h1', {'class': 'product__title heading-size-5'})
    extracted_data['product_title'] = product_title.get_text(strip=True) if product_title else ""
    
    # Process each <details> section
    for details in soup.find_all('details', {'class': 'accordion'}):
        summary = details.find('summary', {'class': 'accordion__title'})
        if summary:
            section_title = summary.get_text(strip=True)
            if section_title in section_map:
                content_div = details.find('div', {'class': 'accordion__body rte'})
                if content_div:
                    # Extracting text from the content area
                    content_text = content_div.get_text(strip=True)
                    extracted_data[section_map[section_title]] = content_text

    return extracted_data


def main():
    try:
        print("Starting script execution...")
        
        # Read the text file content
        with open('Chatbot/urls.txt', 'r') as file:
            text_content = file.read()
            print(f"\nRead file content length: {len(text_content)} characters")
            print("\nFirst 200 characters of file content:")
            print(text_content[:200])
        
        # Parse categories and URLs
        categories_url = parse_categories_from_text(text_content)
        
        if not categories_url:
            print("\nERROR: No categories were parsed from the file!")
            return
        
        # Create session with retry strategy
        session = create_session_with_retries()
        
        # List to store all extracted data
        data = []
        
        # Process each category and URL
        for category, urls in categories_url.items():
            print(f"\nProcessing category: {category}")
            print(f"Number of URLs in category: {len(urls)}")
            
            for url in urls:
                print(f"Fetching: {url}")
                try:
                    response = session.get(url, timeout=10)
                    response.raise_for_status()
                    
                    html_content = response.text
                    extracted_data = extract_data_from_html(html_content, category)
                    extracted_data['url'] = url
                    data.append(extracted_data)
                    
                    # Add a small delay
                    time.sleep(1)
                    
                except requests.exceptions.RequestException as e:
                    print(f"Error fetching {url}: {str(e)}")
                    continue
                except Exception as e:
                    print(f"Unexpected error processing {url}: {str(e)}")
                    continue
        
        if not data:
            print("\nERROR: No data was extracted!")
            return
            
        # Convert to DataFrame and save
        df = pd.DataFrame(data)
        df.to_csv("extracted_data.csv", index=False, encoding='utf-8')
        print("\nData extraction complete. Saved to extracted_data.csv")
        
        # Print summary
        print(f"\nSummary:")
        print(f"Total products processed: {len(data)}")
        print(f"Categories processed: {len(categories_url)}")
        print("\nProducts per category:")
        print(df['category'].value_counts())
        
    except FileNotFoundError:
        print("Error: urls.txt file not found!")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise

if __name__ == "__main__":
    main()