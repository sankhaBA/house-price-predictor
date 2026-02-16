import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
from tqdm import tqdm  # Shows a progress bar
import os
from datetime import datetime

# ==========================================
# 1. LOAD & DEDUPLICATE LINKS
# ==========================================

# Assuming your links are in a file named 'links.txt'
# Upload this file to Colab/Notebook folder first
with open('data/ad-links/ad_links.txt', 'r') as f:
    raw_links = f.readlines()

# Clean whitespace and remove duplicates
# Using set() automatically removes duplicates
unique_links = list(set([link.strip() for link in raw_links if "ikman.lk" in link]))

print(f"Original Count: {len(raw_links)}")
print(f"Unique Links to Scrape: {len(unique_links)}")
print("-" * 30)

# ==========================================
# 2. DEFINE EXTRACTION LOGIC
# ==========================================
def extract_ad_data(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return None
            
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Initialize all fields with empty strings to prevent field shifting
        data = {
            'URL': url,
            'Price': '',
            'Address': '',
            'Bedrooms': '',
            'Bathrooms': '',
            'House_Size': '',
            'Land_Size': '',
            'Description': ''
        }
        
        # 1. EXTRACT PRICE
        # Logic: Find div with class containing 'amount--'
        price_tag = soup.find('div', class_=lambda x: x and 'amount--' in x)
        if price_tag:
            # Cleans "Rs 18,500,000" -> "18500000"
            data['Price'] = price_tag.get_text(strip=True).replace('Rs', '').replace(',', '').strip()

        # 2. EXTRACT ALL DETAILS (Address, Beds, Size, etc.)
        # Logic: Find all 'labels' and get their neighbor 'values'
        
        # Find all divs that look like labels
        labels = soup.find_all('div', class_=lambda x: x and 'label--' in x)
        
        for label in labels:
            key = label.get_text(strip=True).replace(':', '') # e.g., "Land size"
            
            # The value is usually the NEXT sibling div
            value_div = label.find_next_sibling('div')
            
            if value_div:
                # get_text() automatically handles the nested <a><span>3</span></a> issue
                value = value_div.get_text(strip=True)
                
                # Map to our standard column names
                if "Address" in key:
                    data['Address'] = value
                elif "Bedrooms" in key:
                    data['Bedrooms'] = value
                elif "Bathrooms" in key:
                    data['Bathrooms'] = value
                elif "House size" in key:
                    data['House_Size'] = value # e.g., "1,134.0 sqft"
                elif "Land size" in key:
                    data['Land_Size'] = value  # e.g., "9.5 perches"

        # 3. EXTRA: DESCRIPTION (Optional, but good for checking "Unfinished" later)
        desc_tag = soup.find('div', class_=lambda x: x and 'description--' in x)
        if desc_tag:
             data['Description'] = desc_tag.get_text(strip=True)[:200] # First 200 chars only

        return data

    except Exception as e:
        return None

# ==========================================
# 3. RUN SCRAPER WITH CHECKPOINTS
# ==========================================

data_buffer = []
failed_links = []  # Track failed links
csv_filename = "data/ad-data/raw_house_data.csv"

# Create a progress bar
print("Starting Scraping... (Press Stop if you need to pause)")

for i, link in tqdm(enumerate(unique_links), total=len(unique_links)):
    
    # Extract data
    row = extract_ad_data(link)
    
    if row:
        data_buffer.append(row)
    else:
        # Track failed link
        failed_links.append(link)
    
    # SAVE EVERY 50 LINKS (Checkpointing)
    if len(data_buffer) >= 50:
        df_chunk = pd.DataFrame(data_buffer)
        
        # If file doesn't exist, write header. If it does, append without header.
        if not os.path.isfile(csv_filename):
            df_chunk.to_csv(csv_filename, index=False)
        else:
            df_chunk.to_csv(csv_filename, mode='a', header=False, index=False)
        
        data_buffer = [] # Clear buffer
        
    # Rate Limiting (Crucial to avoid ban)
    time.sleep(random.uniform(0.5, 1.5)) 

# Save any remaining data
if data_buffer:
    df_chunk = pd.DataFrame(data_buffer)
    if not os.path.isfile(csv_filename):
        df_chunk.to_csv(csv_filename, index=False)
    else:
        df_chunk.to_csv(csv_filename, mode='a', header=False, index=False)

print(f"Success! Data saved to {csv_filename}")

# Save failed links to a timestamped file
if failed_links:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    failed_links_filename = f"data/failed-links/failed_links_{timestamp}.txt"
    with open(failed_links_filename, 'w') as f:
        f.write("\n".join(failed_links))
    print(f"Failed links saved to {failed_links_filename} ({len(failed_links)} links)")
else:
    print("No failed links!")