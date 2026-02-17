"""
City Extractor for House Sales Data

This script extracts city information from house sale advertisements
by analyzing the Address, URL, and Description fields.
"""

import pandas as pd
import re
from pathlib import Path
from difflib import SequenceMatcher


def load_city_list():
    """
    Load city list from external file.
    
    Returns:
        list: List of city names
    """
    base_dir = Path(__file__).parent.parent
    city_file = base_dir / "data" / "04_additional" / "city_list.txt"
    
    try:
        with open(city_file, 'r', encoding='utf-8') as f:
            cities = [line.strip() for line in f if line.strip()]
        return cities
    except FileNotFoundError:
        print(f"Warning: City list file not found at {city_file}")
        print("Using default city list")
        # Fallback to default list if file not found
        return [
            "Colombo 1", "Colombo 2", "Colombo 3", "Colombo 4", "Colombo 5",
            "Colombo 6", "Colombo 7", "Colombo 8", "Colombo 9", "Colombo 10",
            "Colombo 11", "Colombo 12", "Colombo 13", "Colombo 14", "Colombo 15",
            "Athurugiriya", "Avissawella", "Angoda", "Bambalapitiya", "Battaramulla", "Bokundara", "Boralesgamuwa", "Borella",
            "Dehiwala", "Dematagoda", "Diyagama", "Fort", "Godagama", "Grandpass", "Hanwella", "Havelock Town",
            "Hokandara", "Homagama", "Kaduwela", "Kahathuduwa", "Kalubowila", "Kesbewa", "Kirulapone",
            "Kollupitiya", "Kolonnawa", "Kotahena", "Kottawa", "Kotte", "Kohuwala", "Madapatha", "Maharagama", "Malabe",
            "Maradana", "Mattakkuliya", "Mattegoda", "Modara", "Moratuwa", "Meegoda", "Mount Lavinia", "Mulleriyawa",
            "Narahenpita", "Nawala", "Nugegoda", "Padukka", "Pamankada", "Pannipitiya", "Pelawatte",
            "Peliyagoda", "Pettah", "Polgasowita", "Piliyandala", "Rajagiriya", "Ratmalana", "Slave Island",
            "Talawatugoda", "Thalawathugoda", "Wellampitiya", "Wellawatte"
        ]


# Load city list from external file
CITY_LIST = load_city_list()

# Fuzzy matching threshold (0.0 to 1.0)
# 0.8 means 80% similarity required for a match
FUZZY_MATCH_THRESHOLD = 0.8


def calculate_similarity(str1, str2):
    """
    Calculate similarity ratio between two strings using SequenceMatcher.
    
    Args:
        str1: First string to compare
        str2: Second string to compare
        
    Returns:
        float: Similarity ratio between 0.0 and 1.0
    """
    return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()


def generate_ngrams(tokens, n):
    """
    Generate n-grams from a list of tokens.
    
    Args:
        tokens: List of words
        n: Size of n-grams
        
    Returns:
        list: List of n-gram strings
    """
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def extract_city_exact(text, city_list):
    """
    Extract city using exact matching with word boundaries.
    
    Args:
        text: Text to search in
        city_list: List of cities to search for
        
    Returns:
        str or None: Matched city name or None if no match found
    """
    text_lower = text.lower()
    
    for city in city_list:
        city_lower = city.lower()
        # Use regex word boundaries for exact matching
        pattern = r'\b' + re.escape(city_lower) + r'\b'
        
        if re.search(pattern, text_lower):
            return city
    
    return None


def extract_city_fuzzy(text, city_list, threshold=FUZZY_MATCH_THRESHOLD):
    """
    Extract city using fuzzy matching with n-grams.
    Handles variations and typos in city names (e.g., "Baththaramulla" -> "Battaramulla").
    
    Args:
        text: Text to search in
        city_list: List of cities to search for
        threshold: Minimum similarity score (0.0 to 1.0)
        
    Returns:
        str or None: Best matching city name or None if no good match found
    """
    # Preprocessing: lowercase and remove special characters
    text_cleaned = re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower())
    tokens = text_cleaned.split()
    
    if not tokens:
        return None
    
    best_match = None
    highest_score = 0.0
    
    # Check n-grams of size 1 to 3 to catch multi-word city names
    # Size 1: Single words like "Colombo", "Malabe"
    # Size 2: Two-word names like "Mount Lavinia"
    # Size 3: Three-word names like "Colombo 1 Fort" (edge cases)
    for n in range(1, 4):
        if n > len(tokens):
            break
            
        ngrams = generate_ngrams(tokens, n)
        
        for phrase in ngrams:
            # Compare this phrase against every city in the master list
            for city in city_list:
                similarity = calculate_similarity(phrase, city)
                
                # Update if we found a better match above threshold
                if similarity > highest_score and similarity >= threshold:
                    highest_score = similarity
                    best_match = city
    
    return best_match


def extract_city(row):
    """
    Extract city name from a row by checking Address, URL, and Description fields.
    Uses exact matching first (fast), then falls back to fuzzy matching (slower but handles typos).
    
    Args:
        row: A pandas Series representing a single row of data
        
    Returns:
        str: The extracted city name or "Other" if no city is found
    """
    # Get field values
    address = str(row['Address']) if pd.notna(row['Address']) else ""
    url = str(row['URL']) if pd.notna(row['URL']) else ""
    desc = str(row['Description']) if pd.notna(row['Description']) else ""
    
    # Priority 1: Try exact matching on Address (most reliable and fast)
    if address:
        city = extract_city_exact(address, CITY_LIST)
        if city:
            return city
    
    # Priority 2: Try exact matching on URL (auto-generated, high confidence)
    if url:
        city = extract_city_exact(url, CITY_LIST)
        if city:
            return city
    
    # Priority 3: Try exact matching on Description
    if desc:
        city = extract_city_exact(desc, CITY_LIST)
        if city:
            return city
    
    # Priority 4: Fallback to fuzzy matching on combined text
    # This handles typos like "Baththaramulla" -> "Battaramulla"
    combined_text = f"{address} {url} {desc}"
    if combined_text.strip():
        city = extract_city_fuzzy(combined_text, CITY_LIST)
        if city:
            return city
    
    return "Other"


def print_dataset_summary(df):
    """
    Print a comprehensive summary of the dataset with city distribution.
    
    Args:
        df: DataFrame with extracted city information
    """
    print("\n" + "="*70)
    print(" DATASET SUMMARY - CITY DISTRIBUTION")
    print("="*70)
    
    # Overall statistics
    total_records = len(df)
    print(f"\nTotal Records: {total_records}")
    print(f"Total Cities Found: {df['City'].nunique()}")
    
    # City distribution with counts and percentages
    city_counts = df['City'].value_counts()
    
    print(f"\n{'City':<25} {'Count':>10} {'Percentage':>12}")
    print("-" * 50)
    
    for city, count in city_counts.items():
        percentage = (count / total_records) * 100
        print(f"{city:<25} {count:>10} {percentage:>11.2f}%")
    
    # Special highlight for "Other" category
    other_count = city_counts.get('Other', 0)
    other_percentage = (other_count / total_records) * 100 if total_records > 0 else 0
    
    print("\n" + "-" * 50)
    print(f"\n📊 Ads with identified cities: {total_records - other_count} ({100 - other_percentage:.2f}%)")
    print(f"📊 Ads without identified cities (Other): {other_count} ({other_percentage:.2f}%)")
    print("="*70 + "\n")


def process_house_data(input_file, output_dir):
    """
    Process house sale data and extract city information.
    
    Args:
        input_file: Path to the input CSV file
        output_dir: Directory to save the processed CSV file
    """
    # Read the input CSV
    print(f"Reading data from: {input_file}")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} records")
    
    # Extract city information
    print("\nExtracting city information...")
    print("Using exact matching first, then fuzzy matching for variations and typos...")
    print(f"Fuzzy matching threshold: {FUZZY_MATCH_THRESHOLD * 100:.0f}% similarity")
    df['City'] = df.apply(extract_city, axis=1)
    
    # Display comprehensive summary
    print_dataset_summary(df)
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename
    output_file = output_path / "house_data_with_city.csv"
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"✅ Processed data saved to: {output_file}")
    print(f"✅ Total records processed: {len(df)}")
    
    return df


def main():
    """Main execution function"""
    # Define file paths
    base_dir = Path(__file__).parent.parent
    input_file = base_dir / "data" / "01_raw" / "raw_house_data.csv"
    output_dir = base_dir / "data" / "02_intermediate"
    
    # Check if input file exists
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        return
    
    # Process the data
    process_house_data(input_file, output_dir)


if __name__ == "__main__":
    main()
