import json
import pandas as pd
import re
import spacy
import matplotlib.pyplot as plt


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_to_csv(df, filename):
    df.to_csv(filename, index=False)
    print(f"Saved: {filename}")


# Check skewness 
def analyze_price_distribution(df):
    print("Price Skewness:", df['Price'].skew())

    plt.hist(df['Price'], bins=100)
    plt.xlabel("Price")
    plt.ylabel("Frequency")
    plt.title("Price Distribution")
    plt.show()


# Preprocessing

def preprocess_regex(text, nlp):
    if not isinstance(text, str) or not text.strip():
        return ""

    text = re.sub(r'[^\w\s]', '', text.lower())  # Normalize text
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces

    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc if not token.is_stop and len(token.lemma_) > 2])


def process_text_columns(df, nlp):
    df["Combined_Text"] = (
        df["Title"] + " " +
        df["Description"].apply(lambda x: " ".join(x)) + " " +
        df["Features"].apply(lambda x: " ".join(x))
    )
    df["Processed_Text"] = df["Combined_Text"].apply(lambda x: preprocess_regex(x, nlp))
    return df.drop(columns=["Title", "Description", "Features", "Combined_Text"])


# Calculate volume from details.dimensions

def calculate_volume(dim_str):
    if pd.isna(dim_str) or not str(dim_str).strip():
        return None

    numbers = list(map(float, re.findall(r'[\d\.]+', dim_str)))
    return round(numbers[0] * numbers[1] * numbers[2], 2) if len(numbers) >= 3 else None


def extract_dimensions(df):
    """Extract and calculate volume from dimension-related columns."""
    columns_to_concat = [
        'Details.Product Dimensions', 'Details.Package Dimensions',
        'Details.Item Dimensions LxWxH', 'Details.Item Package Dimensions L x W x H',
        'Details.Item Dimensions  LxWxH'
    ]

    df['final_dimensions'] = df[columns_to_concat].astype(str).agg(' '.join, axis=1)
    df['Volume'] = df['final_dimensions'].apply(calculate_volume)
    return df.drop(columns=columns_to_concat + ['final_dimensions'])


# calculate and standardize weights

def convert_to_lbs(value, unit):
    conversion_factors = {
        "kg": 2.20462, "kilogram": 2.20462,
        "oz": 0.0625, "ounce": 0.0625,
        "lb": 1, "pound": 1
    }
    return value * conversion_factors.get(unit.split()[0], 1) if unit else value


def extract_weight(details):
    if not isinstance(details, dict):
        return None

    weight_pattern = re.compile(r'weight|mass|wt|wgt', re.IGNORECASE)
    exclude_pattern = re.compile(r'maximum|limit|min|recommended|capacity', re.IGNORECASE)

    for key, value in details.items():
        if exclude_pattern.search(str(key)) or not weight_pattern.search(str(key)):
            continue

        if isinstance(value, str):
            match = re.search(r'([\d.]+)\s*(lbs?|pounds?|kg|kilograms?|oz|ounces?)?', value, re.IGNORECASE)
            if match:
                return convert_to_lbs(float(match.group(1)), match.group(2))
        elif isinstance(value, (int, float)):
            return float(value)

    return None

#Extract weight data from product details.
def process_weights(data):    
    return pd.DataFrame([
        {"SKU": item.get("SKU", "No SKU"), "Weight_lbs": extract_weight(item.get("Details", {}))}
        for item in data
    ])



def main():
    
    # Load data
    json_path = "/Users/vishwa/Desktop/products_labeled.json"
    data = load_json(json_path)
    df = pd.json_normalize(data).fillna('')  # Normalize and handle missing values

    
    analyze_price_distribution(df) #skewness

    # Load spaCy model
    nlp = spacy.load("en_core_web_md", disable=["parser", "ner"])  

    # Process text data
    df = process_text_columns(df, nlp)

    # Extract dimensions and compute volume
    df = extract_dimensions(df)

    # Process weight extraction
    df_weights = process_weights(data)
    df = df.merge(df_weights, on="SKU", how="left")

    # Select relevant final columns
    final_df = df[['Manufacturer', 'SKU', 'Price', 'Category', 'Weight_lbs', 'Processed_Text', 'Volume']]

    # Save processed data
    save_to_csv(final_df, "final_spacy.csv")


# Execute main function
if __name__ == "__main__":
    main()


