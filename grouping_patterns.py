import json
import pandas as pd
import re
from collections import defaultdict

with open("/Users/vishwa/Desktop/products_labeled.json", "r") as file:
    data = json.load(file)
# Grouping categories with regex patterns
patterns = {
    "Dimensions": r".*Dimensions.*",  # Keys that contain the word "Dimensions"
    "Weight": r".*Weight.*",          # Keys that contain the word "Weight"
    "Brand": r".*Brand.*|.*Manufacturer.*",  # Keys related to Brand or Manufacturer
    "Size": r".*Size.*",              # Keys related to Size
    "Model": r".*Model.*",            # Keys related to Model
    "Style": r".*Style.*",            # Keys related to Style
    "Components": r".*Components.*",  # Keys related to Components
    "Rank": r".*Rank.*",              # Keys related to Rank
    "Number of Items": r".*Number of Items.*",  # Keys related to Number of Items
    "Color": r".*Color.*",            # Keys related to Color
    "Material": r".*Material.*",      # Keys related to Material
    "Suggested Users": r".*Suggested Users.*",  # Keys related to Suggested Users
}

# Dictionary to hold counts of all keys from the 'Details' sections
all_keys = defaultdict(int)

# Iterate over each dictionary (product) in the dataset
for dictionary in data:
    # Check if 'Details' key exists
    if 'Details' in dictionary:
        # Count occurrences of each key in the 'Details' dictionary
        for key in dictionary['Details'].keys():
            all_keys[key] += 1

# Group keys based on regex patterns
grouped_data = {group: [] for group in patterns.keys()}

# Automatically group keys based on regex patterns and count occurrences
for key, count in all_keys.items():
    for group, pattern in patterns.items():
        if re.match(pattern, key):
            grouped_data.setdefault(group, []).append((key, count))

# Calculate total number of occurrences
total_count = sum(all_keys.values())

# Sort the items in each group based on the count (from high to low)
for group, keys in grouped_data.items():
    grouped_data[group] = sorted(keys, key=lambda x: x[1], reverse=True)

# Print the grouped data with occurrences and sorted by count
for group, keys in grouped_data.items():
    if keys:
        print(f"{group}:")
        for key, count in keys:
            print(f"  {key} (Occurrences: {count})")
        print()

# Print total count of items
print(f"Total count of all items: {total_count}")

# Optionally, save the grouped data with occurrences and sorted by count to a file
with open("grouped_details_keys_with_counts_sorted.txt", "w") as f:
    for group, keys in grouped_data.items():
        if keys:
            f.write(f"{group}:\n")
            for key, count in keys:
                f.write(f"  {key} (Occurrences: {count})\n")
            f.write("\n")
    f.write(f"Total count of all items: {total_count}\n")

print("Grouped 'Details' keys with occurrences (sorted) saved to 'grouped_details_keys_with_counts_sorted.txt'.")
