import json
import pandas as pd

data_list = []
# Open the JSON file with UTF-8 encoding
with open('secure_programming_dpo.json', 'r', encoding='utf-8') as file:
    for line in file:  # Read line by line for multiple JSON objects
        data_list.append(json.loads(line.strip()))

# Combine all parsed JSON objects into a DataFrame
df = pd.json_normalize(data_list)

# Save DataFrame to CSV
df.to_csv('c++.csv', index=False)

print("JSON file has been converted to CSV and saved as output.csv")

